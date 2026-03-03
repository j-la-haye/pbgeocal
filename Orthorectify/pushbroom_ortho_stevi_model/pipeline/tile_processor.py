"""
tile_processor.py — Parallel tile-based orchestration.

Divides the output orthophoto grid into tiles and dispatches them to
worker processes using concurrent.futures.ProcessPoolExecutor.  Each
worker independently runs the full bottom-up pipeline (Newton + resample)
for its tile.

Because multiprocessing requires picklable arguments, this module
serialises the shared data into a form that workers can reconstruct
(file paths + config), and uses numpy memory-mapped files for the BIL
to enable zero-copy shared access.

Tile scheduling:
  • Tiles are ordered by row then column (scanline order)
  • tqdm provides a real-time progress bar with ETA
  • Results are written to the output GeoTIFF as tiles complete

Memory strategy:
  • The BIL image is memory-mapped (read-only, shared across forks)
  • DSM is loaded per-worker (relatively small)
  • SBET trajectory is loaded once and trimmed to the exposure time window
"""

import numpy as np
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from osgeo import gdal
from pathlib import Path

from .config_loader import PipelineConfig
from .sbet_reader import read_sbet, trim_sbet
from .trajectory import Trajectory
from .camera_model import PushbroomCamera
from .dsm_handler import DSMHandler
from .time_solver import TimeSolver
from .visibility import VisibilityChecker
from .ortho_engine import process_tile, TileResult

import spectral
import spectral.io.envi as envi


def _load_bil_memmap(bil_path: str, hdr_path: str):
    """
    Load ENVI BIL as a numpy memory-mapped array.

    Uses python spectral to read metadata, then creates a raw memmap.

    Returns
    -------
    data   : np.memmap  (n_lines, n_bands, n_samples)  for BIL interleave
    n_lines, n_bands, n_samples : dimensions
    dtype  : numpy dtype of the data
    """
    img = envi.open(hdr_path, bil_path)
    meta = img.metadata

    n_lines = int(meta['lines'])
    n_samples = int(meta['samples'])
    n_bands = int(meta['bands'])

    # Map ENVI data type to numpy dtype
    envi_dtype_map = {
        '1': np.uint8, '2': np.int16, '3': np.int32,
        '4': np.float32, '5': np.float64,
        '12': np.uint16, '13': np.uint32, '14': np.int64, '15': np.uint64,
    }
    dtype = envi_dtype_map.get(str(meta.get('data type', '4')), np.float32)

    header_offset = int(meta.get('header offset', 0))

    # BIL interleave: row-major order is (lines, bands, samples)
    data = np.memmap(
        bil_path,
        dtype=dtype,
        mode='r',
        offset=header_offset,
        shape=(n_lines, n_bands, n_samples),
    )

    return data, n_lines, n_bands, n_samples, dtype


def _create_output_geotiff(output_path: str, ncols: int, nrows: int,
                           n_bands: int, x_origin: float, y_origin: float,
                           gsd: float, epsg: int, nodata: float):
    """
    Create the output GeoTIFF with proper CRS and geotransform.
    """
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        output_path, ncols, nrows, n_bands,
        gdal.GDT_Float32,
        options=[
            'COMPRESS=DEFLATE',
            'TILED=YES',
            'BLOCKXSIZE=256',
            'BLOCKYSIZE=256',
            'BIGTIFF=YES',
        ],
    )

    # GeoTransform: (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
    ds.SetGeoTransform((x_origin, gsd, 0.0, y_origin, 0.0, -gsd))

    from osgeo import osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())

    for b in range(1, n_bands + 1):
        band = ds.GetRasterBand(b)
        band.SetNoDataValue(nodata)
        band.Fill(nodata)

    ds.FlushCache()
    return ds


# =====================================================================
# Worker initialiser — each process loads its own copies of heavy objects
# =====================================================================

# Global per-worker state (set by _worker_init)
_worker_state = {}


def _worker_init(config_dict: dict):
    """
    Initialise per-worker state: trajectory, DSM, camera, solver, visibility.

    Called once per worker process by ProcessPoolExecutor initializer.
    """
    from .config_loader import load_config
    import yaml

    # Reconstruct config from dict (passed as serialisable data)
    cfg = config_dict['config_path']
    config = load_config(cfg)

    # SBET
    sbet = read_sbet(config.paths.sbet)

    # Exposure times: text file, one GPS time per line, in 10 μs (1e-5 s) units
    raw_times = np.loadtxt(config.paths.exposure_times, dtype=np.float64)
    exposure_times = raw_times * 1.0e-5  # convert to GPS seconds

    sbet = trim_sbet(sbet, exposure_times[0], exposure_times[-1], margin=2.0)

    # Trajectory
    trajectory = Trajectory(sbet, config.mounting)
    trajectory.build_line_index(exposure_times)

    # Camera — Steviapp pinhole + polynomial distortion model
    camera = PushbroomCamera(
        f=config.camera.focal_length,
        ppx=config.camera.principal_point,
        w=config.camera.detector_width,
        delta_x_coeffs=config.camera.distortion.delta_x,
        delta_y_coeffs=config.camera.distortion.delta_y,
        first_valid_pixel=config.camera.first_valid_pixel,
        angle_lut_path=config.camera.angle_lut_path,
    )

    # DSM
    dsm = DSMHandler(config.paths.dsm, config.crs.dsm_epsg, config.crs.output_epsg)

    # Time solver
    solver = TimeSolver(trajectory, camera, config.processing.newton, exposure_times)

    # Visibility
    vis = VisibilityChecker(dsm, config.processing.visibility)

    # BIL memmap
    bil_data, n_lines, n_bands, n_samples, _ = _load_bil_memmap(
        config.paths.bil_image, config.paths.bil_header
    )

    _worker_state.update({
        'config': config,
        'trajectory': trajectory,
        'camera': camera,
        'dsm': dsm,
        'solver': solver,
        'visibility': vis,
        'bil_data': bil_data,
        'bil_shape': (n_lines, n_bands, n_samples),
        'n_bands': n_bands,
    })


def _worker_process_tile(args: dict) -> dict:
    """
    Worker function: process one tile using per-worker state.

    Parameters
    ----------
    args : dict with row_start, col_start, tile_rows, tile_cols,
           x_coords, y_coords

    Returns
    -------
    dict with row_start, col_start, data (serialised)
    """
    ws = _worker_state
    cfg = ws['config']

    result = process_tile(
        row_start=args['row_start'],
        col_start=args['col_start'],
        tile_rows=args['tile_rows'],
        tile_cols=args['tile_cols'],
        x_coords=args['x_coords'],
        y_coords=args['y_coords'],
        dsm=ws['dsm'],
        trajectory=ws['trajectory'],
        camera=ws['camera'],
        time_solver=ws['solver'],
        visibility_checker=ws['visibility'],
        bil_data=ws['bil_data'],
        bil_shape=ws['bil_shape'],
        n_bands=ws['n_bands'],
        resampling=cfg.processing.resampling,
        nodata=cfg.processing.nodata,
    )

    return {
        'row_start': result.row_start,
        'col_start': result.col_start,
        'data': result.data,
    }


# =====================================================================
# Main orchestrator
# =====================================================================

def run_parallel_ortho(config: PipelineConfig, config_path: str):
    """
    Main entry point: orchestrate tile-parallel orthorectification.

    Parameters
    ----------
    config      : PipelineConfig object
    config_path : path to the YAML config (for worker re-loading)
    """
    print("=" * 70)
    print("  Pushbroom Orthorectification Pipeline")
    print("  Bottom-Up Approach with Newton Iteration")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Load metadata and define output grid
    # -----------------------------------------------------------------
    print("\n[1/5] Loading inputs...")

    # Exposure times: text file, one per line, in 10 μs (1e-5 s) units
    raw_times = np.loadtxt(config.paths.exposure_times, dtype=np.float64)
    exposure_times = raw_times * 1.0e-5  # convert to GPS seconds
    n_scanlines = len(exposure_times)
    print(f"  Exposure times: {n_scanlines} lines, "
          f"t=[{exposure_times[0]:.3f}, {exposure_times[-1]:.3f}] GPS s")

    # BIL metadata
    _, n_lines, n_bands, n_samples, bil_dtype = _load_bil_memmap(
        config.paths.bil_image, config.paths.bil_header
    )
    print(f"  BIL image: {n_lines} lines × {n_samples} samples × {n_bands} bands "
          f"({bil_dtype.__name__})")

    assert n_lines == n_scanlines, (
        f"BIL lines ({n_lines}) ≠ exposure times ({n_scanlines})"
    )

    # DSM
    dsm = DSMHandler(config.paths.dsm, config.crs.dsm_epsg, config.crs.output_epsg)

    # Output grid
    gsd = config.processing.output_gsd
    x_coords, y_coords, x_origin, y_origin, ncols, nrows = dsm.output_grid(gsd)
    print(f"  Output grid: {ncols}×{nrows} pixels at {gsd} m GSD")

    # -----------------------------------------------------------------
    # 2. Create output GeoTIFF
    # -----------------------------------------------------------------
    print("\n[2/5] Creating output GeoTIFF...")
    output_path = config.paths.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    out_ds = _create_output_geotiff(
        output_path, ncols, nrows, n_bands,
        x_origin, y_origin, gsd,
        config.crs.output_epsg, config.processing.nodata,
    )
    print(f"  → {output_path}")

    # -----------------------------------------------------------------
    # 3. Generate tile schedule
    # -----------------------------------------------------------------
    print("\n[3/5] Generating tile schedule...")
    tile_size = config.processing.tile_size
    tiles = []
    for r in range(0, nrows, tile_size):
        for c in range(0, ncols, tile_size):
            tr = min(tile_size, nrows - r)
            tc = min(tile_size, ncols - c)
            tiles.append({
                'row_start': r,
                'col_start': c,
                'tile_rows': tr,
                'tile_cols': tc,
                'x_coords': x_coords,
                'y_coords': y_coords,
            })
    print(f"  {len(tiles)} tiles of {tile_size}×{tile_size}")

    # -----------------------------------------------------------------
    # 4. Process tiles in parallel
    # -----------------------------------------------------------------
    print(f"\n[4/5] Processing tiles ({config.processing.num_workers} workers)...")

    init_args = {'config_path': config_path}

    with ProcessPoolExecutor(
        max_workers=config.processing.num_workers,
        initializer=_worker_init,
        initargs=(init_args,),
    ) as executor:
        futures = {
            executor.submit(_worker_process_tile, tile): tile
            for tile in tiles
        }

        with tqdm(total=len(tiles), desc="Orthorectifying",
                  unit="tile", ncols=80) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    _write_tile_to_geotiff(
                        out_ds, result, n_bands
                    )
                except Exception as e:
                    tile_info = futures[future]
                    print(f"\n  ⚠ Tile ({tile_info['row_start']}, "
                          f"{tile_info['col_start']}) failed: {e}",
                          file=sys.stderr)
                finally:
                    pbar.update(1)

    # -----------------------------------------------------------------
    # 5. Finalise
    # -----------------------------------------------------------------
    print("\n[5/5] Finalising output...")
    out_ds.FlushCache()
    out_ds = None

    print(f"\n✓ Orthophoto written to: {output_path}")
    print(f"  {ncols}×{nrows} pixels, {n_bands} bands, {gsd} m GSD")


def _write_tile_to_geotiff(ds, result: dict, n_bands: int):
    """
    Write a processed tile into the output GeoTIFF.
    """
    data = result['data']      # (tr, tc, n_bands)
    r0 = result['row_start']
    c0 = result['col_start']
    tr, tc = data.shape[0], data.shape[1]

    for b in range(n_bands):
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(data[:, :, b], c0, r0)
