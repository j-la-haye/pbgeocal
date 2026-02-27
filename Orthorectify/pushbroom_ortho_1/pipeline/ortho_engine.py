"""
ortho_engine.py — Core orthorectification logic for a single tile.

Each tile is a rectangular sub-region of the output orthophoto grid.
The engine performs the full bottom-up pipeline for every pixel in the tile:

    ┌─────────────┐
    │ Output grid  │  (x, y) in output CRS
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ DSM lookup   │  z = DSM(x, y)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ CRS → ECEF  │  (x, y, z) → ECEF
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ Newton iter  │  find time t* for each ground point
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ Projection   │  (u, v) distorted pixel in raw BIL
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ Visibility   │  occlusion check along sensor→ground ray
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ Resampling   │  bilinear sample from BIL at (scanline, sample)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ Output tile  │  (tile_rows, tile_cols, n_bands)
    └─────────────┘

The engine is designed to be called from a multiprocessing worker.
All heavy objects (trajectory, DSM, BIL memmap) are passed or
reconstructed per-worker.
"""

import numpy as np
from typing import Tuple, Optional


class TileResult:
    """Result container for a processed tile."""
    def __init__(self, row_start: int, col_start: int,
                 data: np.ndarray):
        self.row_start = row_start
        self.col_start = col_start
        self.data = data   # (tile_rows, tile_cols, n_bands)


def process_tile(
    row_start: int, col_start: int,
    tile_rows: int, tile_cols: int,
    x_coords: np.ndarray,           # (total_cols,) easting
    y_coords: np.ndarray,           # (total_rows,) northing
    dsm,                             # DSMHandler
    trajectory,                      # Trajectory
    camera,                          # PushbroomCamera
    time_solver,                     # TimeSolver
    visibility_checker,              # VisibilityChecker
    bil_data: np.ndarray,            # memory-mapped BIL (lines, bands, samples) or (lines, samples, bands)
    bil_shape: tuple,                # (n_lines, n_bands, n_samples) for BIL interleave
    n_bands: int,
    resampling: str,
    nodata: float,
) -> TileResult:
    """
    Process a single tile of the output orthophoto.

    Parameters
    ----------
    row_start, col_start : top-left corner of this tile in the output grid
    tile_rows, tile_cols : dimensions of this tile
    x_coords, y_coords  : full output grid coordinates
    dsm, trajectory, camera, time_solver, visibility_checker : pipeline objects
    bil_data      : memory-mapped raw BIL image data
    bil_shape     : (n_lines, n_bands, n_samples) — BIL interleave order
    n_bands       : number of spectral bands
    resampling    : "bilinear" or "nearest"
    nodata        : nodata fill value

    Returns
    -------
    TileResult with the rendered tile data
    """
    # Slice grid coordinates for this tile
    xs = x_coords[col_start:col_start + tile_cols]       # (tc,)
    ys = y_coords[row_start:row_start + tile_rows]       # (tr,)

    # Create meshgrid  (row-major: y varies along axis 0)
    xx, yy = np.meshgrid(xs, ys)   # (tr, tc) each
    flat_x = xx.ravel()             # (M,)
    flat_y = yy.ravel()             # (M,)
    M = len(flat_x)

    # Initialise output tile
    output = np.full((tile_rows, tile_cols, n_bands), nodata, dtype=np.float32)

    # -----------------------------------------------------------------
    # 1. DSM lookup
    # -----------------------------------------------------------------
    flat_z = dsm.get_z(flat_x, flat_y)

    # Mask invalid DSM
    valid_dsm = ~np.isnan(flat_z)
    if not np.any(valid_dsm):
        return TileResult(row_start, col_start, output)

    # Work only with valid DSM points
    idx_valid = np.where(valid_dsm)[0]
    vx = flat_x[idx_valid]
    vy = flat_y[idx_valid]
    vz = flat_z[idx_valid]

    # -----------------------------------------------------------------
    # 2. Convert to ECEF
    # -----------------------------------------------------------------
    ground_ecef = dsm.to_ecef(vx, vy, vz)  # (V, 3)

    # -----------------------------------------------------------------
    # 3. Initial time guess
    # -----------------------------------------------------------------
    t_init = trajectory.initial_time_guess(ground_ecef)

    # -----------------------------------------------------------------
    # 4. Newton iteration
    # -----------------------------------------------------------------
    t_solved, u_pixel, y_lab, line_idx, solve_valid = time_solver.solve(
        ground_ecef, t_init
    )

    # -----------------------------------------------------------------
    # 5. Visibility check
    # -----------------------------------------------------------------
    vis_mask = np.ones(len(vx), dtype=bool)
    if visibility_checker.enabled:
        # Get sensor positions at solved times (only for valid solves)
        sensor_ecef = np.zeros((len(vx), 3), dtype=np.float64)
        if np.any(solve_valid):
            pos_cam, _ = trajectory.interpolate_batch(t_solved[solve_valid])
            sensor_ecef[solve_valid] = pos_cam
            vis_sub = visibility_checker.check(
                sensor_ecef[solve_valid],
                vx[solve_valid], vy[solve_valid], vz[solve_valid]
            )
            vis_mask[solve_valid] = vis_sub

    # Combined validity
    final_valid = solve_valid & vis_mask

    if not np.any(final_valid):
        return TileResult(row_start, col_start, output)

    # -----------------------------------------------------------------
    # 6. Resample from BIL
    # -----------------------------------------------------------------
    # BIL interleave: data shape = (n_lines, n_bands, n_samples)
    n_lines, _, n_samples = bil_shape

    fv = idx_valid[final_valid]
    fu = u_pixel[final_valid]       # across-track (sample)
    fl = line_idx[final_valid]      # along-track (fractional line)

    if resampling == "bilinear":
        pixel_values = _bilinear_sample_bil(
            bil_data, fl, fu, n_lines, n_bands, n_samples, nodata
        )
    else:
        pixel_values = _nearest_sample_bil(
            bil_data, fl, fu, n_lines, n_bands, n_samples, nodata
        )

    # Map back to tile grid
    tile_row_idx = fv // tile_cols
    tile_col_idx = fv % tile_cols
    output[tile_row_idx, tile_col_idx, :] = pixel_values

    return TileResult(row_start, col_start, output)


def _bilinear_sample_bil(bil_data: np.ndarray,
                         line: np.ndarray, sample: np.ndarray,
                         n_lines: int, n_bands: int, n_samples: int,
                         nodata: float) -> np.ndarray:
    """
    Bilinear resampling from BIL-interleaved data.

    BIL layout: (n_lines, n_bands, n_samples)
      - line  → axis 0 (along-track / scanline index)
      - band  → axis 1
      - sample → axis 2 (across-track pixel)

    Parameters
    ----------
    bil_data : (n_lines, n_bands, n_samples) memory-mapped array
    line     : (P,) fractional scanline indices
    sample   : (P,) fractional across-track pixel indices

    Returns
    -------
    values : (P, n_bands) interpolated spectral values
    """
    P = len(line)

    l0 = np.floor(line).astype(np.int64)
    s0 = np.floor(sample).astype(np.int64)
    dl = line - l0
    ds = sample - s0

    # Clamp to valid range
    l0 = np.clip(l0, 0, n_lines - 2)
    s0 = np.clip(s0, 0, n_samples - 2)

    # Four neighbours: (l0,s0), (l0,s0+1), (l0+1,s0), (l0+1,s0+1)
    # BIL: bil_data[line, :, sample]
    v00 = bil_data[l0, :, s0].astype(np.float32)          # (P, n_bands)
    v01 = bil_data[l0, :, s0 + 1].astype(np.float32)
    v10 = bil_data[l0 + 1, :, s0].astype(np.float32)
    v11 = bil_data[l0 + 1, :, s0 + 1].astype(np.float32)

    # Bilinear weights
    dl2 = dl[:, None]   # (P, 1)
    ds2 = ds[:, None]

    values = ((1 - dl2) * ((1 - ds2) * v00 + ds2 * v01) +
              dl2 * ((1 - ds2) * v10 + ds2 * v11))

    return values


def _nearest_sample_bil(bil_data: np.ndarray,
                        line: np.ndarray, sample: np.ndarray,
                        n_lines: int, n_bands: int, n_samples: int,
                        nodata: float) -> np.ndarray:
    """
    Nearest-neighbour sampling from BIL data.
    """
    l0 = np.round(line).astype(np.int64)
    s0 = np.round(sample).astype(np.int64)
    l0 = np.clip(l0, 0, n_lines - 1)
    s0 = np.clip(s0, 0, n_samples - 1)

    return bil_data[l0, :, s0].astype(np.float32)
