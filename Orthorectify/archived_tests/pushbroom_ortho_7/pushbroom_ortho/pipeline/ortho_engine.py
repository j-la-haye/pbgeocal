"""
ortho_engine.py — Core orthorectification logic for a single tile.

Each tile is a rectangular sub-region of the output orthophoto grid.
The engine performs the full bottom-up pipeline for every pixel in the tile:

    Output grid (x,y) in output CRS
        → DSM lookup (z)
        → CRS → ECEF
        → Per-scanline initial time guess
        → Newton iteration (find scanline time t*)
        → Camera projection (across-track pixel u, scanline index)
        → Visibility check (optional occlusion along sensor→ground ray)
        → Bilinear resample from BIL at (scanline, sample)
        → Output tile

Swath constraining
------------------
The image swath is constrained naturally by the Newton solver's validity
mask:  any ground point whose solved time maps to a scanline outside
[0, n_lines-1] or whose across-track pixel falls outside [0, num_pixels)
is rejected as invalid.  No explicit swath polygon or distance pre-filter
is needed.

Newton initialisation
---------------------
Uses per-scanline camera positions (trajectory.initial_time_from_lines)
which are interpolated at actual exposure times with lever arm correction.
This gives an initial guess within a few scanlines of the true solution.
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
    bil_data: np.ndarray,            # memory-mapped BIL (lines, bands, samples)
    bil_shape: tuple,                # (n_lines, n_bands, n_samples)
    n_bands: int,
    resampling: str,
    nodata: float,
) -> TileResult:
    """
    Process a single tile of the output orthophoto.
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
    # 3. Initial time guess from per-scanline camera positions
    # -----------------------------------------------------------------
    t_init = trajectory.initial_time_from_lines(ground_ecef)

    # -----------------------------------------------------------------
    # 4. Newton iteration — all valid DSM points go in
    #    The solver's validity mask handles swath constraining:
    #    points outside the image footprint are rejected by
    #    line_idx and pixel bounds checks.
    # -----------------------------------------------------------------
    t_solved, u_pixel, line_idx, solve_valid = time_solver.solve(
        ground_ecef, t_init
    )

    # -----------------------------------------------------------------
    # 5. Visibility check (only for valid solves)
    # -----------------------------------------------------------------
    if visibility_checker.enabled and np.any(solve_valid):
        pos_cam, _ = trajectory.interpolate_batch(t_solved[solve_valid])
        vis_sub = visibility_checker.check(
            pos_cam,
            vx[solve_valid], vy[solve_valid], vz[solve_valid]
        )
        # Fold visibility into solve_valid
        valid_indices = np.where(solve_valid)[0]
        solve_valid[valid_indices[~vis_sub]] = False

    # Final valid mask
    final_valid = solve_valid
    if not np.any(final_valid):
        return TileResult(row_start, col_start, output)

    # -----------------------------------------------------------------
    # 6. Resample from BIL
    # -----------------------------------------------------------------
    n_lines, _, n_samples = bil_shape

    fv = idx_valid[final_valid]     # indices into the flat output grid
    fu = camera.pixel_to_bil_sample(u_pixel[final_valid])  # LUT pixel → BIL sample
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
    """
    P = len(line)

    l0 = np.floor(line).astype(np.int64)
    s0 = np.floor(sample).astype(np.int64)
    dl = line - l0
    ds = sample - s0

    # Clamp to valid range
    l0 = np.clip(l0, 0, n_lines - 2)
    s0 = np.clip(s0, 0, n_samples - 2)

    # Four neighbours
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
