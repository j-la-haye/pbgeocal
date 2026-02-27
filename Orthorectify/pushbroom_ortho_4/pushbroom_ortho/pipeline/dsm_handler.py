"""
dsm_handler.py — DSM handler with proper CRS separation.

Handles the case where the DSM CRS is geographic (e.g. EPSG:4326/4979)
and the output CRS is projected/metric (e.g. EPSG:32632 UTM).

Key design decisions:
  • All horizontal CRS transforms use the 2D variant of the CRS
    (via .to_2d()) so that 3D geographic CRSs like EPSG:4979 work
    correctly — pyproj can silently produce wrong results when a
    3D CRS receives only 2 coordinates.
  • The output CRS must be projected (metric) — a geographic output
    CRS with a metric GSD is physically meaningless.
  • ECEF transforms always include height.

Two coordinate domains:
  • DSM native CRS:  the CRS of the GeoTIFF file (geographic or projected)
  • Output CRS:      the CRS of the orthoimage grid (must be projected/metric)

Methods are explicitly named by which CRS they expect:
  • get_z(x, y)          — output CRS in,  Z out  (for ortho_engine)
  • get_z_native(x, y)   — DSM CRS in,     Z out  (for visibility)
  • to_ecef(x, y, z)     — output CRS in,  ECEF out
  • output_grid(gsd)     — builds metric grid in output CRS
  • output_to_native()   — output CRS → DSM native CRS

The DSM must be a single-band GeoTIFF with ellipsoidal heights.
"""

import numpy as np
from pathlib import Path
from pyproj import Transformer, CRS
from osgeo import gdal

gdal.UseExceptions()


class DSMHandler:
    """
    Digital Surface Model handler with dual-CRS support.

    Parameters
    ----------
    dsm_path    : path to the GeoTIFF
    dsm_epsg    : EPSG code of the DSM CRS (geographic or projected)
    output_epsg : EPSG code of the output orthoimage CRS (must be projected)
    """

    def __init__(self, dsm_path: str, dsm_epsg: int, output_epsg: int):
        path = Path(dsm_path)
        if not path.exists():
            raise FileNotFoundError(f"DSM not found: {dsm_path}")

        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL failed to open: {dsm_path}")

        band = ds.GetRasterBand(1)
        self.data = band.ReadAsArray().astype(np.float64)
        self.nodata = band.GetNoDataValue()
        gt = ds.GetGeoTransform()
        ds = None  # close

        # GeoTransform: (x_origin, x_res, 0, y_origin, 0, y_res)
        # For geographic CRS: x = longitude, y = latitude (GDAL convention)
        # y_res is typically negative (north-up)
        self.x_origin = gt[0]
        self.x_res = gt[1]
        self.y_origin = gt[3]
        self.y_res = gt[5]    # negative for north-up
        self.n_rows, self.n_cols = self.data.shape

        # Extent in DSM native CRS
        self.x_min_native = self.x_origin
        self.x_max_native = self.x_origin + self.n_cols * self.x_res
        self.y_max_native = self.y_origin                             # top
        self.y_min_native = self.y_origin + self.n_rows * self.y_res  # bottom

        # ------------------------------------------------------------------
        # CRS setup — always use 2D horizontal CRS for grid transforms
        # ------------------------------------------------------------------
        self.dsm_epsg = dsm_epsg
        self.output_epsg = output_epsg

        dsm_crs_raw = CRS.from_epsg(dsm_epsg)
        output_crs_raw = CRS.from_epsg(output_epsg)

        # Extract the 2D horizontal component (EPSG:4979 → EPSG:4326, etc.)
        # This ensures pyproj transforms work correctly with (x, y) only.
        self.dsm_crs = dsm_crs_raw.to_2d()
        self.output_crs = output_crs_raw.to_2d()

        self.dsm_is_geographic = self.dsm_crs.is_geographic
        self.output_is_geographic = self.output_crs.is_geographic

        # Validate: output CRS must be projected (metric GSD requires metres)
        if self.output_is_geographic:
            raise ValueError(
                f"Output CRS EPSG:{output_epsg} is geographic (degrees). "
                f"A projected CRS (e.g. UTM) is required for metric GSD. "
                f"Set crs.output_epsg to a projected EPSG code."
            )

        # Are the 2D horizontal CRSs the same?
        self.same_crs = self.dsm_crs.equals(self.output_crs)

        # Horizontal transformers (2D only, safe for geographic CRS)
        if not self.same_crs:
            self._output_to_native = Transformer.from_crs(
                self.output_crs, self.dsm_crs, always_xy=True,
            )
            self._native_to_output = Transformer.from_crs(
                self.dsm_crs, self.output_crs, always_xy=True,
            )
        else:
            self._output_to_native = None
            self._native_to_output = None

        # ECEF transforms — use the original (possibly 3D) CRS with heights
        self._output_to_ecef = Transformer.from_crs(
            output_crs_raw, CRS.from_epsg(4978), always_xy=True,
        )
        self._native_to_ecef = Transformer.from_crs(
            dsm_crs_raw, CRS.from_epsg(4978), always_xy=True,
        )

        # ------------------------------------------------------------------
        # Compute extent in output CRS (projected, metric)
        # ------------------------------------------------------------------
        # Sample along all four edges of the DSM, not just corners,
        # to handle projection curvature over large extents
        n_edge = 20
        edge_x = np.concatenate([
            np.linspace(self.x_min_native, self.x_max_native, n_edge),  # bottom
            np.linspace(self.x_min_native, self.x_max_native, n_edge),  # top
            np.full(n_edge, self.x_min_native),                         # left
            np.full(n_edge, self.x_max_native),                         # right
        ])
        edge_y = np.concatenate([
            np.full(n_edge, self.y_min_native),                         # bottom
            np.full(n_edge, self.y_max_native),                         # top
            np.linspace(self.y_min_native, self.y_max_native, n_edge),  # left
            np.linspace(self.y_min_native, self.y_max_native, n_edge),  # right
        ])

        if not self.same_crs:
            ox, oy = self._native_to_output.transform(edge_x, edge_y)
        else:
            ox, oy = edge_x, edge_y

        self.x_min_output = float(np.min(ox))
        self.x_max_output = float(np.max(ox))
        self.y_min_output = float(np.min(oy))
        self.y_max_output = float(np.max(oy))

        # ------------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------------
        valid = self.data.copy()
        if self.nodata is not None:
            valid[valid == self.nodata] = np.nan
        self.mean_height = float(np.nanmean(valid))

        res_label = (f"{self.x_res:.6f}°" if self.dsm_is_geographic
                     else f"{self.x_res:.2f} m")
        print(f"  DSM loaded: {self.n_cols}×{self.n_rows}, "
              f"res={res_label}, "
              f"Z range=[{np.nanmin(valid):.1f}, {np.nanmax(valid):.1f}] m")
        if not self.same_crs:
            dsm_label = "geographic" if self.dsm_is_geographic else "projected"
            print(f"  DSM CRS: EPSG:{dsm_epsg} ({dsm_label}) "
                  f"→ output CRS: EPSG:{output_epsg}")
            extent_x = self.x_max_output - self.x_min_output
            extent_y = self.y_max_output - self.y_min_output
            print(f"  Output extent: {extent_x:.1f} × {extent_y:.1f} m")

    # ==================================================================
    # Coordinate transforms
    # ==================================================================

    def output_to_native(self, x: np.ndarray, y: np.ndarray):
        """
        Transform output CRS coordinates to DSM native CRS.

        Parameters
        ----------
        x, y : coordinates in output CRS (easting, northing)

        Returns
        -------
        nx, ny : coordinates in DSM native CRS
        """
        if self.same_crs:
            return x, y
        return self._output_to_native.transform(x, y)

    def native_to_output(self, x: np.ndarray, y: np.ndarray):
        """
        Transform DSM native CRS coordinates to output CRS.
        """
        if self.same_crs:
            return x, y
        return self._native_to_output.transform(x, y)

    # ==================================================================
    # Grid coordinate queries (DSM native CRS)
    # ==================================================================

    def _xy_to_colrow(self, x: np.ndarray, y: np.ndarray):
        """Map-space (x, y) in DSM native CRS → continuous (col, row)."""
        col = (x - self.x_origin) / self.x_res
        row = (y - self.y_origin) / self.y_res
        return col, row

    def _bilinear_z(self, col: np.ndarray, row: np.ndarray) -> np.ndarray:
        """
        Bilinear interpolation of DSM elevation at fractional (col, row).

        Returns
        -------
        z : interpolated elevation; NaN where out-of-bounds or nodata
        """
        c0 = np.floor(col).astype(int)
        r0 = np.floor(row).astype(int)
        dc = col - c0
        dr = row - r0

        # Bounds check
        valid = ((c0 >= 0) & (c0 < self.n_cols - 1) &
                 (r0 >= 0) & (r0 < self.n_rows - 1))

        # Clamp for safe indexing (will be masked anyway)
        c0c = np.clip(c0, 0, self.n_cols - 2)
        r0c = np.clip(r0, 0, self.n_rows - 2)

        z00 = self.data[r0c, c0c]
        z01 = self.data[r0c, c0c + 1]
        z10 = self.data[r0c + 1, c0c]
        z11 = self.data[r0c + 1, c0c + 1]

        # Bilinear
        z = ((1 - dr) * ((1 - dc) * z00 + dc * z01) +
             dr * ((1 - dc) * z10 + dc * z11))

        # Mask nodata and out-of-bounds
        z = np.where(valid, z, np.nan)
        if self.nodata is not None:
            nodata_mask = ((z00 == self.nodata) | (z01 == self.nodata) |
                           (z10 == self.nodata) | (z11 == self.nodata))
            z = np.where(nodata_mask, np.nan, z)

        return z

    # ==================================================================
    # Z queries — two interfaces
    # ==================================================================

    def get_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Elevation lookup from OUTPUT CRS coordinates.

        Transforms (x, y) from output CRS to DSM native CRS, then
        performs bilinear interpolation on the DSM grid.

        This is the primary interface for ortho_engine.

        Parameters
        ----------
        x, y : coordinates in output CRS (easting, northing)

        Returns
        -------
        z : interpolated ellipsoidal height; NaN where invalid
        """
        nx, ny = self.output_to_native(x, y)
        col, row = self._xy_to_colrow(nx, ny)
        return self._bilinear_z(col, row)

    def get_z_native(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Elevation lookup from DSM NATIVE CRS coordinates.

        No coordinate transformation — queries the DSM directly.
        Used by visibility checker which works in DSM native CRS.

        Parameters
        ----------
        x, y : coordinates in DSM native CRS

        Returns
        -------
        z : interpolated ellipsoidal height; NaN where invalid
        """
        col, row = self._xy_to_colrow(x, y)
        return self._bilinear_z(col, row)

    # ==================================================================
    # ECEF conversions — two interfaces
    # ==================================================================

    def to_ecef(self, x: np.ndarray, y: np.ndarray,
                z: np.ndarray) -> np.ndarray:
        """
        Convert OUTPUT CRS coordinates to ECEF.

        Parameters
        ----------
        x, y, z : coordinates in output CRS + ellipsoidal height

        Returns
        -------
        ecef : (..., 3) ECEF coordinates
        """
        ex, ey, ez = self._output_to_ecef.transform(x, y, z)
        return np.stack([ex, ey, ez], axis=-1)

    def to_ecef_native(self, x: np.ndarray, y: np.ndarray,
                       z: np.ndarray) -> np.ndarray:
        """
        Convert DSM NATIVE CRS coordinates to ECEF.

        Parameters
        ----------
        x, y, z : coordinates in DSM native CRS + ellipsoidal height

        Returns
        -------
        ecef : (..., 3) ECEF coordinates
        """
        ex, ey, ez = self._native_to_ecef.transform(x, y, z)
        return np.stack([ex, ey, ez], axis=-1)

    # ==================================================================
    # Output grid generation (always in projected output CRS)
    # ==================================================================

    def output_grid(self, gsd: float):
        """
        Generate the output orthoimage grid in the OUTPUT CRS.

        Transforms the DSM extent to the projected output CRS and builds
        a regular metric grid at the requested GSD.  Works correctly
        regardless of whether the DSM is in geographic (degrees) or
        projected (metres) coordinates.

        Parameters
        ----------
        gsd : ground sample distance [m] (output CRS must be projected)

        Returns
        -------
        x_coords : (ncols,) easting values in output CRS
        y_coords : (nrows,) northing values in output CRS (top to bottom)
        x_origin : left edge (output CRS)
        y_origin : top edge  (output CRS)
        ncols, nrows : grid dimensions
        """
        # Snap grid origin to GSD multiples for clean alignment
        x_start = np.floor(self.x_min_output / gsd) * gsd
        y_end = np.floor(self.y_min_output / gsd) * gsd
        x_end = np.ceil(self.x_max_output / gsd) * gsd
        y_start = np.ceil(self.y_max_output / gsd) * gsd

        x_coords = np.arange(x_start, x_end, gsd)
        y_coords = np.arange(y_start, y_end, -gsd)   # top to bottom
        ncols = len(x_coords)
        nrows = len(y_coords)

        return x_coords, y_coords, x_start, y_start, ncols, nrows
