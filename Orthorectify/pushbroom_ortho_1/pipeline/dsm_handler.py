"""
dsm_handler.py — Load a DSM GeoTIFF and provide elevation queries.

Loads the Digital Surface Model (in ellipsoidal height) and provides:
  • Grid extent and resolution metadata
  • Bilinear interpolation of Z at arbitrary (X, Y) coordinates
  • Coordinate transformation from DSM CRS to ECEF (via pyproj)

The DSM must be a single-band GeoTIFF with ellipsoidal heights.
"""

import numpy as np
from pathlib import Path
from pyproj import Transformer, CRS
from osgeo import gdal

gdal.UseExceptions()


class DSMHandler:
    """
    Digital Surface Model handler.

    Parameters
    ----------
    dsm_path   : path to the GeoTIFF
    dsm_epsg   : EPSG code of the DSM CRS
    """

    def __init__(self, dsm_path: str, dsm_epsg: int):
        path = Path(dsm_path)
        if not path.exists():
            raise FileNotFoundError(f"DSM not found: {dsm_path}")

        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL failed to open: {dsm_path}")

        self.band = ds.GetRasterBand(1)
        self.data = self.band.ReadAsArray().astype(np.float64)
        self.nodata = self.band.GetNoDataValue()
        gt = ds.GetGeoTransform()
        ds = None  # close

        # GeoTransform: (x_origin, x_res, 0, y_origin, 0, y_res)
        # y_res is typically negative (north-up)
        self.x_origin = gt[0]
        self.x_res = gt[1]
        self.y_origin = gt[3]
        self.y_res = gt[5]    # negative for north-up
        self.n_rows, self.n_cols = self.data.shape

        # Extent in DSM CRS
        self.x_min = self.x_origin
        self.x_max = self.x_origin + self.n_cols * self.x_res
        self.y_max = self.y_origin                                  # top
        self.y_min = self.y_origin + self.n_rows * self.y_res       # bottom (y_res < 0)

        # CRS info
        self.epsg = dsm_epsg
        self.crs = CRS.from_epsg(dsm_epsg)

        # Transformer: DSM CRS → ECEF (WGS84 geocentric)
        self._to_ecef = Transformer.from_crs(
            self.crs,
            CRS.from_epsg(4978),    # WGS-84 ECEF
            always_xy=True,
        )

        # Transformer: DSM CRS → WGS-84 geographic (for lat/lon queries)
        self._to_lonlat = Transformer.from_crs(
            self.crs,
            CRS.from_epsg(4326),
            always_xy=True,
        )

        # Mean DSM height (for initial guess computations)
        valid = self.data.copy()
        if self.nodata is not None:
            valid[valid == self.nodata] = np.nan
        self.mean_height = float(np.nanmean(valid))

        print(f"  DSM loaded: {self.n_cols}×{self.n_rows}, "
              f"res={self.x_res:.2f} m, "
              f"Z range=[{np.nanmin(valid):.1f}, {np.nanmax(valid):.1f}] m")

    # ------------------------------------------------------------------
    # Grid coordinate queries
    # ------------------------------------------------------------------
    def xy_to_colrow(self, x: np.ndarray, y: np.ndarray):
        """Map-space (x, y) → continuous (col, row) indices."""
        col = (x - self.x_origin) / self.x_res
        row = (y - self.y_origin) / self.y_res
        return col, row

    def get_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Bilinear interpolation of DSM elevation at (x, y) in DSM CRS.

        Parameters
        ----------
        x, y : coordinate arrays (same shape)

        Returns
        -------
        z : interpolated elevation; NaN where out-of-bounds or nodata
        """
        col, row = self.xy_to_colrow(x, y)

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

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------
    def to_ecef(self, x: np.ndarray, y: np.ndarray,
                z: np.ndarray) -> np.ndarray:
        """
        Convert DSM CRS coordinates to ECEF.

        Parameters
        ----------
        x, y, z : arrays in DSM CRS (e.g. UTM easting, northing, ell. height)

        Returns
        -------
        ecef : (..., 3) ECEF coordinates
        """
        ex, ey, ez = self._to_ecef.transform(x, y, z)
        return np.stack([ex, ey, ez], axis=-1)

    def output_grid(self, gsd: float):
        """
        Generate the output orthoimage grid aligned to the DSM extent.

        Parameters
        ----------
        gsd : ground sample distance [m]

        Returns
        -------
        x_coords : (ncols,) easting values
        y_coords : (nrows,) northing values  (top to bottom)
        x_origin : left edge
        y_origin : top edge
        ncols, nrows : grid dimensions
        """
        x_coords = np.arange(self.x_min, self.x_max, gsd)
        y_coords = np.arange(self.y_max, self.y_min, -gsd)   # top to bottom
        ncols = len(x_coords)
        nrows = len(y_coords)

        return x_coords, y_coords, self.x_min, self.y_max, ncols, nrows
