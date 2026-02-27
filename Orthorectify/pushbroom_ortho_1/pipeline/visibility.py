"""
visibility.py — Occlusion detection for true ortho generation.

For each output pixel, checks whether the line of sight from the sensor
to the ground point is obstructed by intervening terrain (the DSM).

Algorithm
---------
1. Given the sensor position S and ground point P (both in DSM CRS + Z),
   parameterise the 3D ray:  R(α) = S + α·(P - S),  α ∈ [0, 1].
2. Sample the ray at `num_samples` evenly spaced α values in (0, 1)
   (excluding endpoints).
3. At each sample, compute the horizontal position (x, y) and the ray
   height z_ray.
4. Query the DSM for z_dsm at (x, y).
5. If z_dsm > z_ray + tolerance at any sample, the ground point is
   **occluded** (hidden behind a taller feature).

This is a conservative test: it catches buildings, ridges, and other
features that block the sensor's view.  The tolerance parameter prevents
false positives from minor DSM noise.

Vectorised implementation processes all points in a tile simultaneously.
"""

import numpy as np
from pyproj import Transformer, CRS
from .dsm_handler import DSMHandler
from .config_loader import VisibilityConfig


class VisibilityChecker:
    """
    Ray-based occlusion detector.

    Parameters
    ----------
    dsm     : DSMHandler with elevation data
    config  : VisibilityConfig (num_ray_samples, height_tolerance)
    """

    def __init__(self, dsm: DSMHandler, config: VisibilityConfig):
        self.dsm = dsm
        self.num_samples = config.num_ray_samples
        self.tolerance = config.height_tolerance
        self.enabled = config.enabled

        # Transformer: ECEF → DSM CRS (for sensor position conversion)
        self._from_ecef = Transformer.from_crs(
            CRS.from_epsg(4978),     # WGS-84 ECEF
            dsm.crs,
            always_xy=True,
        )

    def check(self, sensor_ecef: np.ndarray,
              ground_x: np.ndarray, ground_y: np.ndarray,
              ground_z: np.ndarray) -> np.ndarray:
        """
        Check visibility for a batch of ground points.

        Parameters
        ----------
        sensor_ecef : (M, 3) sensor positions in ECEF at solved times
        ground_x    : (M,)   ground X in DSM CRS (easting)
        ground_y    : (M,)   ground Y in DSM CRS (northing)
        ground_z    : (M,)   ground Z (ellipsoidal height from DSM)

        Returns
        -------
        visible : (M,) boolean — True if ground point is visible from sensor
        """
        if not self.enabled:
            return np.ones(len(ground_x), dtype=bool)

        M = len(ground_x)

        # Convert sensor ECEF → DSM CRS
        sx, sy, sz = self._from_ecef.transform(
            sensor_ecef[:, 0], sensor_ecef[:, 1], sensor_ecef[:, 2]
        )

        # Ray parameterisation: sample α ∈ (0, 1), excluding endpoints
        # α = 0 is at sensor, α = 1 is at ground
        alphas = np.linspace(0.05, 0.95, self.num_samples)  # (S,)

        visible = np.ones(M, dtype=bool)

        # Process in alpha-steps to limit memory (M × S could be large)
        for alpha in alphas:
            # Interpolated position along ray
            rx = sx + alpha * (ground_x - sx)
            ry = sy + alpha * (ground_y - sy)
            rz = sz + alpha * (ground_z - sz)   # ray height at this alpha

            # DSM height at this horizontal position
            z_dsm = self.dsm.get_z(rx, ry)

            # Occluded if DSM is above the ray (with tolerance)
            occluded = z_dsm > (rz + self.tolerance)

            # Handle NaN DSM values (outside coverage) — treat as not occluding
            occluded = np.where(np.isnan(z_dsm), False, occluded)

            # Update visibility mask
            visible &= ~occluded

        return visible
