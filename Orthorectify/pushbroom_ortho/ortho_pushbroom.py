#!/usr/bin/env python3
"""
Pushbroom (Linear-Array) Orthorectification Pipeline
=====================================================

Bottom-up (output-to-input) approach: for every pixel in the output
orthophoto grid, iteratively solve for the exact acquisition time *t*
at which the sensor observed that ground coordinate, then resample the
raw BIL strip.

Mathematical pipeline
---------------------
1.  Define output UTM / local grid from **swath footprint** (not full DSM).
2.  For each output pixel  (E, N)  →  sample DSM  →  (E, N, Z_ell)
3.  **Pre-filter**: reject ground points too far from the flight line
    (cannot be within the sensor swath).
4.  Transform to ECEF.
5.  Iteratively find scanline time *t* via Newton's method:
      – Interpolate SBET pose at *t*  (Slerp for rotations, cubic for pos).
      – Apply lever-arm  →  camera ECEF position.
      – Rotate ground-to-camera vector into the camera frame.
      – Minimise along-track angle  y_cam / z_cam  →  0.
6.  **Post-filter**: reject points where Newton residual is not truly zero.
7.  Project cross-track angle through Brown–Conrady distortion  →  u.
8.  Map *t*  →  line index  v  via the exposure-time LUT.
9.  Bilinear-resample the BIL at  (u, v)  and write the output GeoTIFF.

Anti-curtaining measures
------------------------
-  **Option 2 (spatial pre-filter)**: before Newton, reject DSM cells whose
   2-D distance to the nearest trajectory nadir exceeds the maximum
   geometric half-swath width.  Cheap, removes ~80 % of invalid work.
-  **Option 1 (residual post-filter)**: after Newton, check that the
   absolute along-track residual is truly near zero, not just that the
   iteration stopped.

Georeferencing improvements
---------------------------
-  Configurable ``time_offset`` (seconds) added to exposure times to
   absorb any PPS-to-mid-exposure latency, readout delay, or systematic
   shift between GPS time scales.
-  Output grid is clipped to the projected swath footprint so the
   GeoTIFF only covers the actual image coverage, not the full DSM.

Dependencies
------------
numpy, scipy, pyproj, spectral, rasterio, pyyaml, tqdm
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio
import spectral
import yaml
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

# ── WGS-84 constants ──────────────────────────────────────────────────
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257_223_563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2


###############################################################################
# 1.  SBET READER  (Applanix NED binary – 17 float64 fields = 136 B/record)
###############################################################################

SBET_DTYPE = np.dtype(
    [
        ("time", "f8"),
        ("lat", "f8"),         # radians
        ("lon", "f8"),         # radians
        ("alt", "f8"),         # ellipsoidal height  (m)
        ("x_vel", "f8"),
        ("y_vel", "f8"),
        ("z_vel", "f8"),
        ("roll", "f8"),        # radians, right-wing-down +
        ("pitch", "f8"),       # radians, nose-up +
        ("heading", "f8"),     # radians, CW from North
        ("wander", "f8"),
        ("x_ang_rate", "f8"),
        ("y_ang_rate", "f8"),
        ("z_ang_rate", "f8"),
        ("x_accel", "f8"),
        ("y_accel", "f8"),
        ("z_accel", "f8"),
    ]
)


def read_sbet(path: str | Path) -> np.ndarray:
    """Read an Applanix SBET binary file into a structured array."""
    data = np.fromfile(str(path), dtype=SBET_DTYPE)
    if data.size == 0:
        raise ValueError(f"Empty or unreadable SBET: {path}")
    return data


###############################################################################
# 2.  COORDINATE HELPERS
###############################################################################

def geodetic_to_ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray):
    """Vectorised geodetic (rad) → ECEF (m)."""
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * slat ** 2)
    x = (N + alt) * clat * clon
    y = (N + alt) * clat * slon
    z = (N * (1.0 - WGS84_E2) + alt) * slat
    return x, y, z


def ecef2ned_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Rotation matrices ECEF → local NED.

    Parameters
    ----------
    lat, lon : (N,) arrays in **radians**

    Returns
    -------
    R : (N, 3, 3)  such that  v_ned = R @ v_ecef
    """
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    N = lat.shape[0]
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = -slat * clon
    R[:, 0, 1] = -slat * slon
    R[:, 0, 2] = clat
    R[:, 1, 0] = -slon
    R[:, 1, 1] = clon
    # R[:, 1, 2] = 0
    R[:, 2, 0] = -clat * clon
    R[:, 2, 1] = -clat * slon
    R[:, 2, 2] = -slat
    return R


def rpy_to_matrix(roll: np.ndarray, pitch: np.ndarray, heading: np.ndarray):
    """
    Batch Euler  (roll, pitch, heading)  →  R_ned2body.

    Convention: intrinsic ZYX  –  Rz(heading) · Ry(pitch) · Rx(roll)
    which rotates a vector from the NED frame into the body frame.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)

    R = np.empty((*roll.shape, 3, 3))
    R[..., 0, 0] = ch * cp
    R[..., 0, 1] = sh * cp
    R[..., 0, 2] = -sp
    R[..., 1, 0] = ch * sp * sr - sh * cr
    R[..., 1, 1] = sh * sp * sr + ch * cr
    R[..., 1, 2] = cp * sr
    R[..., 2, 0] = ch * sp * cr + sh * sr
    R[..., 2, 1] = sh * sp * cr - ch * sr
    R[..., 2, 2] = cp * cr
    return R


###############################################################################
# 3.  TRAJECTORY INTERPOLATOR  (cubic pos + Slerp orientation)
###############################################################################

class TrajectoryInterpolator:
    """Interpolates an SBET trajectory at arbitrary GPS times.

    Positions are interpolated as cubic splines in ECEF.
    Orientations (R_ned→body) are interpolated via Slerp on the rotation
    manifold, which is the correct way to interpolate rotation matrices
    (avoids gimbal-lock artefacts that plague Euler-angle interpolation).
    """

    def __init__(self, sbet: np.ndarray):
        self.times = sbet["time"].copy()

        # ── geodetic ──
        self.lat = sbet["lat"]
        self.lon = sbet["lon"]
        self.alt = sbet["alt"]

        # ── ECEF positions ──
        ex, ey, ez = geodetic_to_ecef(self.lat, self.lon, self.alt)
        self._ip_x = interp1d(self.times, ex, kind="cubic", assume_sorted=True)
        self._ip_y = interp1d(self.times, ey, kind="cubic", assume_sorted=True)
        self._ip_z = interp1d(self.times, ez, kind="cubic", assume_sorted=True)

        # ── geodetic interps (for R_ecef2ned at query time) ──
        self._ip_lat = interp1d(self.times, self.lat, kind="cubic", assume_sorted=True)
        self._ip_lon = interp1d(self.times, self.lon, kind="cubic", assume_sorted=True)

        # ── Slerp for R_ned2body ──
        mats = rpy_to_matrix(sbet["roll"], sbet["pitch"], sbet["heading"])
        self._slerp = Slerp(self.times, Rotation.from_matrix(mats))

        # ── useful scalars ──
        self.t_min = float(self.times[0])
        self.t_max = float(self.times[-1])
        self.mean_alt = float(np.mean(self.alt))

    # ── vectorised queries ─────────────────────────────────────────────
    def ecef_pos(self, t: np.ndarray) -> np.ndarray:
        """(N,) → (N, 3)  IMU ECEF positions."""
        return np.column_stack([self._ip_x(t), self._ip_y(t), self._ip_z(t)])

    def latlon(self, t: np.ndarray):
        """(N,) → (lat, lon) each (N,) in radians."""
        return self._ip_lat(t), self._ip_lon(t)

    def R_ned2body(self, t: np.ndarray) -> np.ndarray:
        """(N,) → (N, 3, 3)  rotation matrices NED → body."""
        return self._slerp(t).as_matrix()


###############################################################################
# 4.  BROWN–CONRADY CAMERA MODEL  (forward: undistorted → distorted pixel)
###############################################################################

class BrownConradyPushbroom:
    """Forward projection for a pushbroom linear array.

    Given normalised camera-frame direction cosines  (xn, yn)  =  (X/Z, Y/Z)
    this returns the distorted pixel coordinate *u* along the cross-track
    array.  *yn* is the along-track residual used for Newton convergence and
    is expected to be ≈ 0 after the time-solver converges.
    """

    def __init__(self, f: float, cx: float, k1: float, k2: float,
                 p1: float, p2: float):
        self.f = f
        self.cx = cx
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

    def project(self, xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
        """Return distorted pixel coordinate *u* (cross-track).

        Parameters
        ----------
        xn, yn : (N,) normalised coords  X_cam/Z_cam , Y_cam/Z_cam
        """
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4
        xd = xn * radial + 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn)
        return self.f * xd + self.cx


###############################################################################
# 5.  ORTHO ENGINE  –  core maths for one tile
###############################################################################

class OrthoEngine:
    """Holds all data / interpolators needed by a single worker process."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # ── camera ──
        cam = cfg["camera"]
        dist = cam["distortion"]
        self.camera = BrownConradyPushbroom(
            f=cam["focal_length_px"],
            cx=cam["principal_point_x"],
            k1=dist["k1"], k2=dist["k2"],
            p1=dist["p1"], p2=dist["p2"],
        )
        self.sensor_width = cam["sensor_width_px"]

        # ── mounting / boresight ──
        mnt = cfg["mounting"]
        R_mount = np.array(mnt["matrix"], dtype=np.float64)
        br = mnt["boresight_roll"]
        bp = mnt["boresight_pitch"]
        by = mnt["boresight_yaw"]
        R_boresight = Rotation.from_euler("ZYX", [by, bp, br]).as_matrix()
        self.R_body2cam = R_boresight @ R_mount        # (3, 3)
        self.lever_arm = np.array(mnt["lever_arm"], dtype=np.float64)  # body frame

        # ── trajectory ──
        sbet = read_sbet(cfg["inputs"]["sbet_path"])
        self.traj = TrajectoryInterpolator(sbet)

        # ── exposure times (one GPS second per BIL line) ──
        # Apply configurable time offset for PPS-to-mid-exposure latency
        self.time_offset = cfg.get("georef", {}).get("time_offset", 0.0)
        raw_times = np.loadtxt(cfg["inputs"]["exposure_times_path"],
                               dtype=np.float64).ravel()
        self.line_times = raw_times + self.time_offset
        self.n_lines = self.line_times.size

        # ── build time → line index interpolator  and  line → time ──
        line_idx = np.arange(self.n_lines, dtype=np.float64)
        self._time2line = interp1d(self.line_times, line_idx, kind="linear",
                                   bounds_error=False, fill_value=-1.0)
        self._line2time = interp1d(line_idx, self.line_times, kind="linear",
                                   bounds_error=False, fill_value="extrapolate")

        # ── BIL image (memory-mapped via spectral) ──
        hdr_path = str(Path(cfg["inputs"]["bil_path"]).with_suffix(".hdr"))
        self.bil_img = spectral.open_image(hdr_path)
        self.bil_mm = self.bil_img.open_memmap(writable=False)
        self.n_bands = self.bil_mm.shape[2]

        # ── DSM (rasterio) ──
        self.dsm_ds = rasterio.open(cfg["inputs"]["dsm_path"])
        self.dsm_data = self.dsm_ds.read(1)
        self.dsm_transform = self.dsm_ds.transform
        self.dsm_nodata = self.dsm_ds.nodata

        # ── pyproj transformers ──
        crs_cfg = cfg["crs"]
        crs_out = CRS(crs_cfg["output"])
        crs_dsm = CRS(crs_cfg["dsm"])
        crs_ecef = CRS("EPSG:4978")                     # WGS-84 ECEF

        self.tf_out2ecef = Transformer.from_crs(crs_out, crs_ecef, always_xy=True)
        self.tf_out2dsm = Transformer.from_crs(crs_out, crs_dsm, always_xy=True)

        # ── solver params ──
        proc = cfg["processing"]
        self.max_iter = proc["max_iterations"]
        self.tol = proc["convergence_tol"]
        self.nodata = proc["nodata"]

        # ── compute mean height above ground for swath estimate ──
        valid_dsm = self.dsm_data[self.dsm_data != self.dsm_nodata] if self.dsm_nodata else self.dsm_data.ravel()
        self.mean_dsm_z = float(np.nanmean(valid_dsm)) if valid_dsm.size else 0.0
        self.mean_hag = max(self.traj.mean_alt - self.mean_dsm_z, 100.0)

        # ── precompute ground-track of sensor for fast initial guess ──
        self._build_ground_track()

        # ── precompute maximum swath half-width for pre-filter ──
        half_fov = np.arctan(
            self.sensor_width * 0.5 / self.camera.f
        )
        self.max_half_swath = self.mean_hag * np.tan(half_fov)
        # add generous buffer (20%) to account for terrain variation
        # and off-nadir viewing due to roll
        self.swath_buffer = cfg.get("processing", {}).get("swath_buffer", 1.2)
        self.max_ground_dist = self.max_half_swath * self.swath_buffer
        print(f"  Half-FOV: {np.degrees(half_fov):.2f}°  |  "
              f"Max half-swath: {self.max_half_swath:.1f} m  |  "
              f"Pre-filter radius: {self.max_ground_dist:.1f} m")

    # ------------------------------------------------------------------
    def _build_ground_track(self):
        """Project sensor positions into the output CRS for initial guess
        and swath pre-filter."""
        # subsample to ~1000 points for speed
        step = max(1, self.n_lines // 1000)
        idx = np.arange(0, self.n_lines, step)
        times_sub = self.line_times[idx]

        lat, lon = self.traj.latlon(times_sub)
        # geodetic (rad) → degrees for pyproj
        tf_geo2out = Transformer.from_crs(
            CRS("EPSG:4326"), CRS(self.cfg["crs"]["output"]), always_xy=True
        )
        e, n = tf_geo2out.transform(np.degrees(lon), np.degrees(lat))

        self._gt_times = times_sub
        self._gt_e = e
        self._gt_n = n

    def _initial_time_guess(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        """Nearest-ground-track initial time estimate for a batch of points.

        Returns
        -------
        t0 : (N,) float64  GPS time initial guesses clamped to valid range.
        """
        de = self._gt_e[:, None] - east[None, :]
        dn = self._gt_n[:, None] - north[None, :]
        d2 = de * de + dn * dn
        best = np.argmin(d2, axis=0)
        t0 = self._gt_times[best]
        return np.clip(t0, self.traj.t_min + 1e-6, self.traj.t_max - 1e-6)

    def _swath_distance(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        """Compute minimum horizontal distance from each point to the
        ground track (in output CRS metres).

        Parameters
        ----------
        east, north : (N,) output-CRS coordinates

        Returns
        -------
        dist : (N,) minimum 2-D distance to any ground-track sample (m)
        """
        de = self._gt_e[:, None] - east[None, :]   # (M, N)
        dn = self._gt_n[:, None] - north[None, :]
        d2 = de * de + dn * dn                     # (M, N)
        return np.sqrt(np.min(d2, axis=0))          # (N,)

    # ------------------------------------------------------------------
    def _apply_lever_arm(self, imu_ecef: np.ndarray,
                         R_ned2body: np.ndarray,
                         R_ecef2ned: np.ndarray) -> np.ndarray:
        """Shift from IMU to camera projection-centre in ECEF.

        cam_ecef = imu_ecef + R_ned2ecef @ R_body2ned @ lever_arm_body

        Parameters
        ----------
        imu_ecef   : (N, 3)
        R_ned2body : (N, 3, 3)
        R_ecef2ned : (N, 3, 3)

        Returns
        -------
        cam_ecef   : (N, 3)
        """
        # R_body2ned = R_ned2body^T, R_ned2ecef = R_ecef2ned^T
        # lever_ned  = R_body2ned @ lever  →  (N,3)
        lever_ned = np.einsum("nji,j->ni", R_ned2body, self.lever_arm)
        lever_ecef = np.einsum("nji,nj->ni", R_ecef2ned, lever_ned)
        return imu_ecef + lever_ecef

    # ------------------------------------------------------------------
    def solve_batch(
        self,
        ground_ecef: np.ndarray,
        t_init: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Iteratively solve for (u, v) image coords of a batch of ground pts.

        Returns
        -------
        u       : (N,) cross-track pixel
        v       : (N,) along-track line index (float for sub-pixel resample)
        valid   : (N,) bool  – True where solver converged inside image bounds
        residual: (N,) final along-track residual (for diagnostics / post-filter)
        """
        N = ground_ecef.shape[0]
        t = t_init.copy()
        dt_num = 1e-5   # finite-difference step for Jacobian (seconds)
        residual = np.full(N, np.inf)

        for _ in range(self.max_iter):
            # ── interpolate pose at t ──
            imu_ecef = self.traj.ecef_pos(t)             # (N,3)
            lat, lon = self.traj.latlon(t)                # (N,), (N,)
            R_n2b = self.traj.R_ned2body(t)               # (N,3,3)
            R_e2n = ecef2ned_matrix(lat, lon)             # (N,3,3)

            cam_ecef = self._apply_lever_arm(imu_ecef, R_n2b, R_e2n)

            # ── ground vector in camera frame ──
            d_ecef = ground_ecef - cam_ecef               # (N,3)
            d_ned = np.einsum("nij,nj->ni", R_e2n, d_ecef)
            d_body = np.einsum("nij,nj->ni", R_n2b, d_ned)
            d_cam = np.einsum("ij,nj->ni", self.R_body2cam, d_body)

            # residual = along-track angle
            residual = d_cam[:, 1] / d_cam[:, 2]         # should → 0

            # ── check convergence ──
            if np.all(np.abs(residual) < self.tol):
                break

            # ── numerical Jacobian via forward finite difference ──
            t_p = t + dt_num
            t_p = np.clip(t_p, self.traj.t_min + 1e-6, self.traj.t_max - 1e-6)

            imu_p = self.traj.ecef_pos(t_p)
            lat_p, lon_p = self.traj.latlon(t_p)
            R_n2b_p = self.traj.R_ned2body(t_p)
            R_e2n_p = ecef2ned_matrix(lat_p, lon_p)
            cam_p = self._apply_lever_arm(imu_p, R_n2b_p, R_e2n_p)

            d_ecef_p = ground_ecef - cam_p
            d_ned_p = np.einsum("nij,nj->ni", R_e2n_p, d_ecef_p)
            d_body_p = np.einsum("nij,nj->ni", R_n2b_p, d_ned_p)
            d_cam_p = np.einsum("ij,nj->ni", self.R_body2cam, d_body_p)

            residual_p = d_cam_p[:, 1] / d_cam_p[:, 2]
            dr_dt = (residual_p - residual) / dt_num

            # ── Newton update (with damping / safeguard) ──
            safe = np.abs(dr_dt) > 1e-15
            delta = np.where(safe, residual / dr_dt, 0.0)
            # clamp step to ± 0.5 s to keep iterations stable
            delta = np.clip(delta, -0.5, 0.5)
            t = t - delta
            t = np.clip(t, self.traj.t_min + 1e-6, self.traj.t_max - 1e-6)

        # ── project to pixel coords ──
        xn = d_cam[:, 0] / d_cam[:, 2]
        yn = d_cam[:, 1] / d_cam[:, 2]
        u = self.camera.project(xn, yn)
        v = self._time2line(t)

        # ── validity mask ──────────────────────────────────────────────
        # OPTION 1 (post-filter): absolute residual must be truly near zero.
        # The old check  (< tol * 10)  let through points where the solver
        # stalled at a non-zero residual (= not actually on the focal plane).
        # Use a separate, tighter tolerance here.
        residual_tol = self.cfg.get("processing", {}).get(
            "residual_tolerance", self.tol * 100
        )
        converged = np.abs(residual) < residual_tol

        # Point must be in front of camera
        in_front = d_cam[:, 2] > 0

        # Cross-track pixel within sensor
        u_valid = (u >= 0) & (u < self.sensor_width)

        # Along-track line within image
        v_valid = (v >= 0) & (v < self.n_lines)

        valid = converged & in_front & u_valid & v_valid

        return u, v, valid, residual

    # ------------------------------------------------------------------
    def sample_dsm(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        """Sample DSM heights at output-CRS coordinates.

        Transforms (E, N) from output CRS into DSM CRS, then does
        nearest-neighbour lookup.  Returns np.nan where outside coverage.
        """
        # transform output CRS → DSM CRS
        dsm_e, dsm_n = self.tf_out2dsm.transform(east, north)

        inv_tf = ~self.dsm_transform
        col_f, row_f = inv_tf * (dsm_e, dsm_n)
        col = np.round(col_f).astype(int)
        row = np.round(row_f).astype(int)

        h, w = self.dsm_data.shape
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        z = np.full(east.shape, np.nan, dtype=np.float64)
        z[inside] = self.dsm_data[row[inside], col[inside]]

        if self.dsm_nodata is not None:
            z[z == self.dsm_nodata] = np.nan
        return z

    # ------------------------------------------------------------------
    def resample_bil(self, u: np.ndarray, v: np.ndarray,
                     valid: np.ndarray) -> np.ndarray:
        """Bilinear resample the BIL at sub-pixel (u, v).

        Returns
        -------
        pixels : (N, n_bands)  with nodata where ~valid.
        """
        N = u.shape[0]
        out = np.full((N, self.n_bands), self.nodata, dtype=np.float32)
        if not np.any(valid):
            return out

        uv = u[valid]
        vv = v[valid]

        for b in range(self.n_bands):
            band = self.bil_mm[:, :, b].astype(np.float32)
            coords = np.vstack([vv, uv])    # (2, M)  – row, col ordering
            sampled = map_coordinates(band, coords, order=1, mode="constant",
                                      cval=self.nodata)
            out[valid, b] = sampled

        return out

    # ------------------------------------------------------------------
    def process_tile(
        self,
        tile_row0: int, tile_col0: int,
        tile_rows: int, tile_cols: int,
        origin_e: float, origin_n: float,
        gsd: float,
    ) -> tuple[int, int, int, int, np.ndarray]:
        """Orthorectify one tile of the output grid.

        Returns
        -------
        (row0, col0, rows, cols, data)  where  data : (rows, cols, bands)
        """
        rows = np.arange(tile_rows) + tile_row0
        cols = np.arange(tile_cols) + tile_col0

        cc, rr = np.meshgrid(cols, rows)           # (tile_rows, tile_cols)
        east = origin_e + cc.ravel() * gsd
        north = origin_n - rr.ravel() * gsd        # image row ↓ = north ↓

        n_px = east.size
        result = np.full((n_px, self.n_bands), self.nodata, dtype=np.float32)

        # ── sample DSM ──
        z = self.sample_dsm(east, north)
        has_z = np.isfinite(z)

        if not np.any(has_z):
            return (tile_row0, tile_col0, tile_rows, tile_cols,
                    result.reshape(tile_rows, tile_cols, self.n_bands))

        # ────────────────────────────────────────────────────────────────
        # OPTION 2: SPATIAL PRE-FILTER  –  reject DSM cells too far from
        # the flight line.  A ground point that is farther than the
        # geometric half-swath from any trajectory nadir position can
        # never project onto the sensor array.
        # ────────────────────────────────────────────────────────────────
        e_z = east[has_z]
        n_z = north[has_z]
        z_z = z[has_z]

        ground_dist = self._swath_distance(e_z, n_z)
        in_swath = ground_dist < self.max_ground_dist

        if not np.any(in_swath):
            return (tile_row0, tile_col0, tile_rows, tile_cols,
                    result.reshape(tile_rows, tile_cols, self.n_bands))

        # Build index mapping:  has_z → in_swath subset
        idx_hz = np.where(has_z)[0]       # indices into full tile
        idx_sw = idx_hz[in_swath]         # final subset indices

        e_sub = e_z[in_swath]
        n_sub = n_z[in_swath]
        z_sub = z_z[in_swath]

        # ── output CRS → ECEF ──
        xe, ye, ze = self.tf_out2ecef.transform(e_sub, n_sub, z_sub)
        ground_ecef = np.column_stack([xe, ye, ze])

        # ── initial time guess ──
        t0 = self._initial_time_guess(e_sub, n_sub)

        # ── solve (now returns residual too) ──
        u, v, valid, residual = self.solve_batch(ground_ecef, t0)

        # ── resample ──
        px = self.resample_bil(u, v, valid)

        result[idx_sw] = px
        data = result.reshape(tile_rows, tile_cols, self.n_bands)
        return (tile_row0, tile_col0, tile_rows, tile_cols, data)


###############################################################################
# 6.  WORKER INITIALISER & ENTRY POINT  (ProcessPoolExecutor)
###############################################################################

_engine: OrthoEngine | None = None          # per-worker global


def _init_worker(cfg: dict):
    """Called once per worker process to build the OrthoEngine."""
    global _engine
    _engine = OrthoEngine(cfg)


def _run_tile(args: tuple) -> tuple:
    """Tile worker function – pickleable because it uses the global engine."""
    return _engine.process_tile(*args)


###############################################################################
# 7.  MAIN ORCHESTRATOR
###############################################################################

def compute_swath_footprint(cfg: dict):
    """Compute the ground footprint of the image swath in the output CRS.

    Projects the sensor ground-track and expands it by the geometric
    half-swath width.  Returns a bounding box that tightly encloses
    only the area the sensor actually covers.

    Returns
    -------
    left, bottom, right, top : swath bounding box in output CRS (m)
    """
    sbet = read_sbet(cfg["inputs"]["sbet_path"])
    traj = TrajectoryInterpolator(sbet)

    time_offset = cfg.get("georef", {}).get("time_offset", 0.0)
    line_times = np.loadtxt(cfg["inputs"]["exposure_times_path"],
                            dtype=np.float64).ravel() + time_offset
    n_lines = line_times.size

    # Subsample trajectory for speed
    step = max(1, n_lines // 500)
    idx = np.arange(0, n_lines, step)
    times_sub = line_times[idx]
    times_sub = np.clip(times_sub, traj.t_min + 1e-6, traj.t_max - 1e-6)

    lat, lon = traj.latlon(times_sub)

    tf_geo2out = Transformer.from_crs(
        CRS("EPSG:4326"), CRS(cfg["crs"]["output"]), always_xy=True
    )
    e, n = tf_geo2out.transform(np.degrees(lon), np.degrees(lat))

    # Compute FOV-based half-swath
    cam = cfg["camera"]
    sensor_width = cam["sensor_width_px"]
    focal_length = cam["focal_length_px"]
    half_fov = np.arctan(sensor_width * 0.5 / focal_length)

    # Mean terrain height
    with rasterio.open(cfg["inputs"]["dsm_path"]) as ds:
        dsm_data = ds.read(1)
        dsm_nodata = ds.nodata
    valid_dsm = dsm_data[dsm_data != dsm_nodata] if dsm_nodata else dsm_data.ravel()
    mean_dsm = float(np.nanmean(valid_dsm)) if valid_dsm.size else 0.0
    mean_hag = max(traj.mean_alt - mean_dsm, 100.0)

    half_swath = mean_hag * np.tan(half_fov)
    buffer = cfg.get("processing", {}).get("swath_buffer", 1.2)
    margin = half_swath * buffer

    left = np.min(e) - margin
    right = np.max(e) + margin
    bottom = np.min(n) - margin
    top = np.max(n) + margin

    return left, bottom, right, top


def compute_output_grid(cfg: dict):
    """Derive output raster extent from swath footprint intersected with DSM.

    The grid covers only the area the sensor actually images, clipped
    to DSM coverage.

    Returns
    -------
    origin_e, origin_n : top-left corner in output CRS
    n_cols, n_rows     : output dimensions (pixels)
    gsd                : ground sample distance (m)
    """
    gsd = cfg["outputs"]["gsd"]

    # ── DSM bounds in output CRS ──
    with rasterio.open(cfg["inputs"]["dsm_path"]) as ds:
        dsm_bounds = ds.bounds
        dsm_crs = ds.crs

    crs_dsm = CRS(dsm_crs)
    crs_out = CRS(cfg["crs"]["output"])

    if crs_dsm == crs_out:
        dsm_left, dsm_bottom, dsm_right, dsm_top = dsm_bounds
    else:
        tf = Transformer.from_crs(crs_dsm, crs_out, always_xy=True)
        xs = [dsm_bounds.left, dsm_bounds.right,
              dsm_bounds.left, dsm_bounds.right]
        ys = [dsm_bounds.bottom, dsm_bounds.bottom,
              dsm_bounds.top, dsm_bounds.top]
        ox, oy = tf.transform(xs, ys)
        dsm_left, dsm_right = min(ox), max(ox)
        dsm_bottom, dsm_top = min(oy), max(oy)

    # ── Swath footprint bounds ──
    sw_left, sw_bottom, sw_right, sw_top = compute_swath_footprint(cfg)

    # ── Intersect: output grid covers swath ∩ DSM ──
    left   = max(dsm_left,   sw_left)
    right  = min(dsm_right,  sw_right)
    bottom = max(dsm_bottom, sw_bottom)
    top    = min(dsm_top,    sw_top)

    if left >= right or bottom >= top:
        raise ValueError(
            f"No overlap between DSM extent and swath footprint.\n"
            f"  DSM:   E[{dsm_left:.1f}, {dsm_right:.1f}]  "
            f"N[{dsm_bottom:.1f}, {dsm_top:.1f}]\n"
            f"  Swath: E[{sw_left:.1f}, {sw_right:.1f}]  "
            f"N[{sw_bottom:.1f}, {sw_top:.1f}]"
        )

    n_cols = int(np.ceil((right - left) / gsd))
    n_rows = int(np.ceil((top - bottom) / gsd))
    origin_e = left
    origin_n = top

    print(f"  DSM extent:   E[{dsm_left:.1f}, {dsm_right:.1f}]  "
          f"N[{dsm_bottom:.1f}, {dsm_top:.1f}]")
    print(f"  Swath extent: E[{sw_left:.1f}, {sw_right:.1f}]  "
          f"N[{sw_bottom:.1f}, {sw_top:.1f}]")
    print(f"  Output grid:  E[{left:.1f}, {right:.1f}]  "
          f"N[{bottom:.1f}, {top:.1f}]")

    return origin_e, origin_n, n_cols, n_rows, gsd


def generate_tiles(n_rows: int, n_cols: int, tile_size: int,
                   origin_e: float, origin_n: float, gsd: float):
    """Yield (row0, col0, rows, cols, origin_e, origin_n, gsd) for each tile."""
    for r0 in range(0, n_rows, tile_size):
        rn = min(tile_size, n_rows - r0)
        for c0 in range(0, n_cols, tile_size):
            cn = min(tile_size, n_cols - c0)
            yield (r0, c0, rn, cn, origin_e, origin_n, gsd)


def main(config_path: str = "config.yaml"):
    # ── load config ──
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── derive output grid (now swath-clipped) ──
    origin_e, origin_n, n_cols, n_rows, gsd = compute_output_grid(cfg)
    print(f"Output grid: {n_cols} × {n_rows}  GSD={gsd} m")
    print(f"Origin (E, N): ({origin_e:.2f}, {origin_n:.2f})")

    # ── figure out number of bands from the BIL header ──
    hdr_path = str(Path(cfg["inputs"]["bil_path"]).with_suffix(".hdr"))
    bil_meta = spectral.open_image(hdr_path)
    n_bands = bil_meta.nbands
    print(f"Bands: {n_bands}")

    # ── prepare output GeoTIFF ──
    out_path = cfg["outputs"]["ortho_path"]
    crs_out = CRS(cfg["crs"]["output"])
    transform = rasterio.transform.from_origin(origin_e, origin_n, gsd, gsd)
    nodata = cfg["processing"]["nodata"]

    dst = rasterio.open(
        out_path, "w", driver="GTiff",
        height=n_rows, width=n_cols, count=n_bands,
        dtype="float32", crs=crs_out.to_wkt(),
        transform=transform, nodata=nodata,
        compress="deflate", tiled=True,
        blockxsize=256, blockysize=256,
    )

    # ── generate tile jobs ──
    tile_size = cfg["processing"]["tile_size"]
    tiles = list(generate_tiles(n_rows, n_cols, tile_size,
                                origin_e, origin_n, gsd))
    n_workers = cfg["processing"]["num_workers"]
    print(f"Tiles: {len(tiles)}   Workers: {n_workers}")

    # ── process tiles in parallel ──
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(cfg,),
    ) as pool:
        futures = pool.map(_run_tile, tiles, chunksize=1)
        for r0, c0, rn, cn, data in tqdm(futures, total=len(tiles),
                                          desc="Orthorectifying"):
            # data shape: (rn, cn, bands)
            for b in range(n_bands):
                band_data = data[:, :, b]
                window = rasterio.windows.Window(c0, r0, cn, rn)
                dst.write(band_data, b + 1, window=window)

    dst.close()
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg_file)
