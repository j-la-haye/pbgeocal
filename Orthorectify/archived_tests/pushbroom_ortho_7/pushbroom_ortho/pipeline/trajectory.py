"""
trajectory.py — SBET trajectory interpolator with Slerp for attitude.

Provides interpolation of the 200 Hz SBET data at arbitrary times:
  • Position (lat, lon, alt → ECEF): linear interpolation
  • Attitude (roll, pitch, heading → R_{Body←NED}): quaternion Slerp
  • Full exterior orientation chain:
        R_{Cam←ECEF}(t) = R_{Cam←Body} @ R_{Body←NED}(t) @ R_{NED←ECEF}(t)

Per-scanline initialisation
---------------------------
For precise Newton solver initialisation, build_line_index() interpolates
the actual camera position (with lever arm) at each image exposure time.
initial_time_from_lines() then finds the nearest scanline by 3D distance
to the camera (not the IMU) for each ground point.

This is far superior to the old coarse SBET search because:
  • Uses camera position, not IMU position (lever arm corrected)
  • Searches only within the image time span (no SBET margin)
  • Resolution matches the actual scanline spacing (~1m at 50 m/s)
"""

import numpy as np
from .sbet_reader import SBETData
from .config_loader import MountingConfig
from . import coord_utils as cu


class Trajectory:
    """
    Interpolator for SBET trajectory with Slerp attitude.

    Pre-computes:
      - ECEF positions for all SBET records
      - R_{Body←NED} quaternions for all records (with continuity correction)
      - R_{Cam←Body} = R_boresight @ R_mounting (fixed)
      - Lever arm in body frame
    """

    def __init__(self, sbet: SBETData, mounting: MountingConfig):
        self.sbet = sbet
        self.times = sbet.time          # (N,) GPS seconds
        self.n_records = len(self.times)

        # Pre-compute ECEF positions
        self.pos_ecef = cu.geodetic_to_ecef(sbet.lat, sbet.lon, sbet.alt)  # (N, 3)

        # Pre-compute R_{Body←NED} for every record, then convert to quaternions
        self.R_body_ned = cu.euler_to_rotation_batch(
            sbet.roll, sbet.pitch, sbet.heading
        )  # (N, 3, 3)
        self.q_body_ned = cu.rotation_to_quaternion_batch(self.R_body_ned)  # (N, 4)

        # Pre-compute R_{NED←ECEF} quaternions for Slerp of orientation
        self.R_ned_ecef = cu.rotation_ecef_to_ned_batch(sbet.lat, sbet.lon)  # (N, 3, 3)
        self.q_ned_ecef = cu.rotation_to_quaternion_batch(self.R_ned_ecef)   # (N, 4)

        # Fixed camera-to-body rotation:  R_{Cam←Body} = R_boresight @ R_mounting
        R_boresight = cu.boresight_rotation(*mounting.boresight_rad)
        self.R_cam_body = R_boresight @ mounting.mounting_matrix  # (3, 3)

        # Lever arm in body frame [m]
        self.lever_arm = mounting.lever_arm.copy()  # (3,)

        # Time bounds
        self.t_min = self.times[0]
        self.t_max = self.times[-1]

        # Per-scanline index (populated by build_line_index)
        self._line_cam_ecef = None     # (K, 3)  camera ECEF per scanline
        self._line_times = None        # (K,)    exposure times (decimated)

    # =================================================================
    # Per-scanline camera position index
    # =================================================================

    def build_line_index(self, exposure_times: np.ndarray,
                         max_points: int = 5000):
        """
        Build a spatial index of camera positions at each scanline time.

        Interpolates the full exterior orientation (including lever arm)
        at each exposure time, so the search finds the nearest CAMERA
        position, not the nearest IMU position.

        Parameters
        ----------
        exposure_times : (L,) GPS times of each scanline
        max_points     : decimate to at most this many index entries
        """
        L = len(exposure_times)
        stride = max(1, L // max_points)

        # Decimated subset of scanlines
        idx = np.arange(0, L, stride)
        times_sub = exposure_times[idx]

        # Interpolate camera positions at these times
        # This applies the full lever arm correction automatically
        pos_sub, _ = self.interpolate_batch(times_sub)

        self._line_cam_ecef = pos_sub     # (K, 3) camera ECEF
        self._line_times = times_sub      # (K,)

        print(f"  Line index: {len(times_sub)} camera positions "
              f"(stride={stride}, {L} total scanlines)")

    def initial_time_from_lines(self, ground_ecef: np.ndarray) -> np.ndarray:
        """
        Find initial time guess for each ground point using the
        per-scanline camera position index.

        For each ground point, finds the scanline whose camera position
        is closest in 3D ECEF distance.  Since the search is over actual
        camera positions (lever-arm corrected) at actual exposure times,
        the initial guess is within a few scanlines of the true solution.

        Parameters
        ----------
        ground_ecef : (M, 3) ground points in ECEF

        Returns
        -------
        t_guess : (M,) initial time estimates
        """
        if self._line_cam_ecef is None:
            raise RuntimeError(
                "Call build_line_index(exposure_times) before "
                "initial_time_from_lines()"
            )

        coarse_pos = self._line_cam_ecef    # (K, 3)
        coarse_time = self._line_times      # (K,)

        M = ground_ecef.shape[0]
        chunk = 5000
        t_guess = np.empty(M, dtype=np.float64)

        for i in range(0, M, chunk):
            end = min(i + chunk, M)
            diff = ground_ecef[i:end, None, :] - coarse_pos[None, :, :]
            dist2 = np.sum(diff ** 2, axis=2)
            best = np.argmin(dist2, axis=1)
            t_guess[i:end] = coarse_time[best]

        return t_guess

    # =================================================================
    # Scalar interpolation
    # =================================================================

    def interpolate(self, t: float):
        """
        Interpolate trajectory at a single time t.

        Returns
        -------
        pos_cam_ecef : (3,) camera position in ECEF [m]
        R_cam_ecef   : (3,3) rotation matrix  R_{Cam←ECEF}
        """
        # Find bracketing indices
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, self.n_records - 2)
        frac = (t - self.times[idx]) / (self.times[idx + 1] - self.times[idx])

        # Linearly interpolate ECEF position
        pos_imu = (1 - frac) * self.pos_ecef[idx] + frac * self.pos_ecef[idx + 1]

        # Slerp for R_{Body←NED}
        q_bn = cu.slerp(self.q_body_ned[idx], self.q_body_ned[idx + 1], frac)
        R_body_ned = cu.quaternion_to_rotation(q_bn)

        # Slerp for R_{NED←ECEF}
        q_ne = cu.slerp(self.q_ned_ecef[idx], self.q_ned_ecef[idx + 1], frac)
        R_ned_ecef = cu.quaternion_to_rotation(q_ne)

        # Full chain:  R_{Cam←ECEF} = R_{Cam←Body} @ R_{Body←NED} @ R_{NED←ECEF}
        R_cam_ecef = self.R_cam_body @ R_body_ned @ R_ned_ecef

        # Camera position in ECEF:
        lever_ecef = R_ned_ecef.T @ R_body_ned.T @ self.lever_arm
        pos_cam = pos_imu + lever_ecef

        return pos_cam, R_cam_ecef

    # =================================================================
    # Vectorised batch interpolation
    # =================================================================

    def interpolate_batch(self, t: np.ndarray):
        """
        Interpolate trajectory at an array of times.

        Parameters
        ----------
        t : (M,) array of GPS times

        Returns
        -------
        pos_cam_ecef : (M, 3) camera positions in ECEF
        R_cam_ecef   : (M, 3, 3) rotation matrices R_{Cam←ECEF}
        """
        M = len(t)
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, self.n_records - 2)
        dt = self.times[idx + 1] - self.times[idx]
        # Guard against zero dt
        dt = np.where(dt == 0, 1.0, dt)
        frac = (t - self.times[idx]) / dt
        frac = np.clip(frac, 0.0, 1.0)

        # Linear interpolation of ECEF position
        pos_imu = ((1 - frac)[:, None] * self.pos_ecef[idx] +
                    frac[:, None] * self.pos_ecef[idx + 1])

        # Slerp for R_{Body←NED}
        q_bn = cu.slerp_batch(
            self.q_body_ned[idx], self.q_body_ned[idx + 1], frac
        )
        R_body_ned = cu.quaternion_to_rotation_batch(q_bn)  # (M, 3, 3)

        # Slerp for R_{NED←ECEF}
        q_ne = cu.slerp_batch(
            self.q_ned_ecef[idx], self.q_ned_ecef[idx + 1], frac
        )
        R_ned_ecef = cu.quaternion_to_rotation_batch(q_ne)  # (M, 3, 3)

        # Full chain:  R_{Cam←ECEF} = R_{Cam←Body} @ R_{Body←NED} @ R_{NED←ECEF}
        R_cam_ned = np.einsum('ij,mjk->mik', self.R_cam_body, R_body_ned)
        R_cam_ecef = np.einsum('mij,mjk->mik', R_cam_ned, R_ned_ecef)

        # Lever arm in ECEF for each record
        lever_body_exp = np.broadcast_to(self.lever_arm, (M, 3))
        lever_ned = np.einsum('mji,mj->mi', R_body_ned, lever_body_exp)
        lever_ecef = np.einsum('mji,mj->mi', R_ned_ecef, lever_ned)

        pos_cam = pos_imu + lever_ecef

        return pos_cam, R_cam_ecef
