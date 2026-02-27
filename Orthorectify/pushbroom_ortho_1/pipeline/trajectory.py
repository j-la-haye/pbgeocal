"""
trajectory.py — SBET trajectory interpolator with Slerp for attitude.

Provides interpolation of the 200 Hz SBET data at arbitrary times:
  • Position (lat, lon, alt → ECEF): linear interpolation
  • Attitude (roll, pitch, heading → R_{Body←NED}): quaternion Slerp
  • Full exterior orientation chain:
        R_{Cam←ECEF}(t) = R_{Cam←Body} @ R_{Body←NED}(t) @ R_{NED←ECEF}(t)

The Trajectory object pre-computes quaternions from Euler angles for all
SBET records and provides both scalar and vectorised interpolation.

Usage
-----
    traj = Trajectory(sbet, mounting_config)
    pos_ecef, R_cam_ecef = traj.interpolate(t)            # scalar
    pos_ecef, R_cam_ecef = traj.interpolate_batch(times)   # vectorised
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

    # -----------------------------------------------------------------
    # Scalar interpolation
    # -----------------------------------------------------------------
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
        #   lever_ecef = R_{ECEF←NED} @ R_{NED←Body} @ lever_body
        #              = (R_{NED←ECEF})^T @ (R_{Body←NED})^T @ lever_body
        lever_ecef = R_ned_ecef.T @ R_body_ned.T @ self.lever_arm
        pos_cam = pos_imu + lever_ecef

        return pos_cam, R_cam_ecef

    # -----------------------------------------------------------------
    # Vectorised batch interpolation
    # -----------------------------------------------------------------
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
        # R_cam_body is (3,3), R_body_ned is (M,3,3), R_ned_ecef is (M,3,3)
        R_cam_ned = np.einsum('ij,mjk->mik', self.R_cam_body, R_body_ned)  # (M,3,3)
        R_cam_ecef = np.einsum('mij,mjk->mik', R_cam_ned, R_ned_ecef)      # (M,3,3)

        # Lever arm in ECEF for each record
        # lever_ecef = R_ned_ecef^T @ R_body_ned^T @ lever_body
        lever_body_exp = np.broadcast_to(self.lever_arm, (M, 3))
        lever_ned = np.einsum('mji,mj->mi', R_body_ned, lever_body_exp)  # R_body_ned^T @ lever
        lever_ecef = np.einsum('mji,mj->mi', R_ned_ecef, lever_ned)     # R_ned_ecef^T @ lever_ned

        pos_cam = pos_imu + lever_ecef

        return pos_cam, R_cam_ecef

    # -----------------------------------------------------------------
    # Utility: initial time guess from ground positions
    # -----------------------------------------------------------------
    def initial_time_guess(self, ground_ecef: np.ndarray) -> np.ndarray:
        """
        Provide an initial time estimate for the Newton solver by finding the
        closest SBET record (in ECEF distance) for each ground point.

        Uses a coarse search over decimated trajectory records.

        Parameters
        ----------
        ground_ecef : (M, 3) ground points in ECEF

        Returns
        -------
        t_guess : (M,) initial time estimates
        """
        # Decimate trajectory for speed (every 20th record ≈ 10 Hz)
        stride = max(1, self.n_records // 500)
        coarse_pos = self.pos_ecef[::stride]    # (K, 3)
        coarse_time = self.times[::stride]      # (K,)

        # For each ground point, find nearest coarse position
        # Using broadcasting: (M,1,3) - (1,K,3) → (M,K,3) → (M,K) → argmin
        # For large M, do in chunks to limit memory
        M = ground_ecef.shape[0]
        chunk = 5000
        t_guess = np.empty(M, dtype=np.float64)

        for i in range(0, M, chunk):
            end = min(i + chunk, M)
            diff = ground_ecef[i:end, None, :] - coarse_pos[None, :, :]  # (c, K, 3)
            dist2 = np.sum(diff ** 2, axis=2)                            # (c, K)
            best = np.argmin(dist2, axis=1)                              # (c,)
            t_guess[i:end] = coarse_time[best]

        return t_guess
