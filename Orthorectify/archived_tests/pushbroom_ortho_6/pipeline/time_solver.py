"""
time_solver.py — Newton iteration to find the scanline time for each ground point.

This is the mathematical core of the bottom-up orthorectification.

Problem Statement
-----------------
Given a ground point P in ECEF and a moving pushbroom sensor with known
trajectory, find the time t* at which the sensor's linear array sweeps
across the ground point.

At time t*, the along-track residual in camera frame must equal the
smile offset at the observed pixel position:

    f(t) = s·(Yc/Zc) + Δcy − smile_tan(pixel(t))  →  0

Anti-Curtaining Design
----------------------
There are TWO silent clamping traps that cause curtaining if not handled:

1. TIME CLAMPING (along-track curtains):
   If Newton clamps t to the exposure time range [t_first, t_last],
   points outside the image's along-track extent stall at the boundary,
   delta_t becomes small, and the point falsely "converges" at the
   first/last scanline.  np.interp(t → line_idx) also clamps.
   FIX: Newton clamps to SBET trajectory bounds only. Post-solve check
   rejects points whose time falls outside the exposure range.

2. ANGLE CLAMPING (across-track curtains):
   np.interp in camera.project() clamps angles beyond the LUT range
   to pixel 0 or pixel N-1.  is_valid_pixel() then passes because the
   pixel index is technically in [0, N).  Every DSM pixel outside the
   sensor FOV gets mapped to edge pixels.
   FIX: Post-solve check uses camera.is_within_fov(d_cam) which tests
   the actual angle against the LUT angular bounds.
"""

import numpy as np
from .trajectory import Trajectory
from .camera_model import PushbroomCamera
from .config_loader import NewtonConfig


class TimeSolver:
    """
    Vectorised Newton iteration for scanline time determination.

    Parameters
    ----------
    trajectory : Trajectory interpolator
    camera     : PushbroomCamera model
    config     : NewtonConfig with iteration parameters
    exposure_times : (L,) GPS times of each scanline in the BIL
    """

    def __init__(self, trajectory: Trajectory, camera: PushbroomCamera,
                 config: NewtonConfig, exposure_times: np.ndarray):
        self.traj = trajectory
        self.cam = camera
        self.max_iter = config.max_iterations
        self.tol = config.convergence_threshold
        self.dt = config.numerical_dt

        self.exposure_times = exposure_times
        self.n_lines = len(exposure_times)

        # Exposure time bounds (image extent)
        self.t_first = exposure_times[0]
        self.t_last = exposure_times[-1]

        # SBET trajectory bounds (wider than exposure, has margin)
        # Newton is free to explore within these without extrapolation.
        self.t_traj_min = trajectory.t_min
        self.t_traj_max = trajectory.t_max

    def _compute_residual(self, t: np.ndarray,
                          ground_ecef: np.ndarray) -> np.ndarray:
        """
        Compute the along-track residual in tangent space for each point.

        f(t) = s·(Yc/Zc) + Δcy − smile_tan(pixel(t))

        Parameters
        ----------
        t : (M,) times
        ground_ecef : (M, 3) ground points in ECEF

        Returns
        -------
        residual : (M,) along-track residuals [tangent-space units]
        """
        pos_cam, R_cam_ecef = self.traj.interpolate_batch(t)

        # Vector from camera to ground in ECEF
        d_ecef = ground_ecef - pos_cam  # (M, 3)

        # Rotate to camera frame:  d_cam = R_{Cam←ECEF} @ d_ecef
        d_cam = np.einsum('mij,mj->mi', R_cam_ecef, d_ecef)  # (M, 3)

        # Project to get across-track pixel (for smile lookup)
        u_pixel, y_lab = self.cam.project(d_cam)

        # Smile-corrected residual
        return self.cam.along_track_residual(d_cam, u_pixel)

    def solve(self, ground_ecef: np.ndarray,
              t_init: np.ndarray) -> tuple:
        """
        Solve for scanline times using vectorised Newton iteration.

        Parameters
        ----------
        ground_ecef : (M, 3) ground points in ECEF
        t_init      : (M,) initial time guesses (must match ground_ecef rows)

        Returns
        -------
        t_solved : (M,) converged GPS times
        u_pixel  : (M,) across-track pixel coordinates
        y_lab    : (M,) along-track lab-corrected tangent
        line_idx : (M,) fractional scanline indices in the BIL
        valid    : (M,) boolean mask — True for converged, in-bounds points
        """
        M = ground_ecef.shape[0]
        t = t_init.copy()

        # Clamp initial guesses to TRAJECTORY bounds (NOT exposure bounds!)
        # This prevents interpolation extrapolation but lets Newton
        # freely explore times outside the image range.
        t = np.clip(t, self.t_traj_min, self.t_traj_max)

        converged = np.zeros(M, dtype=bool)

        for iteration in range(self.max_iter):
            active = ~converged
            if not np.any(active):
                break

            # Compute residual and central-difference Jacobian
            f_t = self._compute_residual(t, ground_ecef)

            t_plus = t.copy()
            t_minus = t.copy()
            t_plus[active] += self.dt
            t_minus[active] -= self.dt

            f_plus = self._compute_residual(t_plus, ground_ecef)
            f_minus = self._compute_residual(t_minus, ground_ecef)

            f_prime = (f_plus - f_minus) / (2.0 * self.dt)
            f_prime = np.where(np.abs(f_prime) < 1e-20, 1e-20, f_prime)

            # Newton update (NO clamping to exposure bounds!)
            delta_t = f_t / f_prime
            t[active] -= delta_t[active]

            # Clamp to TRAJECTORY bounds only (prevent extrapolation crash)
            t = np.clip(t, self.t_traj_min, self.t_traj_max)

            # Convergence check
            newly_converged = np.abs(delta_t) < self.tol
            converged |= newly_converged

        # -----------------------------------------------------------------
        # Final projection to get pixel coordinates
        # -----------------------------------------------------------------
        pos_cam, R_cam_ecef = self.traj.interpolate_batch(t)
        d_ecef = ground_ecef - pos_cam
        d_cam = np.einsum('mij,mj->mi', R_cam_ecef, d_ecef)

        u_pixel, y_lab = self.cam.project(d_cam)

        # Map solved time to fractional scanline index
        # NOTE: np.interp clamps, so line_idx is always in [0, n_lines-1].
        # Do NOT rely on line_idx for bounds checking — use t directly.
        line_idx = np.interp(t, self.exposure_times,
                             np.arange(self.n_lines))

        # -----------------------------------------------------------------
        # Validity mask — defeating BOTH clamping traps
        # -----------------------------------------------------------------
        #
        # CHECK 1: Newton converged
        #   Small delta_t in last iteration
        #
        # CHECK 2: Solved time within image exposure range
        #   ANTI-CURTAIN (along-track): points outside the image converge
        #   to times beyond [t_first, t_last]. Since Newton does NOT clamp
        #   to exposure bounds, these points overshoot and are caught here.
        #   Without this, np.interp clamps t → valid line_idx → curtains.
        #
        # CHECK 3: Across-track angle within sensor FOV
        #   ANTI-CURTAIN (across-track): np.interp in project() clamps
        #   out-of-FOV angles to edge pixels (0 or N-1), making
        #   is_valid_pixel() useless. is_within_fov() checks the actual
        #   angle against the LUT angular bounds.
        #
        # CHECK 4: Point in front of camera (Zc > 0)

        in_time_range = (t >= self.t_first) & (t <= self.t_last)
        in_fov = self.cam.is_within_fov(d_cam)

        valid = (converged &
                 in_time_range &
                 in_fov &
                 (d_cam[:, 2] > 0))

        return t, u_pixel, y_lab, line_idx, valid
