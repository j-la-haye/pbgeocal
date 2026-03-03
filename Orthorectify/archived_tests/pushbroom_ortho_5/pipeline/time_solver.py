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

Newton's Method
---------------
    t_{n+1} = t_n - f(t_n) / f'(t_n)

f'(t) is computed via central finite differences:
    f'(t) ≈ [f(t + δ) - f(t - δ)] / (2δ)

Convergence is typically achieved in 3–5 iterations for well-initialised
points.

Validity
--------
A pixel is valid only if ALL of:
  1. Newton converged (|Δt| < threshold)
  2. Across-track pixel within sensor bounds [0, num_pixels)
  3. Solved time maps to a scanline within the image [0, n_lines-1]
  4. Ground point is in front of camera (Zc > 0)

Note: The swath is constrained naturally by these checks. Any ground
point outside the image footprint either maps to a pixel outside [0,
num_pixels) or to a time outside [0, n_lines-1].  No explicit swath
polygon or distance mask is needed.
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
        self.t_min = exposure_times[0]
        self.t_max = exposure_times[-1]

    def _compute_residual(self, t: np.ndarray,
                          ground_ecef: np.ndarray) -> np.ndarray:
        """
        Compute the along-track residual in tangent space for each point.

        For the decoupled camera model with in-flight correction:
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

        # Clamp initial guesses to valid time range
        t = np.clip(t, self.t_min, self.t_max)

        converged = np.zeros(M, dtype=bool)

        for iteration in range(self.max_iter):
            # Points still needing iteration
            active = ~converged

            if not np.any(active):
                break

            # Compute residual at t and t ± dt for numerical derivative
            f_t = self._compute_residual(t, ground_ecef)

            t_plus = t.copy()
            t_minus = t.copy()
            t_plus[active] += self.dt
            t_minus[active] -= self.dt

            f_plus = self._compute_residual(t_plus, ground_ecef)
            f_minus = self._compute_residual(t_minus, ground_ecef)

            # Central difference Jacobian
            f_prime = (f_plus - f_minus) / (2.0 * self.dt)

            # Guard against zero derivative
            f_prime = np.where(np.abs(f_prime) < 1e-20, 1e-20, f_prime)

            # Newton update
            delta_t = f_t / f_prime
            t[active] -= delta_t[active]

            # Clamp to valid time range
            t = np.clip(t, self.t_min, self.t_max)

            # Check convergence
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
        line_idx = np.interp(t, self.exposure_times,
                             np.arange(self.n_lines))

        # -----------------------------------------------------------------
        # Validity mask — this naturally constrains to the image swath
        # -----------------------------------------------------------------
        # 1. Newton converged
        # 2. Across-track pixel within sensor [0, num_pixels)
        # 3. Scanline index within image [0, n_lines - 1]
        #    This is the KEY swath constraint: ground points outside the
        #    along-track extent of the image are rejected here.
        # 4. Point is in front of camera (Zc > 0)
        valid = (converged &
                 self.cam.is_valid_pixel(u_pixel) &
                 (line_idx >= 0) & (line_idx <= self.n_lines - 1) &
                 (d_cam[:, 2] > 0))

        return t, u_pixel, y_lab, line_idx, valid
