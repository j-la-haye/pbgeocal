"""
time_solver.py — Newton iteration to find the scanline time for each ground point.

This is the mathematical core of the bottom-up orthorectification.

Problem Statement
-----------------
Given a ground point P in ECEF and a moving pushbroom sensor with known
trajectory, find the time t* at which the sensor's linear array sweeps
across the ground point.

At time t*, the point P must lie on the sensor's across-track scanline.
In the camera frame, this means the along-track component is zero:

    f(t) = P_cam_y(t) / P_cam_z(t) = 0

where P_cam(t) = R_{Cam←ECEF}(t) · (P_ecef - S_ecef(t))

    S_ecef(t) = camera position at time t
    R_{Cam←ECEF}(t) = orientation at time t

Newton's Method
---------------
    t_{n+1} = t_n - f(t_n) / f'(t_n)

f'(t) is computed via central finite differences:

    f'(t) ≈ [f(t + δ) - f(t - δ)] / (2δ)

Convergence is typically achieved in 3–5 iterations for well-initialised
points, given the smooth trajectory and small angular rates of airborne
platforms.

Vectorised Implementation
-------------------------
The solver processes all ground points for a tile simultaneously using
numpy broadcasting.  Invalid or non-converging points are masked.
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
        self.t_min = exposure_times[0]
        self.t_max = exposure_times[-1]

    def _compute_residual(self, t: np.ndarray,
                          ground_ecef: np.ndarray) -> np.ndarray:
        """
        Compute the along-track residual in tangent space for each point.

        For the decoupled camera model with in-flight correction:
            f(t) = s·(Yc/Zc) + Δcy − smile_tan(pixel(t))

        The residual is in tangent-space units, which is smooth and well-
        conditioned for Newton iteration.

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

        # Smile-corrected residual:  atan2(Yc,Zc) - smile(u)
        return self.cam.along_track_residual(d_cam, u_pixel)

    def solve(self, ground_ecef: np.ndarray,
              t_init: np.ndarray) -> tuple:
        """
        Solve for scanline times using vectorised Newton iteration.

        Parameters
        ----------
        ground_ecef : (M, 3) ground points in ECEF
        t_init      : (M,) initial time guesses (from Trajectory.initial_time_guess)

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
                             np.arange(len(self.exposure_times)))

        # Validity mask
        valid = (converged &
                 self.cam.is_valid_pixel(u_pixel) &
                 (t >= self.t_min) & (t <= self.t_max) &
                 (d_cam[:, 2] > 0))   # point must be in front of camera

        return t, u_pixel, y_lab, line_idx, valid
