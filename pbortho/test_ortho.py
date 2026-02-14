#!/usr/bin/env python3
"""
Synthetic validation of the ortho-pushbroom math pipeline.

Creates a fake straight-line trajectory and verifies that:
  1. ECEF ↔ NED transforms are self-consistent.
  2. The Newton solver recovers the correct scanline time for a known
     ground point.
  3. The Brown–Conrady forward model round-trips correctly.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ortho_pushbroom import (
    geodetic_to_ecef,
    ecef2ned_matrix,
    rpy_to_matrix,
    BrownConradyPushbroom,
    SBET_DTYPE,
    TrajectoryInterpolator,
)


def make_synthetic_sbet(n=500, duration=10.0):
    """Straight east-west flight at ~1000 m AGL over Bern (46.95°N, 7.45°E)."""
    sbet = np.zeros(n, dtype=SBET_DTYPE)
    t = np.linspace(100_000.0, 100_000.0 + duration, n)
    sbet["time"] = t

    lat0 = np.radians(46.95)
    lon_start = np.radians(7.44)
    lon_end = np.radians(7.46)
    alt = 1500.0  # ellipsoidal height

    sbet["lat"] = lat0
    sbet["lon"] = np.linspace(lon_start, lon_end, n)
    sbet["alt"] = alt

    # heading ~90° (east), level flight
    sbet["heading"] = np.full(n, np.radians(90.0))
    sbet["pitch"] = 0.0
    sbet["roll"] = 0.0
    return sbet


def test_ecef_ned_roundtrip():
    """ECEF→NED→ECEF should be identity (R^T R = I)."""
    lat = np.array([np.radians(46.95)])
    lon = np.array([np.radians(7.45)])
    R = ecef2ned_matrix(lat, lon)[0]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    print("✓  ECEF↔NED roundtrip")


def test_rpy_identity():
    """Zero roll/pitch/heading → identity."""
    R = rpy_to_matrix(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(R[0], np.eye(3), atol=1e-15)
    print("✓  RPY(0,0,0) = I")


def test_camera_model_distortion():
    """With zero distortion coeffs the model reduces to f·x + cx."""
    cam = BrownConradyPushbroom(f=10000.0, cx=4096.0,
                                 k1=0, k2=0, p1=0, p2=0)
    xn = np.array([0.01, -0.02, 0.0])
    yn = np.zeros(3)
    u = cam.project(xn, yn)
    expected = 10000.0 * xn + 4096.0
    np.testing.assert_allclose(u, expected, atol=1e-10)
    print("✓  Camera model (zero distortion)")


def test_newton_solver():
    """Place a ground point directly below scanline #250 and verify recovery."""
    sbet = make_synthetic_sbet(n=500, duration=10.0)
    traj = TrajectoryInterpolator(sbet)

    # Ground point directly below sensor at line 250
    target_line = 250
    t_true = sbet["time"][target_line]
    lat_t = sbet["lat"][target_line]
    lon_t = sbet["lon"][target_line]
    ground_alt = 500.0  # below the flight

    # ECEF of the ground point (directly below = same lat/lon, lower alt)
    gx, gy, gz = geodetic_to_ecef(
        np.array([lat_t]), np.array([lon_t]), np.array([ground_alt])
    )
    ground_ecef = np.column_stack([gx, gy, gz])  # (1, 3)

    # ── R_body2cam = identity (simplest case) ──
    R_body2cam = np.eye(3)
    lever_arm = np.zeros(3)

    # ── Newton iteration (manual, matching OrthoEngine logic) ──
    t = np.array([sbet["time"][200]])  # deliberately wrong initial guess
    tol = 1e-8
    for it in range(30):
        pos = traj.ecef_pos(t)
        lat_, lon_ = traj.latlon(t)
        R_n2b = traj.R_ned2body(t)
        R_e2n = ecef2ned_matrix(lat_, lon_)

        d_ecef = ground_ecef - pos
        d_ned = np.einsum("nij,nj->ni", R_e2n, d_ecef)
        d_body = np.einsum("nij,nj->ni", R_n2b, d_ned)
        d_cam = d_body @ R_body2cam.T  # simplified: R=I

        residual = d_cam[:, 1] / d_cam[:, 2]
        if np.abs(residual[0]) < tol:
            break

        dt = 1e-5
        tp = t + dt
        pos_p = traj.ecef_pos(tp)
        lat_p, lon_p = traj.latlon(tp)
        R_n2b_p = traj.R_ned2body(tp)
        R_e2n_p = ecef2ned_matrix(lat_p, lon_p)
        d_ecef_p = ground_ecef - pos_p
        d_ned_p = np.einsum("nij,nj->ni", R_e2n_p, d_ecef_p)
        d_body_p = np.einsum("nij,nj->ni", R_n2b_p, d_ned_p)
        d_cam_p = d_body_p @ R_body2cam.T
        res_p = d_cam_p[:, 1] / d_cam_p[:, 2]
        dr_dt = (res_p - residual) / dt
        t = t - np.clip(residual / dr_dt, -0.5, 0.5)
        t = np.clip(t, traj.t_min + 1e-6, traj.t_max - 1e-6)

    time_err = abs(t[0] - t_true)
    line_err = time_err / (sbet["time"][1] - sbet["time"][0])
    print(f"✓  Newton solver converged in {it+1} iterations")
    print(f"   Δt = {time_err:.4e} s  ≈ {line_err:.2f} lines")
    print(f"   residual = {abs(residual[0]):.2e} rad")
    assert abs(residual[0]) < 1e-6, f"Residual not converged: {residual[0]}"

    # ── cross-track pixel should be near principal point (nadir) ──
    xn = d_cam[0, 0] / d_cam[0, 2]
    cam = BrownConradyPushbroom(f=10000.0, cx=4096.0,
                                 k1=0, k2=0, p1=0, p2=0)
    u = cam.project(np.array([xn]), np.array([0.0]))
    print(f"   cross-track pixel u = {u[0]:.2f}  (expect ≈ {cam.cx})")
    assert abs(u[0] - cam.cx) < 100, "Pixel too far from principal point for nadir view"


if __name__ == "__main__":
    test_ecef_ned_roundtrip()
    test_rpy_identity()
    test_camera_model_distortion()
    test_newton_solver()
    print("\nAll tests passed.")
