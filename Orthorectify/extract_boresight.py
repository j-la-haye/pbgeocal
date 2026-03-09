#!/usr/bin/env python3
"""
extract_boresight.py — Extract boresight Euler angles from optimised axis-angle.

The factor graph jointly optimises the complete body→camera rotation
(mounting + boresight) as a single axis-angle vector.  This script
decomposes it into the known mounting matrix and the residual boresight
rotation, expressed as ZYX Euler angles (roll, pitch, yaw).

Rotation chain in the pipeline:

    R_{Cam←Body}  =  R_boresight  @  R_mount

      R_mount     :  known mechanical installation  (body→camera, 3×3)
      R_boresight :  small residual misalignment     (ZYX Euler)
      R_optimised :  combined result from factor graph (axis-angle vector)

Decomposition:

    R_boresight  =  R_optimised  @  R_mount^T

Then decompose R_boresight into intrinsic ZYX Euler angles:

    R_boresight  =  Rz(yaw) @ Ry(pitch) @ Rx(roll)
"""

import numpy as np


# =========================================================================
#  Core functions
# =========================================================================

def axis_angle_to_matrix(v: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle vector to 3×3 rotation matrix (Rodrigues' formula).

    Parameters
    ----------
    v : (3,) axis-angle vector.  Direction = rotation axis,
        magnitude = rotation angle [radians].

    Returns
    -------
    R : (3, 3) rotation matrix
    """
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v)
    if theta < 1e-15:
        return np.eye(3)
    k = v / theta
    K = np.array([
        [ 0,    -k[2],  k[1]],
        [ k[2],  0,    -k[0]],
        [-k[1],  k[0],  0   ],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_euler_zyx(R: np.ndarray) -> tuple:
    """
    Decompose rotation matrix into intrinsic ZYX Euler angles.

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    This matches the pipeline convention in coord_utils.euler_to_rotation().

    Parameters
    ----------
    R : (3, 3) rotation matrix

    Returns
    -------
    roll, pitch, yaw : floats [radians]
    """
    # pitch from -R[2,0] = sin(pitch)
    sp = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sp)

    if np.abs(np.cos(pitch)) > 1e-10:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock (pitch ≈ ±90°) — shouldn't happen for boresight
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw  = 0.0

    return roll, pitch, yaw


def euler_zyx_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Reconstruct rotation matrix from ZYX Euler angles.

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ], dtype=np.float64)


def extract_boresight(axis_angle_vec, R_mount):
    """
    Extract boresight ZYX Euler angles from an optimised axis-angle vector.

    Parameters
    ----------
    axis_angle_vec : (3,) optimised body→camera rotation (axis-angle)
    R_mount        : (3, 3) known mounting matrix (body→camera)

    Returns
    -------
    dict with keys:
        roll_rad, pitch_rad, yaw_rad  : boresight angles [radians]
        roll_deg, pitch_deg, yaw_deg  : boresight angles [degrees]
        R_boresight                   : (3, 3) boresight rotation matrix
        R_optimised                   : (3, 3) full optimised rotation
        roundtrip_error               : Frobenius norm of roundtrip check
    """
    v = np.asarray(axis_angle_vec, dtype=np.float64)
    R_mount = np.asarray(R_mount, dtype=np.float64)

    # Axis-angle → full rotation matrix
    R_opt = axis_angle_to_matrix(v)

    # Decompose:  R_opt = R_boresight @ R_mount
    #           → R_boresight = R_opt @ R_mount^T
    R_boresight = R_opt @ R_mount.T

    # Extract ZYX Euler angles
    roll, pitch, yaw = matrix_to_euler_zyx(R_boresight)

    # Roundtrip verification
    R_check = euler_zyx_to_matrix(roll, pitch, yaw) @ R_mount
    roundtrip_err = np.linalg.norm(R_opt - R_check, 'fro')

    return {
        'roll_rad':  roll,
        'pitch_rad': pitch,
        'yaw_rad':   yaw,
        'roll_deg':  np.rad2deg(roll),
        'pitch_deg': np.rad2deg(pitch),
        'yaw_deg':   np.rad2deg(yaw),
        'R_boresight': R_boresight,
        'R_optimised': R_opt,
        'roundtrip_error': roundtrip_err,
    }


# =========================================================================
#  Main — example usage and validation
# =========================================================================

if __name__ == '__main__':

    # Known mounting matrix from config
    R_mount = np.array([
        [ 0.0,  1.0,  0.0],    # cam_X = body_Y
        [-1.0,  0.0,  0.0],    # cam_Y = -body_X
        [ 0.0,  0.0,  1.0],    # cam_Z = body_Z
    ])

    # Optimised body→camera axis-angle from factor graph
    axis_angle_opt = np.array([
        -0.09456387162208557,
         0.011100600473582745,
        -1.5692365169525146,
    ])

    print("=" * 65)
    print(" Extract boresight from optimised axis-angle")
    print("=" * 65)

    result = extract_boresight(axis_angle_opt, R_mount)

    print(f"\nInput axis-angle: {axis_angle_opt}")
    print(f"Axis-angle magnitude: {np.linalg.norm(axis_angle_opt):.6f} rad "
          f"= {np.rad2deg(np.linalg.norm(axis_angle_opt)):.4f}°")

    print(f"\nR_optimised (body→camera):")
    print(np.array2string(result['R_optimised'], precision=8))

    print(f"\nR_boresight (residual, near identity):")
    print(np.array2string(result['R_boresight'], precision=8))

    print(f"\nBoresight ZYX Euler angles:")
    print(f"  roll  = {result['roll_deg']:+.6f}°  ({result['roll_rad']:+.10f} rad)")
    print(f"  pitch = {result['pitch_deg']:+.6f}°  ({result['pitch_rad']:+.10f} rad)")
    print(f"  yaw   = {result['yaw_deg']:+.6f}°  ({result['yaw_rad']:+.10f} rad)")
    print(f"  total = {np.rad2deg(np.linalg.norm(np.array([result['roll_rad'], result['pitch_rad'], result['yaw_rad']]))):.4f}°")

    print(f"\nRoundtrip error: {result['roundtrip_error']:.2e}")

    # config.yaml snippet
    print(f"\nconfig.yaml snippet:")
    print(f"mounting:")
    print(f"  boresight:")
    print(f"    roll:  {result['roll_deg']:.10f}")
    print(f"    pitch: {result['pitch_deg']:.10f}")
    print(f"    yaw:   {result['yaw_deg']:.10f}")
    print(f"  mounting_matrix:")
    print(f"    - [ 0.0,  1.0,  0.0]")
    print(f"    - [-1.0,  0.0,  0.0]")
    print(f"    - [ 0.0,  0.0,  1.0]")

    # Cross-check with scipy
    print(f"\n{'='*65}")
    print(f" Scipy cross-check")
    print(f"{'='*65}")
    try:
        from scipy.spatial.transform import Rotation

        R_bs = Rotation.from_matrix(result['R_boresight'])

        # Intrinsic ZYX
        yaw_s, pitch_s, roll_s = R_bs.as_euler('ZYX', degrees=True)
        print(f"  scipy intrinsic ZYX: roll={roll_s:+.6f}°, "
              f"pitch={pitch_s:+.6f}°, yaw={yaw_s:+.6f}°")

        # Also show axis-angle of boresight alone
        bs_rotvec = R_bs.as_rotvec()
        bs_angle = np.linalg.norm(bs_rotvec)
        bs_axis = bs_rotvec / bs_angle if bs_angle > 0 else bs_rotvec
        print(f"  Boresight axis-angle: {np.rad2deg(bs_angle):.4f}° "
              f"about [{bs_axis[0]:+.4f}, {bs_axis[1]:+.4f}, {bs_axis[2]:+.4f}]")

        # Verify full chain: euler → rotvec matches input
        R_full = Rotation.from_matrix(
            euler_zyx_to_matrix(result['roll_rad'], result['pitch_rad'],
                                result['yaw_rad']) @ R_mount)
        v_check = R_full.as_rotvec()
        print(f"\n  Reconstructed axis-angle: {v_check}")
        print(f"  Original axis-angle:     {axis_angle_opt}")
        print(f"  Difference norm: {np.linalg.norm(v_check - axis_angle_opt):.2e}")

    except ImportError:
        print("  scipy not available, skipping cross-check")
