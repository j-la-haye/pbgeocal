"""
coord_utils.py — Coordinate transformations and rotation utilities.

Provides:
  • Geodetic ↔ ECEF conversions (WGS-84)
  • ECEF → NED rotation matrix at a given lat/lon
  • Euler (roll, pitch, heading) → rotation matrix (NED → Body)
  • Euler → quaternion, quaternion Slerp, quaternion → rotation matrix
  • Boresight rotation from small-angle Euler triplet

All rotations use the right-hand rule.  Euler order is ZYX
(heading, pitch, roll) following the Applanix NED convention.

Frame definitions:
  NED    :  X-North, Y-East, Z-Down  (navigation frame at sensor position)
  Body   :  X-Forward, Y-Right, Z-Down (Applanix body frame = NED rotated
            by heading, pitch, roll)
  Camera :  X-Right, Y-Back, Z-Down  (defined by mounting matrix)
  ECEF   :  Earth-Centered Earth-Fixed (WGS-84)
"""

import numpy as np

# WGS-84 ellipsoid constants
WGS84_A = 6378137.0                          # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563                # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)         # semi-minor axis
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2     # first eccentricity squared


# =============================================================================
# Geodetic ↔ ECEF
# =============================================================================

def geodetic_to_ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray):
    """
    Convert geodetic (lat, lon, alt) to ECEF (X, Y, Z).

    Parameters are in radians / metres.  Works for scalars and arrays.

    Returns
    -------
    ecef : (..., 3) array  [X, Y, Z] in metres
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)

    X = (N + alt) * cos_lat * cos_lon
    Y = (N + alt) * cos_lat * sin_lon
    Z = (N * (1.0 - WGS84_E2) + alt) * sin_lat

    return np.stack([X, Y, Z], axis=-1)


def ecef_to_geodetic(ecef: np.ndarray):
    """
    ECEF → geodetic using Bowring's iterative method (2 iterations suffice).

    Parameters
    ----------
    ecef : (..., 3)

    Returns
    -------
    lat, lon, alt  (radians, radians, metres)
    """
    X = ecef[..., 0]
    Y = ecef[..., 1]
    Z = ecef[..., 2]

    p = np.sqrt(X ** 2 + Y ** 2)
    lon = np.arctan2(Y, X)

    # Initial Bowring estimate
    theta = np.arctan2(Z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(
        Z + (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    # One refinement
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    alt = p / np.cos(lat) - N

    return lat, lon, alt


# =============================================================================
# Rotation matrices
# =============================================================================

def rotation_ecef_to_ned(lat: float, lon: float) -> np.ndarray:
    """
    3×3 rotation matrix R_{NED←ECEF} that transforms a vector from ECEF to NED
    at the geodetic position (lat, lon) in radians.

    v_ned = R @ v_ecef
    """
    sl, cl = np.sin(lat), np.cos(lat)
    slo, clo = np.sin(lon), np.cos(lon)

    R = np.array([
        [-sl * clo, -sl * slo,  cl],   # North
        [-slo,       clo,       0.0],   # East
        [-cl * clo, -cl * slo, -sl],    # Down
    ], dtype=np.float64)
    return R


def rotation_ecef_to_ned_batch(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Vectorised: (N,3,3) rotation matrices R_{NED←ECEF}.
    """
    N = len(lat)
    sl, cl = np.sin(lat), np.cos(lat)
    slo, clo = np.sin(lon), np.cos(lon)

    R = np.zeros((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = -sl * clo
    R[:, 0, 1] = -sl * slo
    R[:, 0, 2] = cl
    R[:, 1, 0] = -slo
    R[:, 1, 1] = clo
    R[:, 1, 2] = 0.0
    R[:, 2, 0] = -cl * clo
    R[:, 2, 1] = -cl * slo
    R[:, 2, 2] = -sl
    return R


def euler_to_rotation(roll: float, pitch: float, heading: float) -> np.ndarray:
    """
    Euler angles → 3×3 rotation matrix R_{Body←NED}.

    Convention:  R = Rz(heading) @ Ry(pitch) @ Rx(roll)

    Applanix NED:  heading about Z-Down, pitch about Y-East, roll about X-North.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)

    # Rz(h) @ Ry(p) @ Rx(r)
    R = np.array([
        [ch * cp,  ch * sp * sr - sh * cr,  ch * sp * cr + sh * sr],
        [sh * cp,  sh * sp * sr + ch * cr,  sh * sp * cr - ch * sr],
        [-sp,      cp * sr,                 cp * cr               ],
    ], dtype=np.float64)
    return R


def euler_to_rotation_batch(roll: np.ndarray, pitch: np.ndarray,
                            heading: np.ndarray) -> np.ndarray:
    """
    Vectorised Euler → (N,3,3) rotation matrices R_{Body←NED}.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)

    N = len(roll)
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = ch * cp
    R[:, 0, 1] = ch * sp * sr - sh * cr
    R[:, 0, 2] = ch * sp * cr + sh * sr
    R[:, 1, 0] = sh * cp
    R[:, 1, 1] = sh * sp * sr + ch * cr
    R[:, 1, 2] = sh * sp * cr - ch * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


# =============================================================================
# Quaternion utilities (for Slerp interpolation of attitude)
# =============================================================================

def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    3×3 rotation matrix → unit quaternion [w, x, y, z] (Shepperd's method).
    """
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def rotation_to_quaternion_batch(R: np.ndarray) -> np.ndarray:
    """
    (N,3,3) rotation matrices → (N,4) quaternions [w,x,y,z].
    """
    N = R.shape[0]
    q = np.empty((N, 4), dtype=np.float64)
    for i in range(N):
        q[i] = rotation_to_quaternion(R[i])
    # Ensure quaternion continuity (no sign flips)
    for i in range(1, N):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] *= -1.0
    return q


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """
    Unit quaternion [w, x, y, z] → 3×3 rotation matrix.
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def quaternion_to_rotation_batch(q: np.ndarray) -> np.ndarray:
    """
    (N,4) quaternions → (N,3,3) rotation matrices.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = len(w)
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two unit quaternions.

    Parameters
    ----------
    q0, q1 : (4,) quaternions [w,x,y,z]
    t      : interpolation parameter in [0, 1]

    Returns
    -------
    (4,) interpolated quaternion
    """
    dot = np.dot(q0, q1)

    # Ensure shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # Nearly parallel — linear interpolation + renormalise
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def slerp_batch(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Vectorised Slerp: interpolate between corresponding pairs of quaternions.

    Parameters
    ----------
    q0 : (N, 4) start quaternions
    q1 : (N, 4) end quaternions
    t  : (N,) interpolation parameters in [0, 1]

    Returns
    -------
    (N, 4) interpolated quaternions
    """
    dot = np.sum(q0 * q1, axis=1)

    # Flip for shortest path
    flip_mask = dot < 0
    q1 = q1.copy()
    q1[flip_mask] *= -1.0
    dot[flip_mask] *= -1.0
    dot = np.clip(dot, -1.0, 1.0)

    # Near-parallel fallback
    near = dot > 0.9995
    theta_0 = np.arccos(dot)

    # Safe division
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    # Avoid division by zero for near-parallel
    safe_denom = np.where(near, 1.0, sin_theta_0)
    s0 = np.where(near, 1.0 - t, np.cos(theta) - dot * sin_theta / safe_denom)
    s1 = np.where(near, t, sin_theta / safe_denom)

    result = s0[:, None] * q0 + s1[:, None] * q1
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / norms


def boresight_rotation(roll_rad: float, pitch_rad: float,
                       yaw_rad: float) -> np.ndarray:
    """
    Small-angle boresight correction matrix R_boresight.

    Applied as: R_cam_body = R_boresight @ R_mounting
    """
    return euler_to_rotation(roll_rad, pitch_rad, yaw_rad)
