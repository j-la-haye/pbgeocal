import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

c = np.cos
s = np.sin

#mapping frame m refers to local enu tangent plane with specified, fixed, origin.
def R1(r):
    """
    Rotation matrix around the x-axis, r in radians
    """
    return np.array([[1,    0,    0],
                     [0, c(r), -s(r)],
                     [0,s(r), c(r)]])

def R2(p):
    """
    Rotation matrix around the y-axis, p in radians
    """
    return np.array([[c(p), 0,s(p)],
                     [   0, 1,    0],
                     [-s(p), 0, c(p)]])

def R3(y):
    """
    Rotation matrix around the z-axis, y in radians
    """
    return np.array([[ c(y), -s(y), 0],
                     [s(y), c(y), 0],
                     [    0,    0, 1]])

def D1(r):
    """
    Derivative of rotation matrix around the x-axis, r in radians
    """
    return np.array([[0,     0,     0],
                     [0,-s(r), c(r)],
                     [0,-c(r),-s(r)]])

def D2(p):
    """
    Derivative of rotation matrix around the y-axis, p in radians
    """
    return np.array([[-s(p), 0,-c(p)],
                     [    0, 0,     0],
                     [ c(p), 0,-s(p)]])

def D3(y):
    """
    Derivative of rotation matrix around the z-axis, y in radians
    """
    return np.array([[-s(y), c(y), 0],
                     [-c(y),-s(y), 0],
                     [    0,    0, 0]])

def R_b2ned(r, p, y):
    """
    Rotation matrix from body to local NED frame given roll (r), pitch (p), yaw (y) in radians
    """
    return (R1(r) @ R2(p) @ R3(y)).T
    return (R3(y) @ R2(p) @ R1(r))

def R_ned2b(r, p, y):
    """
    Rotation matrix from local NED to body frame given roll (r), pitch (p), yaw (y) in radians
    """
    return R1(r) @ R2(p) @ R3(y)
    #return (R3(y) @ R2(p) @ R1(r))
    

def dR_b2ned_dr(r, p, y):
    """
    Derivative of rotation matrix from body to local NED frame with respect to roll (r) in radians
    """
    return R3(y).T @ R2(p).T @ D1(r).T

def dR_b2ned_dp(r, p, y):
    """
    Derivative of rotation matrix from body to local NED frame with respect to pitch (p) in radians
    """
    return R3(y).T @ D2(p).T @ R1(r).T

def dR_b2ned_dy(r, p, y):
    """
    Derivative of rotation matrix from body to local NED frame with respect to yaw (y) in radians
    """
    return D3(y).T @ R2(p).T @ R1(r).T

def R_ned2e(lat,lon):
    """
    Rotation matrix from local level NED to ECEF frame.
    :param lat, lon: latitude and longitude in radians
    :return: rotation matrix
    """
    return np.array([[ -s(lat)*c(lon),-s(lon), -c(lat)*c(lon)],
                     [ -s(lat)*s(lon), c(lon), -c(lat)*s(lon)],
                     [         c(lat),      0,        -s(lat)]])


def T_enu_ned():
    """
    Rotation matrix from local level ENU to NED frame and vice versa
    """
    return np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0,-1]])

def R_b2m(lat, lon, r, p, y, R_e2m):
    """
    Rotation matrix from body to mapping enu frame 
    """

    R_b2ned_matrix = R_b2ned(r, p, y)

    R_ned2e_matrix = R_ned2e(lat, lon)

    return R_e2m @ R_ned2e_matrix @ R_b2ned_matrix

def R_e2enu(lat, lon):
    """
    Rotation matrix from ECEF to enu frame 
    """

    R_ned2e_matrix = R_ned2e(lat, lon)

    return T_enu_ned() @ R_ned2e_matrix.T 


def R_ecef2enu(lon, lat):
        """
        Calculates the rotation matrix from ECEF to the local ENU frame.
        The rows of this matrix are the E, N, and U unit vectors in ECEF.
        """
        # Convert to radians
        lam = np.deg2rad(lon)
        phi = np.deg2rad(lat)

        # ENU unit vectors in ECEF coordinates
        # East  = [-sin(λ),           cos(λ),          0]
        # North = [-sin(φ)cos(λ), -sin(φ)sin(λ),  cos(φ)]
        # Up    = [ cos(φ)cos(λ),  cos(φ)sin(λ),  sin(φ)]
        
        R_ecef_enu = np.array([
            [-np.sin(lam),           np.cos(lam),          0],
            [-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam),  np.cos(phi)],
            [ np.cos(phi)*np.cos(lam),  np.cos(phi)*np.sin(lam),  np.sin(phi)]
        ])
        return R_ecef_enu

def skew(u):
    """
    Skew symmetric matrix of a vector
    """
    assert u.shape == (3,1) or u.shape == (3,)
    u = u.flatten()

    return np.array([[   0, -u[2],  u[1]],
                     [ u[2],     0, -u[0]],
                     [-u[1],  u[0],     0]])

def skewT(u):
    """
    Transpose of skew symmetric matrix of a vector
    """
    assert u.shape == (3,1) or u.shape == (3,)
    u = u.flatten()

    return np.array([[   0,  u[2], -u[1]],
                     [-u[2],     0,  u[0]],
                     [ u[1], -u[0],     0]])

def dcm2quat_scipy(R):
    quat = R_scipy.from_matrix(R).as_quat()  # Returns [qx, qy, qz, qw]
    # Reorder to [qw, qx, qy, qz]
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    return quat

def dcm2quat(R):
    """
    Convert rotation matrix to quaternion representation
    :param R: rotation matrix
    :return: quaternion [qw, qx, qy, qz]
    """
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return np.array([qw, qx, qy, qz])

def quat2dcm(q):
    """
    Convert a quaternion into a dcm matrix.

    Parameters:
    q : ndarray shape (4,)
        Quaternion in the form [w, x, y, z]

    Returns:
    ndarray shape (3, 3)
        Rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [w*w + x*x - y*y - z*z,     2*(x*y - w*z),         2*(x*z + w*y)],
        [2*(x*y + w*z),             w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
        [2*(x*z - w*y),             2*(y*z + w*x),         w*w - x*x - y*y + z*z]
    ])


def rpy_from_R_ned2b(R, as_degrees=False):
    """
    Extract roll, pitch, yaw from rotation matrix from ned to body, SO eq. 3.21
    """
    if abs(R[2,0]) != 1:
        r = np.arctan2(R[1,2], R[2,2])
        p = -np.arcsin(R[0,2])
        y = np.arctan2(R[0,1], R[0,0])
    #TODO: Add gimbal lock case support

    if as_degrees:
        return np.array([r, p, y])*180/np.pi
    else:
        return np.array([r, p, y])

def r_mat_to_opk(R_wc, return_radians=False,camZdown=True):
    """World b→camera(ENU) rotation to photogrammetric (ω,φ,κ)."""
    # CAMEO Socet convention
    if abs(R_wc[2,0]) != 1:
        omega = np.atan2(-R_wc[1, 2], R_wc[2, 2])
        phi = np.asin(R_wc[0, 2])
        kappa = np.atan2(-R_wc[0, 1], R_wc[0, 0])
        if camZdown:
            kappa = -kappa

    if return_radians:
        return np.array([omega, phi, kappa])
    return np.degrees([omega, phi, kappa])
    
    #omega = np.arctan2(-c[1, 2], c[2, 2])
    #phi   = np.arcsin(c[0, 2])
    #kappa = np.arctan2(-c[0, 1], c[0, 0])

def T_b2cam():
    """
    Rotation matrix from camera to NED frame.
    """
    return np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])
def T_cam_z_90():
    """
    Rotation matrix from 90 camera about Z axis.
    """
    return np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

def rpy_to_opk(roll, pitch, heading,radians=True):
    if radians:
        r, p, h = np.array([roll, pitch, heading])
    else:
        r, p, h = np.deg2rad([roll, pitch, heading])
    R_ned2b = R3(h) @ R2(p) @ R1(r)
    R_enu2b = T_enu_ned() @ R_ned2b @ T_enu_ned() # world (ENU) → camera
    R_enu2c = T_b2cam() @ R_enu2b   # world (NED) → camera
    

    r00, r01, r02 = R_enu2c[0]
    r10, r11, r12 = R_enu2c[1]
    r20, r21, r22 = R_enu2c[2]

    if abs(r02) >= 0.9999:  # Gimbal lock
        phi = np.arcsin(np.clip(r02, -1, 1))
        if phi > 0:  # phi ≈ 90°
            omega = 0
            kappa = np.arctan2(r10, r11)
        else:  # phi ≈ -90°
            omega = 0
            kappa = np.arctan2(-r10, r11)
    else:
        phi   = np.arcsin(r02)
        omega = np.arctan2(-r12, r22)
        # negate kappa to account for z axis down but camera is in ENU frame
        kappa = -np.arctan2(-r01, r00)
    
    return np.rad2deg([omega, phi, kappa])  # ω, φ, κ in degrees


def transform_rpy_with_mount(roll, pitch, yaw, mount_matrix, degrees=True):
    """
    Convert RPY to a 4x4 transform and multiply with a mounting matrix.

    Args:
        roll, pitch, yaw : floats
            Euler angles (RPY).
        mount_matrix : (4x4) numpy array
            Sensor/IMU transformation matrix.
        degrees : bool
            True = angles in degrees, False = radians.

    Returns:
        M_Tbore : numpy array
            mount_matrix @ bore_matrix
        Tbore_M : numpy array
            bore_matrix @ mount_matrix
        bore_matrix : numpy array
            4x4 rotation transform from RPY
    """

    # Convert from degrees if necessary
    if degrees:
        roll  = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw   = np.deg2rad(yaw)

    # Rotation matrices
    def R_x(r):
        return np.array([[1,0,0],
                        [0,np.cos(r),-np.sin(r)],
                        [0,np.sin(r), np.cos(r)]])

    def R_y(p):
        return np.array([[ np.cos(p),0,np.sin(p)],
                         [0,1,0],
                         [-np.sin(p),0,np.cos(p)]])

    def R_z(y):
        return np.array([[np.cos(y),-np.sin(y),0],
                         [np.sin(y), np.cos(y),0],
                         [0,0,1]])

    # yaw–pitch–roll: Z * Y * X
    R = R_z(yaw) @ R_y(pitch) @ R_x(roll)

    # embed into 4×4
    T_bore = np.eye(4)
    T_bore[:3,:3] = R

    # multiply
    M_Tbore = mount_matrix @ T_bore
    Tbore_M = T_bore @ mount_matrix

    return M_Tbore, Tbore_M, T_bore



