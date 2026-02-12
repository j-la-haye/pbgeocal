
import numpy as np
from liblibor.rotations import *
from scipy.spatial.transform import Rotation as R
import numpy as np



def read_rotation_3x3_from_txt(filepath):
    """
    Reads a 3x3 rotation matrix from a text file.
    Expected format: 3 rows, 3 columns, whitespace separated.
    """
    R = np.loadtxt(filepath)
    if R.shape != (3, 3):
        raise ValueError("File must contain a 3x3 matrix.")
    return R

import numpy as np

def quaternion_to_rotation_matrix_validated(Q, tolerance=1e-6):
    """
    Converts a quaternion to a rotation matrix with normalization checks.
    
    Input:
    :param Q: A 4-element array-like quaternion (w, x, y, z)
    :param tolerance: The acceptable deviation from unit magnitude (default 1e-6)
    
    Output:
    :return: 3x3 Rotation Matrix (numpy array)
    :raises: ValueError if Q is not normalized
    """
    # Ensure input is a numpy array
    Q = np.array(Q, dtype=float)
    
    # Check shape
    if Q.shape != (4,):
        raise ValueError(f"Input must be a 4-element vector. Got shape {Q.shape}")

    # 1. Validation Step: Check Magnitude
    norm = np.linalg.norm(Q)
    
    if not np.isclose(norm, 1.0, atol=tolerance):
        raise ValueError(f"Quaternion is not normalized. Norm is {norm:.4f}, expected 1.0.")

        # Warning: Divide by zero if norm is 0
        if norm == 0:
            raise ValueError("Quaternion magnitude is 0 (cannot normalize).")
        Q = Q / norm  # Normalize the quaternion

    # Extract components (w, x, y, z)
    w, x, y, z = Q

    # 2. Computation Step (Optimized for NumPy)
    # Calculate common terms to save operations
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot_matrix = np.array([
        [1.0 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1.0 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1.0 - 2 * (xx + yy)]
    ])

    return rot_matrix

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

    Omega = skew(np.array([roll, pitch, yaw]))

    # embed into 4×4
    T_bore = np.eye(4)
    #T_bore[:3,:3] = R
    Wbore = np.eye(4)
    Wbore[:3,:3] = Omega

    # multiply
    M_Tbore = mount_matrix[:3,:3] @ R #T_bore
    Tbore_M = R @ mount_matrix[:3,:3]
    OmegaT = (np.eye(3) + Omega) @ mount_matrix[:3,:3]

    return M_Tbore, Tbore_M, R, OmegaT

# ------------------ Inverse (Extract RPY from 4x4) ------------------
def extract_bore_rpy_from_composite(T_composite, mount_matrix, order="T_bore_M",
                                    degrees=True):
    """
    Recover bore-sight RPY from a composite transform and the mount matrix.

    Args:
        T_composite : 4x4 numpy array
            Either T_bore @ mount_matrix  (order="T_bore_M")
            or     mount_matrix @ T_bore  (order="M_Tbore")
        mount_matrix : 4x4 numpy array
        order : str
            "T_bore_M" or "M_Tbore"
        degrees : bool
            Return angles in degrees if True, else radians.

    Returns:
        roll, pitch, yaw (bore-sight)
    """

    Minv = np.linalg.inv(mount_matrix)

    if order == "T_bore_M":
        # T_composite = T_bore @ M  →  T_bore = T_composite @ M⁻¹
        T_bore = T_composite @ Minv
    elif order == "M_Tbore":
        # T_composite = M @ T_bore  →  T_bore = M⁻¹ @ T_composite
        T_bore = Minv @ T_composite
    else:
        raise ValueError('order must be "T_bore_M" or "M_Tbore"')

    return extract_rpy_from_T(T_bore, degrees=degrees)

def extract_rpy_from_T(T, degrees=True):
    """
    Extract roll, pitch, yaw from 4x4 transform using ZYX convention.

    Returns:
        roll, pitch, yaw
    """

    R = T[:3, :3]

    # Guard for numerical safety
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)

    singular = sy < 1e-9

    # if abs(R[2,0]) != 1:
    #     roll = np.arctan2(R[1,2], R[2,2])
    #     pitch = -np.arcsin(R[0,2])
    #     yaw = np.arctan2(R[0,1], R[0,0])
    
    if not singular:
       roll  = np.arctan2(R[2,1], R[2,2])
       pitch = np.arctan2(-R[2,0], sy)
       yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        # Gimbal lock case
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0

    if degrees:
        return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)

    return roll, pitch, yaw


# ------------------ Example usage ------------------
if __name__ == "__main__":
    # mount = np.array([
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [1, 0, 0, 0.150150],
    #     [0, 0, 0, 1]
    # ])

    mount = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    M_T, T_M, T_bore,OmT = transform_rpy_with_mount(
        roll=0.20381,
        pitch=-0.06323,
        yaw=0.27142,
        mount_matrix=mount,
        degrees=True   # angles are already in radians
    )

    print("M @ T_bore:\n", M_T)
    print("\nT_bore @ M:\n", T_M)
    print("\nT_bore:\n", T_bore)
    print("\nOmegaT:\n", OmT)

    # ---- 1) Read composite rotation from text file ----
    #mount_path = "/media/addLidar/Projects/Lidar_Processing/0008_ITA-MonteIato_UZH_2025_AVIRIS4-1560II-SPO/01_EPFL_Proc/01_DN_Proc/250609_Montelato/ODyN_Results/VQ1560II_Lidar1-mount_opt.4x4"   # <-- your file name here
    #R_comp = read_rotation_3x3_from_txt(mount_path)

    q_wxyz = [7.0449641555784903e-01,   1.8258959781162258e-02,   2.2140635905053534e-02,   7.0912707119165308e-01]

    # Reorder to (x, y, z, w) for SciPy
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

    # Create rotation object
    r = R.from_quat(q_xyzw)

    # Get the matrix
    R_mount = r.as_matrix()

    #--- Example Usage ---

    # 1. Valid Quaternion (Normalized)
    # q_valid = [0.7071068, 0.7071068, 0, 0] 
    # try:
    #     mat = quaternion_to_rotation_matrix_validated(q_valid)
    #     print("Success: Quaternion converted.")
    # except ValueError as e:
    #     print(f"Error: {e}")

    # print("-" * 30)

    # # 2. Invalid Quaternion (Magnitude != 1)
    # q_invalid = [3.0, 1.0, 0, 0] # Norm is approx 3.16
    # try:
    #     mat = quaternion_to_rotation_matrix_validated(q_invalid)
    # except ValueError as e:
    #     print(f"Error: {e}")

    mat = quaternion_to_rotation_matrix_validated(q_wxyz, tolerance=1e-6)
    DCM = quat2dcm(q_wxyz)

    r2, p2, y2 = extract_bore_rpy_from_composite(R_mount, mount, order="T_bore_M", degrees=True)
    r3, p3, y3 = extract_bore_rpy_from_composite(R_mount, mount, order="M_Tbore", degrees=True)
    r4, p4, y4 = extract_bore_rpy_from_composite(DCM,mount,order="T_bore_M", degrees=True)

    # VQ1560II_Lidar1-mount rpy from text file: 0.20381, -0.06323, 0.27142 (in degrees)
    # AV4 0.032306,0.156340,0.066279
    roll  = np.rad2deg(0.032306)
    pitch = np.rad2deg(0.156340)
    yaw   = np.rad2deg(0.066279)

    print("Original RPY (deg):", roll, pitch, yaw)
    print("Recovered RPY from T_bore @ M:", r2, p2, y2)
    #print("Difference (deg):", r2 - roll, p2 - pitch, y2 - yaw)
    print("Recovered RPY from M @ T_bore:", r3, p3, y3)
    #print("Difference (deg):", r3 - roll, p3 - pitch, y3 - yaw)
    print("Recovered RPY from DCM:", r4, p4, y4)
    #print("Difference (deg):", r4 - roll, p4 - pitch, y4 - yaw) 