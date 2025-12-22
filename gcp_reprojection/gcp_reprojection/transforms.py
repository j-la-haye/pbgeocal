"""
Coordinate transformation module for GCP reprojection.

This module handles all coordinate system transformations:
    1. ECEF (geocentric) to local NED
    2. NED to body frame (using trajectory orientation)
    3. Body frame to camera frame
    
Coordinate System Definitions:
    - ECEF: Earth-Centered, Earth-Fixed (X towards 0°lon, Y towards 90°E, Z towards North Pole)
    - NED: North-East-Down (local tangent plane)
    - Body: Forward-Right-Down (SBET/aircraft convention)
    - Camera: X-right, Y-back, Z-down (as specified)

Rotation Conventions:
    - All rotations use right-hand rule
    - Euler angles applied in ZYX order (yaw, pitch, roll)
    - Angles in SBET are typically in radians for internal use
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # First eccentricity squared


@dataclass
class TrajectoryPose:
    """
    Trajectory pose at a specific image capture time.
    
    Position is in WGS84 geodetic coordinates.
    Orientation is in local NED frame using aerospace (ZYX) Euler angles.
    
    Attributes:
        latitude: Geodetic latitude in degrees
        longitude: Geodetic longitude in degrees
        height: Ellipsoidal height in meters (above WGS84 ellipsoid)
        roll: Roll angle in degrees (rotation about forward axis)
        pitch: Pitch angle in degrees (rotation about right axis)
        yaw: Yaw/heading angle in degrees (rotation about down axis, from North)
    """
    latitude: float
    longitude: float
    height: float
    roll: float
    pitch: float
    yaw: float


class CoordinateTransformer:
    """
    Handles coordinate transformations for GCP reprojection.
    
    The transformation chain is:
        GCP(ECEF) → NED → Body → Camera → Image
        
    Key considerations:
        1. The camera position in ECEF is computed from trajectory WGS84 position
        2. The lever arm offset is applied in body frame before rotation
        3. The boresight correction is applied between body and camera frames
    """
    
    # Fixed rotation from body frame (Forward-Right-Down) to camera frame (Right-Back-Down)
    # Camera X = Body Y (right)
    # Camera Y = -Body X (back = -forward)  
    # Camera Z = Body Z (down)
    R_CAM_BODY = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    def __init__(
        self,
        boresight_roll: float = 0.0,
        boresight_pitch: float = 0.0,
        boresight_yaw: float = 0.0,
        lever_arm_x: float = 0.0,
        lever_arm_y: float = 0.0,
        lever_arm_z: float = 0.0,
    ):
        """
        Initialize the coordinate transformer.
        
        Args:
            boresight_roll: Boresight roll offset in degrees
            boresight_pitch: Boresight pitch offset in degrees
            boresight_yaw: Boresight yaw offset in degrees
            lever_arm_x: Lever arm X offset in meters (forward in body frame)
            lever_arm_y: Lever arm Y offset in meters (right in body frame)
            lever_arm_z: Lever arm Z offset in meters (down in body frame)
        """
        # Convert boresight to radians and compute correction rotation
        self.boresight_rad = np.deg2rad([boresight_roll, boresight_pitch, boresight_yaw])
        self.R_boresight = self._euler_to_rotation_matrix(*self.boresight_rad)
        
        # Lever arm in body frame (meters)
        self.lever_arm_body = np.array([lever_arm_x, lever_arm_y, lever_arm_z])
        
        logger.debug(f"Initialized transformer with boresight: {self.boresight_rad} rad")
        logger.debug(f"Lever arm (body frame): {self.lever_arm_body} m")
    
    @staticmethod
    def _euler_to_rotation_matrix(
        roll: float, pitch: float, yaw: float
    ) -> np.ndarray:
        """
        Compute rotation matrix from Euler angles (ZYX convention).
        
        This follows the aerospace convention where:
            - Yaw (ψ) rotates about Z (down)
            - Pitch (θ) rotates about Y (right)
            - Roll (φ) rotates about X (forward)
        
        The combined rotation is: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        
        This gives the rotation FROM NED TO body frame.
        
        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Rotation about X (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])
        
        # Rotation about Y (pitch)
        Ry = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ])
        
        # Rotation about Z (yaw)
        Rz = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz @ Ry @ Rx
        return Rz @ Ry @ Rx
    
    @staticmethod
    def geodetic_to_ecef(
        lat: float, lon: float, h: float
    ) -> np.ndarray:
        """
        Convert geodetic coordinates (WGS84) to ECEF.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            h: Ellipsoidal height in meters
            
        Returns:
            ECEF coordinates as (X, Y, Z) in meters
            
        Reference:
            NIMA TR8350.2, "Department of Defense World Geodetic System 1984"
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        
        # Radius of curvature in the prime vertical
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
        
        X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
        Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
        Z = (N * (1 - WGS84_E2) + h) * np.sin(lat_rad)
        
        return np.array([X, Y, Z])
    
    @staticmethod
    def ecef_to_ned_rotation(lat: float, lon: float) -> np.ndarray:
        """
        Compute rotation matrix from ECEF to local NED frame.
        
        The NED frame is defined at a point on the Earth's surface:
            - N (North): Tangent to ellipsoid, pointing north
            - E (East): Tangent to ellipsoid, pointing east
            - D (Down): Normal to ellipsoid, pointing toward Earth center
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            3x3 rotation matrix from ECEF to NED
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        
        clat, slat = np.cos(lat_rad), np.sin(lat_rad)
        clon, slon = np.cos(lon_rad), np.sin(lon_rad)
        
        # Rotation matrix from ECEF to NED
        # Row 1: North direction in ECEF
        # Row 2: East direction in ECEF
        # Row 3: Down direction in ECEF
        R_ned_ecef = np.array([
            [-slat * clon, -slat * slon, clat],
            [-slon, clon, 0],
            [-clat * clon, -clat * slon, -slat]
        ])
        
        return R_ned_ecef
    
    def compute_camera_position_ecef(
        self, pose: TrajectoryPose
    ) -> np.ndarray:
        """
        Compute camera position in ECEF, accounting for lever arm.
        
        The lever arm offset is defined in the body frame and must be
        rotated to ECEF before being added to the IMU position.
        
        Args:
            pose: Trajectory pose (IMU position and orientation)
            
        Returns:
            Camera position in ECEF (meters)
        """
        # IMU position in ECEF
        imu_ecef = self.geodetic_to_ecef(pose.latitude, pose.longitude, pose.height)
        
        if np.allclose(self.lever_arm_body, 0):
            return imu_ecef
        
        # Rotation from NED to ECEF (transpose of ECEF to NED)
        R_ecef_ned = self.ecef_to_ned_rotation(pose.latitude, pose.longitude).T
        
        # Rotation from body to NED (transpose of NED to body)
        roll_rad = np.deg2rad(pose.roll)
        pitch_rad = np.deg2rad(pose.pitch)
        yaw_rad = np.deg2rad(pose.yaw)
        R_body_ned = self._euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
        R_ned_body = R_body_ned.T
        
        # Transform lever arm from body to ECEF
        lever_arm_ned = R_ned_body @ self.lever_arm_body
        lever_arm_ecef = R_ecef_ned @ lever_arm_ned
        
        return imu_ecef + lever_arm_ecef
    
    def ecef_to_camera_frame(
        self,
        point_ecef: np.ndarray,
        pose: TrajectoryPose,
    ) -> np.ndarray:
        """
        Transform a point from ECEF to camera frame.
        
        Transformation chain:
            1. Translate to camera-centered ECEF (account for lever arm)
            2. Rotate ECEF to NED
            3. Rotate NED to body frame
            4. Apply boresight correction
            5. Rotate body to camera frame
        
        Args:
            point_ecef: Point coordinates in ECEF (meters)
            pose: Camera/IMU trajectory pose
            
        Returns:
            Point coordinates in camera frame (meters)
        """
        # Step 1: Get camera position in ECEF and compute relative position
        camera_ecef = self.compute_camera_position_ecef(pose)
        delta_ecef = point_ecef - camera_ecef
        
        # Step 2: Rotate to NED
        R_ned_ecef = self.ecef_to_ned_rotation(pose.latitude, pose.longitude)
        point_ned = R_ned_ecef @ delta_ecef
        
        # Step 3: Rotate NED to body frame
        roll_rad = np.deg2rad(pose.roll)
        pitch_rad = np.deg2rad(pose.pitch)
        yaw_rad = np.deg2rad(pose.yaw)
        R_body_ned = self._euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
        point_body = R_body_ned @ point_ned
        
        # Step 4: Apply boresight correction (small angular offset)
        point_body_corrected = self.R_boresight @ point_body
        
        # Step 5: Rotate body to camera frame
        point_camera = self.R_CAM_BODY @ point_body_corrected
        
        return point_camera
    
    def transform_points_batch(
        self,
        points_ecef: np.ndarray,
        pose: TrajectoryPose,
    ) -> np.ndarray:
        """
        Transform multiple points from ECEF to camera frame.
        
        Args:
            points_ecef: Nx3 array of ECEF coordinates
            pose: Camera trajectory pose
            
        Returns:
            Nx3 array of camera frame coordinates
        """
        results = np.zeros_like(points_ecef)
        for i, point in enumerate(points_ecef):
            results[i] = self.ecef_to_camera_frame(point, pose)
        return results


def validate_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Validate that a matrix is a proper rotation matrix.
    
    A proper rotation matrix must:
        1. Be orthogonal: R @ R.T = I
        2. Have determinant = +1 (not a reflection)
    
    Args:
        R: 3x3 matrix to validate
        tol: Numerical tolerance
        
    Returns:
        True if R is a valid rotation matrix
    """
    if R.shape != (3, 3):
        return False
    
    # Check orthogonality
    should_be_identity = R @ R.T
    if not np.allclose(should_be_identity, np.eye(3), atol=tol):
        return False
    
    # Check determinant
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=tol):
        return False
    
    return True
