"""
Tests for coordinate transformation module.

These tests verify the correctness of:
    - Geodetic to ECEF conversion
    - ECEF to NED rotation
    - Euler angle to rotation matrix conversion
    - Full transformation chain
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from gcp_reprojection.transforms import (
    CoordinateTransformer,
    TrajectoryPose,
    validate_rotation_matrix,
    WGS84_A,
    WGS84_B,
)


class TestGeodeticToECEF:
    """Tests for geodetic to ECEF conversion."""
    
    def test_equator_prime_meridian(self):
        """Point at equator/prime meridian should be on +X axis."""
        result = CoordinateTransformer.geodetic_to_ecef(0, 0, 0)
        
        # At equator, prime meridian, height=0: X ≈ a, Y = 0, Z = 0
        assert_allclose(result[0], WGS84_A, rtol=1e-10)
        assert_allclose(result[1], 0, atol=1e-10)
        assert_allclose(result[2], 0, atol=1e-10)
    
    def test_equator_90_east(self):
        """Point at equator/90°E should be on +Y axis."""
        result = CoordinateTransformer.geodetic_to_ecef(0, 90, 0)
        
        assert_allclose(result[0], 0, atol=1e-6)
        assert_allclose(result[1], WGS84_A, rtol=1e-10)
        assert_allclose(result[2], 0, atol=1e-10)
    
    def test_north_pole(self):
        """North pole should be on +Z axis."""
        result = CoordinateTransformer.geodetic_to_ecef(90, 0, 0)
        
        assert_allclose(result[0], 0, atol=1e-6)
        assert_allclose(result[1], 0, atol=1e-6)
        # At pole, Z ≈ b (semi-minor axis)
        assert_allclose(result[2], WGS84_B, rtol=1e-10)
    
    def test_south_pole(self):
        """South pole should be on -Z axis."""
        result = CoordinateTransformer.geodetic_to_ecef(-90, 0, 0)
        
        assert_allclose(result[0], 0, atol=1e-6)
        assert_allclose(result[1], 0, atol=1e-6)
        assert_allclose(result[2], -WGS84_B, rtol=1e-10)
    
    def test_height_increases_distance(self):
        """Adding height should increase distance from center."""
        result_0 = CoordinateTransformer.geodetic_to_ecef(45, 45, 0)
        result_1000 = CoordinateTransformer.geodetic_to_ecef(45, 45, 1000)
        
        dist_0 = np.linalg.norm(result_0)
        dist_1000 = np.linalg.norm(result_1000)
        
        assert dist_1000 - dist_0 == pytest.approx(1000, rel=0.01)
    
    def test_known_location(self):
        """Test with a known location (approximate)."""
        # Mount Everest summit: 27.9881° N, 86.9250° E, ~8848m
        result = CoordinateTransformer.geodetic_to_ecef(27.9881, 86.9250, 8848)
        
        # Just verify reasonable magnitude
        dist = np.linalg.norm(result)
        assert dist > WGS84_B  # Further than polar radius
        assert dist < WGS84_A + 10000  # Not too far


class TestECEFToNEDRotation:
    """Tests for ECEF to NED rotation matrix."""
    
    def test_equator_prime_meridian(self):
        """At equator/prime meridian, NED should align with specific ECEF axes."""
        R = CoordinateTransformer.ecef_to_ned_rotation(0, 0)
        
        assert validate_rotation_matrix(R)
        
        # At (0, 0): North points -Z in ECEF, East points +Y, Down points -X
        north_ecef = R.T @ np.array([1, 0, 0])  # North in ECEF
        east_ecef = R.T @ np.array([0, 1, 0])   # East in ECEF
        down_ecef = R.T @ np.array([0, 0, 1])   # Down in ECEF
        
        assert_allclose(north_ecef, [0, 0, 1], atol=1e-10)
        assert_allclose(east_ecef, [0, 1, 0], atol=1e-10)
        assert_allclose(down_ecef, [-1, 0, 0], atol=1e-10)
    
    def test_north_pole(self):
        """At north pole, down should point along -Z in ECEF."""
        R = CoordinateTransformer.ecef_to_ned_rotation(90, 0)
        
        assert validate_rotation_matrix(R)
        
        down_ecef = R.T @ np.array([0, 0, 1])
        assert_allclose(down_ecef, [0, 0, -1], atol=1e-10)
    
    def test_rotation_is_orthogonal(self):
        """Rotation matrix should be orthogonal at various locations."""
        locations = [
            (0, 0), (45, 45), (-30, 120), (89, -45), (-89, 180),
        ]
        
        for lat, lon in locations:
            R = CoordinateTransformer.ecef_to_ned_rotation(lat, lon)
            assert validate_rotation_matrix(R), f"Failed at ({lat}, {lon})"


class TestEulerToRotationMatrix:
    """Tests for Euler angle to rotation matrix conversion."""
    
    def test_identity(self):
        """Zero angles should give identity matrix."""
        R = CoordinateTransformer._euler_to_rotation_matrix(0, 0, 0)
        assert_allclose(R, np.eye(3), atol=1e-10)
    
    def test_pure_roll(self):
        """Test pure roll rotation."""
        roll = np.pi / 4  # 45 degrees
        R = CoordinateTransformer._euler_to_rotation_matrix(roll, 0, 0)
        
        assert validate_rotation_matrix(R)
        
        # Roll about X: Y rotates toward Z
        y_axis = R @ np.array([0, 1, 0])
        expected = np.array([0, np.cos(roll), np.sin(roll)])
        assert_allclose(y_axis, expected, atol=1e-10)
    
    def test_pure_pitch(self):
        """Test pure pitch rotation."""
        pitch = np.pi / 6  # 30 degrees
        R = CoordinateTransformer._euler_to_rotation_matrix(0, pitch, 0)
        
        assert validate_rotation_matrix(R)
        
        # Pitch about Y: X rotates toward -Z (nose up means +Z goes forward)
        x_axis = R @ np.array([1, 0, 0])
        expected = np.array([np.cos(pitch), 0, -np.sin(pitch)])
        assert_allclose(x_axis, expected, atol=1e-10)
    
    def test_pure_yaw(self):
        """Test pure yaw rotation."""
        yaw = np.pi / 3  # 60 degrees
        R = CoordinateTransformer._euler_to_rotation_matrix(0, 0, yaw)
        
        assert validate_rotation_matrix(R)
        
        # Yaw about Z: X rotates toward Y
        x_axis = R @ np.array([1, 0, 0])
        expected = np.array([np.cos(yaw), np.sin(yaw), 0])
        assert_allclose(x_axis, expected, atol=1e-10)
    
    def test_combined_rotation(self):
        """Combined rotation should still be orthogonal."""
        roll = np.deg2rad(10)
        pitch = np.deg2rad(-5)
        yaw = np.deg2rad(45)
        
        R = CoordinateTransformer._euler_to_rotation_matrix(roll, pitch, yaw)
        assert validate_rotation_matrix(R)


class TestBodyToCameraRotation:
    """Tests for the fixed body-to-camera rotation."""
    
    def test_rotation_is_valid(self):
        """Body-to-camera rotation should be a valid rotation matrix."""
        assert validate_rotation_matrix(CoordinateTransformer.R_CAM_BODY)
    
    def test_axis_mapping(self):
        """Verify axis mapping from body to camera frame."""
        R = CoordinateTransformer.R_CAM_BODY
        
        # Body X (forward) should map to Camera -Y (back)
        body_x = np.array([1, 0, 0])
        camera_result = R @ body_x
        assert_allclose(camera_result, [0, -1, 0], atol=1e-10)
        
        # Body Y (right) should map to Camera X (right)
        body_y = np.array([0, 1, 0])
        camera_result = R @ body_y
        assert_allclose(camera_result, [1, 0, 0], atol=1e-10)
        
        # Body Z (down) should map to Camera Z (down)
        body_z = np.array([0, 0, 1])
        camera_result = R @ body_z
        assert_allclose(camera_result, [0, 0, 1], atol=1e-10)


class TestCoordinateTransformer:
    """Integration tests for the full transformation chain."""
    
    @pytest.fixture
    def transformer(self):
        """Create a transformer with no boresight or lever arm."""
        return CoordinateTransformer()
    
    @pytest.fixture
    def transformer_with_offsets(self):
        """Create a transformer with boresight and lever arm."""
        return CoordinateTransformer(
            boresight_roll=1.0,
            boresight_pitch=-0.5,
            boresight_yaw=0.2,
            lever_arm_x=0.5,
            lever_arm_y=0.1,
            lever_arm_z=-0.2,
        )
    
    def test_point_directly_ahead(self, transformer):
        """
        A point directly ahead of the camera should have positive Y in camera frame.
        
        Setup: Camera at (0, 0, 1000m), pointing north, level
        GCP: 100m north of camera position
        Expected: Point should be at (0, +, +) in camera frame
        """
        # Camera pose: at origin, 1000m up, facing north, level
        pose = TrajectoryPose(
            latitude=0.0,
            longitude=0.0,
            height=1000.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,  # Facing north
        )
        
        # GCP: 100m north (approximately)
        gcp_ecef = CoordinateTransformer.geodetic_to_ecef(0.001, 0, 1000)
        
        point_camera = transformer.ecef_to_camera_frame(gcp_ecef, pose)
        
        # In camera frame (X-right, Y-back, Z-down):
        # Point ahead should have negative Y (forward = -Y in camera)
        assert point_camera[1] < 0, "Point ahead should have negative Y in camera frame"
    
    def test_point_to_right(self, transformer):
        """
        A point to the right of the camera should have positive X in camera frame.
        """
        pose = TrajectoryPose(
            latitude=0.0,
            longitude=0.0,
            height=1000.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )
        
        # GCP: ~100m east
        gcp_ecef = CoordinateTransformer.geodetic_to_ecef(0, 0.001, 1000)
        
        point_camera = transformer.ecef_to_camera_frame(gcp_ecef, pose)
        
        # Right in body frame = right in camera frame (+X)
        assert point_camera[0] > 0, "Point to right should have positive X"
    
    def test_point_below(self, transformer):
        """
        A point below the camera should have positive Z in camera frame.
        """
        pose = TrajectoryPose(
            latitude=0.0,
            longitude=0.0,
            height=1000.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )
        
        # GCP: directly below (on ground)
        gcp_ecef = CoordinateTransformer.geodetic_to_ecef(0, 0, 0)
        
        point_camera = transformer.ecef_to_camera_frame(gcp_ecef, pose)
        
        # Down in both body and camera frames = +Z
        assert point_camera[2] > 0, "Point below should have positive Z"
    
    def test_yaw_rotation(self, transformer):
        """
        Test that yaw rotation affects the camera frame coordinates.
        
        With yaw=90° (heading east), a point to the north should appear
        in a different position than when facing north.
        """
        # Reference pose facing north
        pose_north = TrajectoryPose(
            latitude=0.0,
            longitude=0.0,
            height=1000.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,  # Facing north
        )
        
        # Pose facing east
        pose_east = TrajectoryPose(
            latitude=0.0,
            longitude=0.0,
            height=1000.0,
            roll=0.0,
            pitch=0.0,
            yaw=90.0,  # Facing east
        )
        
        # GCP: 100m north
        gcp_ecef = CoordinateTransformer.geodetic_to_ecef(0.001, 0, 1000)
        
        point_north = transformer.ecef_to_camera_frame(gcp_ecef, pose_north)
        point_east = transformer.ecef_to_camera_frame(gcp_ecef, pose_east)
        
        # The X coordinate should be different between the two orientations
        # When facing north, point ahead should have small X
        # When facing east, the same point is no longer ahead
        assert abs(point_east[0]) > abs(point_north[0]), \
            "Point position should change with yaw rotation"
    
    def test_lever_arm_effect(self, transformer_with_offsets):
        """Lever arm should shift the effective camera position."""
        pose = TrajectoryPose(
            latitude=0.0,
            longitude=0.0,
            height=1000.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )
        
        # Compute camera position with and without lever arm
        cam_pos_with = transformer_with_offsets.compute_camera_position_ecef(pose)
        
        transformer_no_lever = CoordinateTransformer(
            boresight_roll=1.0,
            boresight_pitch=-0.5,
            boresight_yaw=0.2,
        )
        cam_pos_without = transformer_no_lever.compute_camera_position_ecef(pose)
        
        # Positions should differ due to lever arm
        diff = np.linalg.norm(cam_pos_with - cam_pos_without)
        expected_diff = np.linalg.norm([0.5, 0.1, -0.2])
        
        assert_allclose(diff, expected_diff, rtol=0.01)


class TestValidateRotationMatrix:
    """Tests for rotation matrix validation."""
    
    def test_valid_identity(self):
        """Identity is a valid rotation."""
        assert validate_rotation_matrix(np.eye(3))
    
    def test_valid_rotation(self):
        """Random rotation should be valid."""
        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        assert validate_rotation_matrix(R)
    
    def test_invalid_reflection(self):
        """Reflection (det=-1) should be invalid."""
        reflection = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        assert not validate_rotation_matrix(reflection)
    
    def test_invalid_scaling(self):
        """Scaled matrix should be invalid."""
        scaled = 2 * np.eye(3)
        assert not validate_rotation_matrix(scaled)
    
    def test_invalid_shape(self):
        """Wrong shape should be invalid."""
        assert not validate_rotation_matrix(np.eye(2))
        assert not validate_rotation_matrix(np.eye(4))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
