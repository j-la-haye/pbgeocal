"""
Tests for trajectory interpolation module.
"""

import pytest
import numpy as np
import tempfile
from numpy.testing import assert_allclose

from gcp_reprojection.trajectory_interpolator import (
    TrajectoryInterpolator,
    TrajectoryEpoch,
    load_trajectory_interpolator,
)
from gcp_reprojection.transforms import TrajectoryPose


class TestTrajectoryInterpolator:
    """Tests for trajectory interpolation."""
    
    @pytest.fixture
    def simple_trajectory(self):
        """Create a simple trajectory for testing."""
        epochs = [
            TrajectoryEpoch(
                time=0.0,
                pose=TrajectoryPose(
                    latitude=45.0,
                    longitude=-122.0,
                    height=1000.0,
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )
            ),
            TrajectoryEpoch(
                time=1.0,
                pose=TrajectoryPose(
                    latitude=45.001,
                    longitude=-121.999,
                    height=1010.0,
                    roll=1.0,
                    pitch=-1.0,
                    yaw=10.0,
                )
            ),
            TrajectoryEpoch(
                time=2.0,
                pose=TrajectoryPose(
                    latitude=45.002,
                    longitude=-121.998,
                    height=1020.0,
                    roll=2.0,
                    pitch=-2.0,
                    yaw=20.0,
                )
            ),
        ]
        return TrajectoryInterpolator(epochs)
    
    def test_interpolate_at_epoch(self, simple_trajectory):
        """Test interpolation exactly at an epoch time."""
        pose = simple_trajectory.interpolate(0.0)
        
        assert pose is not None
        assert pose.latitude == pytest.approx(45.0)
        assert pose.longitude == pytest.approx(-122.0)
        assert pose.height == pytest.approx(1000.0)
    
    def test_interpolate_midpoint(self, simple_trajectory):
        """Test interpolation at midpoint between epochs."""
        pose = simple_trajectory.interpolate(0.5)
        
        assert pose is not None
        # Should be average of first two epochs
        assert pose.latitude == pytest.approx(45.0005)
        assert pose.longitude == pytest.approx(-121.9995)
        assert pose.height == pytest.approx(1005.0)
        assert pose.roll == pytest.approx(0.5)
        assert pose.pitch == pytest.approx(-0.5)
        assert pose.yaw == pytest.approx(5.0)
    
    def test_interpolate_quarter_point(self, simple_trajectory):
        """Test interpolation at 25% between epochs."""
        pose = simple_trajectory.interpolate(0.25)
        
        assert pose is not None
        assert pose.height == pytest.approx(1002.5)
        assert pose.roll == pytest.approx(0.25)
    
    def test_interpolate_outside_range_before(self, simple_trajectory):
        """Test interpolation before trajectory start."""
        pose = simple_trajectory.interpolate(-1.0)
        assert pose is None
    
    def test_interpolate_outside_range_after(self, simple_trajectory):
        """Test interpolation after trajectory end."""
        pose = simple_trajectory.interpolate(3.0)
        assert pose is None
    
    def test_interpolate_at_end(self, simple_trajectory):
        """Test interpolation at trajectory end."""
        pose = simple_trajectory.interpolate(2.0)
        
        assert pose is not None
        assert pose.latitude == pytest.approx(45.002)
        assert pose.height == pytest.approx(1020.0)
    
    def test_time_range(self, simple_trajectory):
        """Test time range properties."""
        assert simple_trajectory.time_start == pytest.approx(0.0)
        assert simple_trajectory.time_end == pytest.approx(2.0)
    
    def test_unsorted_input(self):
        """Test that unsorted input epochs are sorted."""
        epochs = [
            TrajectoryEpoch(time=2.0, pose=TrajectoryPose(0, 0, 100, 0, 0, 0)),
            TrajectoryEpoch(time=0.0, pose=TrajectoryPose(0, 0, 0, 0, 0, 0)),
            TrajectoryEpoch(time=1.0, pose=TrajectoryPose(0, 0, 50, 0, 0, 0)),
        ]
        interp = TrajectoryInterpolator(epochs)
        
        # Should still work correctly
        pose = interp.interpolate(0.5)
        assert pose is not None
        assert pose.height == pytest.approx(25.0)
    
    def test_empty_epochs(self):
        """Test error on empty epochs."""
        with pytest.raises(ValueError):
            TrajectoryInterpolator([])


class TestAngleInterpolation:
    """Tests for angle interpolation with wrap-around."""
    
    @pytest.fixture
    def yaw_wrap_trajectory(self):
        """Trajectory with yaw wrap-around (350° to 10°)."""
        epochs = [
            TrajectoryEpoch(
                time=0.0,
                pose=TrajectoryPose(0, 0, 0, 0, 0, 350.0)
            ),
            TrajectoryEpoch(
                time=1.0,
                pose=TrajectoryPose(0, 0, 0, 0, 0, 10.0)
            ),
        ]
        return TrajectoryInterpolator(epochs)
    
    def test_yaw_wrap_around(self, yaw_wrap_trajectory):
        """Test that yaw interpolation handles wrap-around correctly."""
        pose = yaw_wrap_trajectory.interpolate(0.5)
        
        assert pose is not None
        # Should interpolate through 0° (360°), not through 180°
        # Midpoint should be around 0° (or 360°)
        # The difference is 20° (from 350 to 10 going through 360)
        # So midpoint should be 350 + 10 = 360 = 0
        expected_yaw = 0.0  # or equivalently 360.0
        
        # Allow for the result to be near 0 or 360
        assert abs(pose.yaw) < 1.0 or abs(pose.yaw - 360) < 1.0


class TestTrajectoryFromCSV:
    """Tests for loading trajectory from CSV."""
    
    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV trajectory content."""
        return """time,latitude,longitude,height,roll,pitch,yaw
0.0,45.0,-122.0,1000.0,0.0,0.0,0.0
1.0,45.001,-121.999,1010.0,1.0,-1.0,10.0
2.0,45.002,-121.998,1020.0,2.0,-2.0,20.0
"""
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_content):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            return f.name
    
    def test_from_csv(self, temp_csv_file):
        """Test loading from CSV."""
        interp = TrajectoryInterpolator.from_csv(temp_csv_file)
        
        assert len(interp.epochs) == 3
        assert interp.time_start == pytest.approx(0.0)
        assert interp.time_end == pytest.approx(2.0)
    
    def test_from_csv_interpolation(self, temp_csv_file):
        """Test interpolation after loading from CSV."""
        interp = TrajectoryInterpolator.from_csv(temp_csv_file)
        pose = interp.interpolate(0.5)
        
        assert pose is not None
        assert pose.height == pytest.approx(1005.0)


class TestImagePoseRetrieval:
    """Tests for getting poses for images based on timing."""
    
    @pytest.fixture
    def trajectory(self):
        """Create test trajectory."""
        epochs = [
            TrajectoryEpoch(time=0.0, pose=TrajectoryPose(45.0, -122.0, 1000.0, 0, 0, 0)),
            TrajectoryEpoch(time=1.0, pose=TrajectoryPose(45.001, -121.999, 1010.0, 1, -1, 10)),
            TrajectoryEpoch(time=2.0, pose=TrajectoryPose(45.002, -121.998, 1020.0, 2, -2, 20)),
        ]
        return TrajectoryInterpolator(epochs)
    
    def test_get_pose_at_image_time(self, trajectory):
        """Test getting pose for a single image."""
        timings = {1001: 0.5, 1002: 1.5}
        
        pose = trajectory.get_pose_at_image_time(1001, timings)
        assert pose is not None
        assert pose.height == pytest.approx(1005.0)
    
    def test_get_pose_missing_image(self, trajectory):
        """Test getting pose for missing image ID."""
        timings = {1001: 0.5}
        
        pose = trajectory.get_pose_at_image_time(9999, timings)
        assert pose is None
    
    def test_get_all_image_poses(self, trajectory):
        """Test getting poses for all images."""
        timings = {1001: 0.0, 1002: 1.0, 1003: 2.0}
        
        poses = trajectory.get_all_image_poses(timings)
        
        assert len(poses) == 3
        assert 1001 in poses
        assert 1002 in poses
        assert 1003 in poses
        assert poses[1001].height == pytest.approx(1000.0)
        assert poses[1002].height == pytest.approx(1010.0)
    
    def test_get_all_image_poses_with_out_of_range(self, trajectory):
        """Test that out-of-range times are handled gracefully."""
        timings = {1001: 0.5, 1002: 5.0}  # 5.0 is out of range
        
        poses = trajectory.get_all_image_poses(timings)
        
        assert len(poses) == 1
        assert 1001 in poses
        assert 1002 not in poses


class TestLoadTrajectoryInterpolator:
    """Tests for the convenience loader function."""
    
    @pytest.fixture
    def temp_csv_file(self):
        """Create temporary CSV trajectory."""
        content = """time,latitude,longitude,height,roll,pitch,yaw
0.0,45.0,-122.0,1000.0,0.0,0.0,0.0
1.0,45.001,-121.999,1010.0,1.0,-1.0,10.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_load_csv_auto(self, temp_csv_file):
        """Test auto-detection of CSV format."""
        interp = load_trajectory_interpolator(temp_csv_file, file_format='auto')
        assert len(interp.epochs) == 2
    
    def test_load_csv_explicit(self, temp_csv_file):
        """Test explicit CSV format."""
        interp = load_trajectory_interpolator(temp_csv_file, file_format='csv')
        assert len(interp.epochs) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
