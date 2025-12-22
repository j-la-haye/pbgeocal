"""
Trajectory interpolation module.

Provides interpolation of trajectory poses based on time.

Trajectory data is typically provided at regular intervals (e.g., 200 Hz for SBET).
Image capture times may not coincide with trajectory epochs, so interpolation
is necessary to obtain the camera pose at the exact image capture time.

Interpolation Methods:
    - Linear: Simple linear interpolation (position and angles separately)
    - Slerp: Spherical linear interpolation for rotations (more accurate for large angle changes)
    
Trajectory File Format (SBET-style):
    time, latitude, longitude, height, roll, pitch, yaw
    
    - time: GPS time or other monotonic time reference (seconds)
    - latitude, longitude: degrees
    - height: ellipsoidal height (meters)
    - roll, pitch, yaw: orientation in degrees
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import csv
import logging

from .transforms import TrajectoryPose

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEpoch:
    """A single trajectory epoch with time and pose."""
    time: float
    pose: TrajectoryPose


class TrajectoryInterpolator:
    """
    Interpolates trajectory poses based on time.
    
    Supports:
        - Linear interpolation for position
        - Linear interpolation for orientation (suitable for small angle changes)
        - Optional SLERP for orientation (better for large angle changes)
        
    The trajectory must be sorted by time for efficient interpolation.
    """
    
    def __init__(
        self,
        epochs: List[TrajectoryEpoch],
        use_slerp: bool = False,
        angle_wrap_threshold: float = 180.0,
    ):
        """
        Initialize interpolator with trajectory epochs.
        
        Args:
            epochs: List of trajectory epochs (will be sorted by time)
            use_slerp: Whether to use SLERP for angle interpolation
            angle_wrap_threshold: Threshold for detecting angle wrapping (degrees)
        """
        # Sort epochs by time
        self.epochs = sorted(epochs, key=lambda e: e.time)
        self.use_slerp = use_slerp
        self.angle_wrap_threshold = angle_wrap_threshold
        
        if not self.epochs:
            raise ValueError("No trajectory epochs provided")
        
        # Extract times for fast lookup
        self.times = np.array([e.time for e in self.epochs])
        
        self.time_start = self.times[0]
        self.time_end = self.times[-1]
        
        logger.info(
            f"Trajectory interpolator initialized with {len(self.epochs)} epochs, "
            f"time range: {self.time_start:.3f} to {self.time_end:.3f}"
        )
    
    def interpolate(self, time: float) -> Optional[TrajectoryPose]:
        """
        Interpolate trajectory pose at given time.
        
        Args:
            time: Query time
            
        Returns:
            Interpolated TrajectoryPose, or None if time is outside range
        """
        # Check bounds
        if time < self.time_start or time > self.time_end:
            logger.warning(
                f"Time {time:.3f} outside trajectory range "
                f"[{self.time_start:.3f}, {self.time_end:.3f}]"
            )
            return None
        
        # Find bracketing epochs using binary search
        idx = np.searchsorted(self.times, time)
        
        # Handle edge cases
        if idx == 0:
            return self.epochs[0].pose
        if idx >= len(self.epochs):
            return self.epochs[-1].pose
        
        # Get bracketing epochs
        epoch_before = self.epochs[idx - 1]
        epoch_after = self.epochs[idx]
        
        # Compute interpolation factor
        dt = epoch_after.time - epoch_before.time
        if dt == 0:
            return epoch_before.pose
        
        t = (time - epoch_before.time) / dt
        
        # Interpolate pose
        return self._interpolate_poses(epoch_before.pose, epoch_after.pose, t)
    
    def _interpolate_poses(
        self,
        pose1: TrajectoryPose,
        pose2: TrajectoryPose,
        t: float,
    ) -> TrajectoryPose:
        """
        Interpolate between two poses.
        
        Args:
            pose1: Starting pose
            pose2: Ending pose
            t: Interpolation factor (0 = pose1, 1 = pose2)
            
        Returns:
            Interpolated pose
        """
        # Linear interpolation for position
        lat = pose1.latitude + t * (pose2.latitude - pose1.latitude)
        lon = pose1.longitude + t * (pose2.longitude - pose1.longitude)
        height = pose1.height + t * (pose2.height - pose1.height)
        
        # Interpolate angles (handling wrap-around)
        roll = self._interpolate_angle(pose1.roll, pose2.roll, t)
        pitch = self._interpolate_angle(pose1.pitch, pose2.pitch, t)
        yaw = self._interpolate_angle(pose1.yaw, pose2.yaw, t)
        
        return TrajectoryPose(
            latitude=lat,
            longitude=lon,
            height=height,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
    
    def _interpolate_angle(
        self,
        angle1: float,
        angle2: float,
        t: float,
    ) -> float:
        """
        Interpolate between two angles, handling wrap-around.
        
        Args:
            angle1: Starting angle in degrees
            angle2: Ending angle in degrees
            t: Interpolation factor
            
        Returns:
            Interpolated angle in degrees
        """
        # Compute the difference
        diff = angle2 - angle1
        
        # Handle wrap-around (e.g., 350° to 10°)
        if diff > self.angle_wrap_threshold:
            diff -= 360.0
        elif diff < -self.angle_wrap_threshold:
            diff += 360.0
        
        result = angle1 + t * diff
        
        # Normalize to [-180, 180) or [0, 360) as needed
        # Keeping in original range for consistency
        return result
    
    def get_pose_at_image_time(
        self,
        image_id: int,
        image_timings: Dict[int, float],
    ) -> Optional[TrajectoryPose]:
        """
        Get interpolated pose for an image given its timing.
        
        Args:
            image_id: Image identifier
            image_timings: Dictionary mapping image_id to capture time
            
        Returns:
            Interpolated pose or None if image_id not found or time out of range
        """
        if image_id not in image_timings:
            logger.warning(f"No timing found for image {image_id}")
            return None
        
        time = image_timings[image_id]
        return self.interpolate(time)
    
    def get_all_image_poses(
        self,
        image_timings: Dict[int, float],
    ) -> Dict[int, TrajectoryPose]:
        """
        Get interpolated poses for all images.
        
        Args:
            image_timings: Dictionary mapping image_id to capture time
            
        Returns:
            Dictionary mapping image_id to interpolated pose
        """
        poses = {}
        
        for image_id, time in image_timings.items():
            pose = self.interpolate(time)
            if pose is not None:
                poses[image_id] = pose
            else:
                logger.warning(f"Could not interpolate pose for image {image_id} at time {time}")
        
        logger.info(f"Interpolated poses for {len(poses)}/{len(image_timings)} images")
        return poses
    
    @classmethod
    def from_csv(
        cls,
        filepath: str,
        time_col: str = 'time',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        height_col: str = 'height',
        roll_col: str = 'roll',
        pitch_col: str = 'pitch',
        yaw_col: str = 'yaw',
        **kwargs,
    ) -> 'TrajectoryInterpolator':
        """
        Load trajectory from CSV file.
        
        Args:
            filepath: Path to CSV file
            time_col: Column name for time
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            height_col: Column name for height
            roll_col: Column name for roll
            pitch_col: Column name for pitch
            yaw_col: Column name for yaw (heading)
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            TrajectoryInterpolator instance
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")
        
        epochs = []
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                epoch = TrajectoryEpoch(
                    time=float(row[time_col]),
                    pose=TrajectoryPose(
                        latitude=float(row[lat_col]),
                        longitude=float(row[lon_col]),
                        height=float(row[height_col]),
                        roll=float(row[roll_col]),
                        pitch=float(row[pitch_col]),
                        yaw=float(row[yaw_col]),
                    ),
                )
                epochs.append(epoch)
        
        logger.info(f"Loaded {len(epochs)} trajectory epochs from {filepath}")
        return cls(epochs, **kwargs)
    
    @classmethod
    def from_sbet_csv(
        cls,
        filepath: str,
        **kwargs,
    ) -> 'TrajectoryInterpolator':
        """
        Load trajectory from SBET-style CSV.
        
        Expected columns: time, latitude, longitude, height, roll, pitch, heading
        
        Args:
            filepath: Path to SBET CSV file
            **kwargs: Additional arguments
            
        Returns:
            TrajectoryInterpolator instance
        """
        return cls.from_csv(
            filepath,
            time_col='time',
            lat_col='latitude',
            lon_col='longitude', 
            height_col='height',
            roll_col='roll',
            pitch_col='pitch',
            yaw_col='heading',
            **kwargs,
        )


class SBETReader:
    """
    Reader for binary SBET (Smoothed Best Estimate of Trajectory) files.
    
    SBET binary format (per epoch):
        - time: double (8 bytes) - GPS seconds of week
        - latitude: double (8 bytes) - radians
        - longitude: double (8 bytes) - radians
        - altitude: double (8 bytes) - meters
        - x_velocity: double (8 bytes) - m/s
        - y_velocity: double (8 bytes) - m/s
        - z_velocity: double (8 bytes) - m/s
        - roll: double (8 bytes) - radians
        - pitch: double (8 bytes) - radians
        - heading: double (8 bytes) - radians
        - wander_angle: double (8 bytes) - radians
        - x_acceleration: double (8 bytes) - m/s²
        - y_acceleration: double (8 bytes) - m/s²
        - z_acceleration: double (8 bytes) - m/s²
        - x_angular_rate: double (8 bytes) - rad/s
        - y_angular_rate: double (8 bytes) - rad/s
        - z_angular_rate: double (8 bytes) - rad/s
        
    Total: 17 doubles = 136 bytes per epoch
    """
    
    RECORD_SIZE = 136  # bytes
    NUM_FIELDS = 17
    
    def __init__(self, filepath: str):
        """
        Initialize SBET reader.
        
        Args:
            filepath: Path to binary SBET file
        """
        self.filepath = filepath
        self.path = Path(filepath)
        
        if not self.path.exists():
            raise FileNotFoundError(f"SBET file not found: {filepath}")
    
    def read_epochs(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        downsample: int = 1,
    ) -> List[TrajectoryEpoch]:
        """
        Read trajectory epochs from SBET file.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            downsample: Read every Nth epoch (1 = all, 10 = every 10th)
            
        Returns:
            List of TrajectoryEpoch objects
        """
        epochs = []
        
        with open(self.filepath, 'rb') as f:
            record_count = 0
            
            while True:
                data = f.read(self.RECORD_SIZE)
                if len(data) < self.RECORD_SIZE:
                    break
                
                record_count += 1
                
                # Downsample
                if record_count % downsample != 0:
                    continue
                
                # Unpack binary data
                values = np.frombuffer(data, dtype=np.float64)
                
                time = values[0]
                
                # Apply time filter
                if start_time is not None and time < start_time:
                    continue
                if end_time is not None and time > end_time:
                    break
                
                # Convert radians to degrees
                lat_deg = np.rad2deg(values[1])
                lon_deg = np.rad2deg(values[2])
                height = values[3]
                roll_deg = np.rad2deg(values[7])
                pitch_deg = np.rad2deg(values[8])
                heading_deg = np.rad2deg(values[9])
                
                epoch = TrajectoryEpoch(
                    time=time,
                    pose=TrajectoryPose(
                        latitude=lat_deg,
                        longitude=lon_deg,
                        height=height,
                        roll=roll_deg,
                        pitch=pitch_deg,
                        yaw=heading_deg,
                    ),
                )
                epochs.append(epoch)
        
        logger.info(f"Read {len(epochs)} epochs from SBET file (downsampled {downsample}x)")
        return epochs
    
    def to_interpolator(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        downsample: int = 1,
        **kwargs,
    ) -> TrajectoryInterpolator:
        """
        Create interpolator from SBET file.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            downsample: Read every Nth epoch
            **kwargs: Additional arguments for TrajectoryInterpolator
            
        Returns:
            TrajectoryInterpolator instance
        """
        epochs = self.read_epochs(start_time, end_time, downsample)
        return TrajectoryInterpolator(epochs, **kwargs)


def load_trajectory_interpolator(
    filepath: str,
    file_format: str = 'auto',
    **kwargs,
) -> TrajectoryInterpolator:
    """
    Load trajectory interpolator from file.
    
    Args:
        filepath: Path to trajectory file
        file_format: 'csv', 'sbet', or 'auto' (detect from extension)
        **kwargs: Additional arguments
        
    Returns:
        TrajectoryInterpolator instance
    """
    path = Path(filepath)
    
    if file_format == 'auto':
        ext = path.suffix.lower()
        if ext in ['.sbet', '.out']:
            file_format = 'sbet'
        else:
            file_format = 'csv'
    
    if file_format == 'sbet':
        reader = SBETReader(filepath)
        return reader.to_interpolator(**kwargs)
    else:
        return TrajectoryInterpolator.from_csv(filepath, **kwargs)
