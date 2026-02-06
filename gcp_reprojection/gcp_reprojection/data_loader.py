"""
Data loader module for reading GCP and trajectory data.

Supports:
    - BINGO format correspondence files
    - CSV format correspondence files  
    - CSV and SBET trajectory files
    - Image timing files for trajectory interpolation

BINGO Correspondence Format:
    gcp_id gcp_name
    image_id U V
    -99

CSV Correspondence Format:
    gcp_id, image_id, u, v

GCP Geocentric Coordinates:
    gcp_id, x, y, z (ECEF in meters)

Timing File:
    image_id, time
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .config import FilePaths, CoordinateConventions, CameraIntrinsics
from .transforms import TrajectoryPose
from .bingo_parser import BINGOParser, BINGOObservation, parse_timing_file
from .trajectory_interpolator import (
    TrajectoryInterpolator, 
    load_trajectory_interpolator,
)

logger = logging.getLogger(__name__)


@dataclass
class GCPMeasurement:
    """A single GCP measurement in an image."""
    gcp_id: str
    gcp_name: str
    image_id: int  # Changed to int to match BINGO format
    u: float  # Pixel coordinate (origin top-left)
    v: float  # Pixel coordinate (origin top-left)
    u_photo: float = 0.0  # Original photo-coordinate (for reference)
    v_photo: float = 0.0  # Original photo-coordinate (for reference)


@dataclass  
class GCPCoordinate:
    """3D geocentric (ECEF) coordinates of a GCP."""
    gcp_id: str
    x: float  # ECEF X in meters
    y: float  # ECEF Y in meters
    z: float  # ECEF Z in meters
    
    def as_array(self) -> np.ndarray:
        """Return coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])


class DataLoader:
    """
    Loads and manages GCP and trajectory data from various file formats.
    
    Supports:
        - BINGO format correspondence files
        - CSV format correspondence files
        - Trajectory interpolation based on image timing
        
    Workflow:
        1. Load GCP correspondence (BINGO or CSV)
        2. Load GCP 3D coordinates
        3. Load trajectory and timing
        4. Interpolate poses for each image
    """
    
    def __init__(
        self,
        file_paths: FilePaths,
        conventions: CoordinateConventions,
        camera: Optional[CameraIntrinsics] = None,
    ):
        """
        Initialize data loader.
        
        Args:
            file_paths: Paths to the data files
            conventions: Coordinate system conventions
            camera: Camera intrinsics (needed for photo-to-pixel conversion)
        """
        self.file_paths = file_paths
        self.conventions = conventions
        self.camera = camera
        
        # Data storage
        self.gcp_measurements: List[GCPMeasurement] = []
        self.gcp_coordinates: Dict[str, GCPCoordinate] = {}
        self.image_timings: Dict[int, float] = {}
        self.trajectory_interpolator: Optional[TrajectoryInterpolator] = None
        self.interpolated_poses: Dict[int, TrajectoryPose] = {}
        
        # Indices for fast lookup
        self._measurements_by_image: Dict[int, List[GCPMeasurement]] = {}
        self._measurements_by_gcp: Dict[str, List[GCPMeasurement]] = {}
        
        # BINGO parser
        self.bingo_parser = BINGOParser(v_axis_up=conventions.v_axis_up)
    
    def load_all(self) -> None:
        """Load all data files and interpolate poses."""
        self.load_gcp_image_coords()
        self.load_gcp_geocentric_coords()
        self.load_trajectory_and_timing()
        self._build_indices()
        self._validate_data()
    
    def load_gcp_image_coords(self) -> None:
        """
        Load GCP image coordinate measurements.
        
        Supports BINGO and CSV formats based on configuration.
        """
        path = Path(self.file_paths.gcp_image_coords)
        if not path.exists():
            raise FileNotFoundError(f"GCP image coordinates file not found: {path}")
        
        if self.conventions.correspondence_format.lower() == 'bingo':
            self._load_bingo_correspondences(path)
        else:
            self._load_csv_correspondences(path)
        
        logger.info(f"Loaded {len(self.gcp_measurements)} GCP measurements")
    
    def _load_bingo_correspondences(self, path: Path) -> None:
        """Load correspondences from BINGO format file."""
        observations = self.bingo_parser.parse_file(str(path))
        
        self.gcp_measurements = []
        
        for obs in observations:
            # Convert photo-coordinates to pixel coordinates
            if self.camera and self.camera.image_width > 0:
                pixel_u, pixel_v = self.bingo_parser.to_pixel_coordinates(
                    obs.u, obs.v,
                    self.camera.image_width,
                    self.camera.image_height,
                )
            else:
                # If no camera info, assume already in pixels or will be converted later
                pixel_u, pixel_v = obs.u, obs.v
                logger.warning(
                    "No camera dimensions provided; using raw BINGO coordinates. "
                    "Set camera.image_width and camera.image_height for proper conversion."
                )
            
            measurement = GCPMeasurement(
                gcp_id=str(obs.gcp_id),
                gcp_name=obs.gcp_name,
                image_id=obs.image_id,
                u=pixel_u,
                v=pixel_v,
                u_photo=obs.u,
                v_photo=obs.v,
            )
            self.gcp_measurements.append(measurement)
    
    def _load_csv_correspondences(self, path: Path) -> None:
        """Load correspondences from CSV format file."""
        self.gcp_measurements = []
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            required = {'gcp_id', 'image_id', 'u', 'v'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Missing required columns in {path}. "
                    f"Required: {required}, Found: {reader.fieldnames}"
                )
            
            for row in reader:
                measurement = GCPMeasurement(
                    gcp_id=row['gcp_id'].strip(),
                    gcp_name=row.get('gcp_name', row['gcp_id']).strip(),
                    image_id=int(row['image_id']),
                    u=float(row['u']),
                    v=float(row['v']),
                )
                self.gcp_measurements.append(measurement)
    
    def load_gcp_geocentric_coords(self) -> None:
        """
        Load GCP 3D geocentric (ECEF) coordinates.
        
        Expected columns: gcp_id, x, y, z
        """
        path = Path(self.file_paths.gcp_geocentric_coords)
        if not path.exists():
            raise FileNotFoundError(f"GCP geocentric coordinates file not found: {path}")
        
        self.gcp_coordinates = {}
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            required = {'gcp_id', 'x', 'y', 'z'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Missing required columns in {path}. "
                    f"Required: {required}, Found: {reader.fieldnames}"
                )
            
            for row in reader:
                gcp_id = row['gcp_id'].strip()
                coord = GCPCoordinate(
                    gcp_id=gcp_id,
                    x=float(row['x']),
                    y=float(row['y']),
                    z=float(row['z']),
                )
                self.gcp_coordinates[gcp_id] = coord
        
        logger.info(f"Loaded {len(self.gcp_coordinates)} GCP coordinates")
    
    def load_trajectory_and_timing(self) -> None:
        """
        Load trajectory data and image timing, then interpolate poses.
        
        If timing file is provided, interpolates poses for each image.
        If no timing file, expects trajectory to have image_id column.
        """
        traj_path = Path(self.file_paths.trajectory)
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        
        # Load timing file if provided
        if self.file_paths.timing:
            timing_path = Path(self.file_paths.timing)
            if not timing_path.exists():
                raise FileNotFoundError(f"Timing file not found: {timing_path}")
            
            self.image_timings = parse_timing_file(str(timing_path))
            logger.info(f"Loaded {len(self.image_timings)} image timings")
            
            # Load trajectory interpolator
            self.trajectory_interpolator = load_trajectory_interpolator(
                str(traj_path),
                file_format=self.conventions.trajectory_format,
            )
            
            # Interpolate poses for all images
            self.interpolated_poses = self.trajectory_interpolator.get_all_image_poses(
                self.image_timings
            )
            
            logger.info(f"Interpolated poses for {len(self.interpolated_poses)} images")
        else:
            # No timing file - load trajectory with direct image_id mapping
            self._load_trajectory_direct(traj_path)
    
    def _load_trajectory_direct(self, path: Path) -> None:
        """
        Load trajectory with direct image_id mapping (no interpolation).
        
        Expected columns: image_id, latitude, longitude, height, roll, pitch, yaw
        """
        self.interpolated_poses = {}
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            required = {'image_id', 'latitude', 'longitude', 'height', 'roll', 'pitch', 'yaw'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Missing required columns in {path}. "
                    f"Required: {required}, Found: {reader.fieldnames}"
                )
            
            for row in reader:
                image_id = int(row['image_id'])
                pose = TrajectoryPose(
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    height=float(row['height']),
                    roll=float(row['roll']),
                    pitch=float(row['pitch']),
                    yaw=float(row['yaw']),
                )
                self.interpolated_poses[image_id] = pose
        
        logger.info(f"Loaded {len(self.interpolated_poses)} trajectory records directly")
    
    def _build_indices(self) -> None:
        """Build lookup indices for efficient querying."""
        self._measurements_by_image = {}
        self._measurements_by_gcp = {}
        
        for m in self.gcp_measurements:
            # Index by image
            if m.image_id not in self._measurements_by_image:
                self._measurements_by_image[m.image_id] = []
            self._measurements_by_image[m.image_id].append(m)
            
            # Index by GCP
            if m.gcp_id not in self._measurements_by_gcp:
                self._measurements_by_gcp[m.gcp_id] = []
            self._measurements_by_gcp[m.gcp_id].append(m)
    
    def _validate_data(self) -> None:
        """Validate data consistency across files."""
        warnings = []
        
        # Check that all measured GCPs have 3D coordinates
        measured_gcp_ids = set(m.gcp_id for m in self.gcp_measurements)
        coord_gcp_ids = set(self.gcp_coordinates.keys())
        
        missing_coords = measured_gcp_ids - coord_gcp_ids
        if missing_coords:
            warnings.append(
                f"GCPs with measurements but no 3D coordinates: {missing_coords}"
            )
        
        unused_coords = coord_gcp_ids - measured_gcp_ids
        if unused_coords:
            warnings.append(
                f"GCPs with 3D coordinates but no measurements: {unused_coords}"
            )
        
        # Check that all images with measurements have poses
        measured_image_ids = set(m.image_id for m in self.gcp_measurements)
        pose_image_ids = set(self.interpolated_poses.keys())
        
        missing_poses = measured_image_ids - pose_image_ids
        if missing_poses:
            warnings.append(
                f"Images with measurements but no poses: {missing_poses}"
            )
        
        for warning in warnings:
            logger.warning(warning)
    
    def get_measurements_for_image(self, image_id: int) -> List[GCPMeasurement]:
        """Get all GCP measurements for a specific image."""
        return self._measurements_by_image.get(image_id, [])
    
    def get_measurements_for_gcp(self, gcp_id: str) -> List[GCPMeasurement]:
        """Get all measurements of a specific GCP across images."""
        return self._measurements_by_gcp.get(gcp_id, [])
    
    def get_gcp_coordinate(self, gcp_id: str) -> Optional[GCPCoordinate]:
        """Get 3D coordinates for a GCP."""
        return self.gcp_coordinates.get(gcp_id)
    
    def get_trajectory_pose(self, image_id: int) -> Optional[TrajectoryPose]:
        """Get (interpolated) trajectory pose for an image."""
        return self.interpolated_poses.get(image_id)
    
    def get_all_image_ids(self) -> List[int]:
        """Get list of all image IDs with measurements."""
        return list(self._measurements_by_image.keys())
    
    def get_all_gcp_ids(self) -> List[str]:
        """Get list of all GCP IDs with measurements."""
        return list(self._measurements_by_gcp.keys())
    
    def get_statistics(self) -> Dict[str, int]:
        """Get summary statistics about loaded data."""
        return {
            'num_measurements': len(self.gcp_measurements),
            'num_gcps': len(self.gcp_coordinates),
            'num_images': len(self._measurements_by_image),
            'num_poses': len(self.interpolated_poses),
            'num_timings': len(self.image_timings),
        }
