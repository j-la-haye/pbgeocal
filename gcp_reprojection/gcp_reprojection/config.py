"""
Configuration module for GCP reprojection validation.

Handles loading and validation of configuration from YAML files.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length in x (pixels)
    fy: float  # Focal length in y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    k1: float = 0.0  # Radial distortion coefficient
    k2: float = 0.0  # Radial distortion coefficient
    k3: float = 0.0  # Radial distortion coefficient
    p1: float = 0.0  # Tangential distortion coefficient
    p2: float = 0.0  # Tangential distortion coefficient
    image_width: int = 0  # Image width in pixels
    image_height: int = 0  # Image height in pixels


@dataclass
class BoresightAngles:
    """
    Boresight misalignment angles between IMU body frame and camera frame.
    These are small correction angles applied after the nominal body-to-camera rotation.
    """
    roll: float = 0.0   # Roll offset in degrees
    pitch: float = 0.0  # Pitch offset in degrees
    yaw: float = 0.0    # Yaw offset in degrees


@dataclass
class LeverArm:
    """
    Lever arm offset from IMU center to camera projection center.
    Expressed in body frame coordinates (Forward-Right-Down).
    """
    x: float = 0.0  # Forward offset in meters
    y: float = 0.0  # Right offset in meters
    z: float = 0.0  # Down offset in meters


@dataclass
class FilePaths:
    """Paths to input data files."""
    gcp_image_coords: str  # GCP ID and (u,v) image coordinates (CSV or BINGO format)
    gcp_geocentric_coords: str  # GCP ID and 3D geocentric (ECEF) coordinates
    trajectory: str  # Trajectory file (CSV or binary SBET)
    timing: Optional[str] = None  # Image timing file (required for trajectory interpolation)


@dataclass
class CoordinateConventions:
    """
    Coordinate system conventions for input data.
    
    BINGO photo-coordinates typically have origin at image center.
    This configuration specifies how to convert to pixel coordinates.
    """
    # BINGO V-axis convention: True if V is positive upward (photogrammetric)
    v_axis_up: bool = True
    
    # Input file format for correspondence
    correspondence_format: str = 'bingo'  # 'bingo' or 'csv'
    
    # Trajectory file format
    trajectory_format: str = 'csv'  # 'csv' or 'sbet'


@dataclass
class Config:
    """
    Main configuration class for GCP reprojection validation.
    
    Attributes:
        epsg_code: EPSG code defining the geocentric coordinate system (e.g., 4978 for WGS84 ECEF)
        camera: Camera intrinsic parameters
        boresight: Boresight misalignment angles
        lever_arm: Lever arm offset from IMU to camera
        files: Paths to input data files
        conventions: Coordinate system conventions
        validation_threshold: Maximum acceptable reprojection error in pixels
    """
    epsg_code: int
    camera: CameraIntrinsics
    files: FilePaths
    boresight: BoresightAngles = field(default_factory=BoresightAngles)
    lever_arm: LeverArm = field(default_factory=LeverArm)
    conventions: CoordinateConventions = field(default_factory=CoordinateConventions)
    validation_threshold: float = 2.0  # pixels
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config object with loaded parameters
            
        Example YAML structure:
            epsg_code: 4978
            camera:
              fx: 5000.0
              fy: 5000.0
              cx: 2000.0
              cy: 1500.0
              k1: -0.1
              k2: 0.01
              image_width: 4000
              image_height: 3000
            boresight:
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
            lever_arm:
              x: 0.0
              y: 0.0
              z: 0.0
            conventions:
              v_axis_up: true
              correspondence_format: bingo
              trajectory_format: csv
            files:
              gcp_image_coords: "correspondences.bingo"
              gcp_geocentric_coords: "gcp_ecef.csv"
              trajectory: "trajectory.csv"
              timing: "image_timing.csv"
            validation_threshold: 2.0
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        logger.info(f"Loading configuration from {config_path}")
        
        # Parse camera intrinsics
        cam_data = data.get('camera', {})
        camera = CameraIntrinsics(
            fx=cam_data['fx'],
            fy=cam_data['fy'],
            cx=cam_data['cx'],
            cy=cam_data['cy'],
            k1=cam_data.get('k1', 0.0),
            k2=cam_data.get('k2', 0.0),
            k3=cam_data.get('k3', 0.0),
            p1=cam_data.get('p1', 0.0),
            p2=cam_data.get('p2', 0.0),
            image_width=cam_data.get('image_width', 0),
            image_height=cam_data.get('image_height', 0),
        )
        
        # Parse boresight angles (optional)
        bore_data = data.get('boresight', {})
        boresight = BoresightAngles(
            roll=bore_data.get('roll', 0.0),
            pitch=bore_data.get('pitch', 0.0),
            yaw=bore_data.get('yaw', 0.0),
        )
        
        # Parse lever arm (optional)
        lever_data = data.get('lever_arm', {})
        lever_arm = LeverArm(
            x=lever_data.get('x', 0.0),
            y=lever_data.get('y', 0.0),
            z=lever_data.get('z', 0.0),
        )
        
        # Parse conventions (optional)
        conv_data = data.get('conventions', {})
        conventions = CoordinateConventions(
            v_axis_up=conv_data.get('v_axis_up', True),
            correspondence_format=conv_data.get('correspondence_format', 'bingo'),
            trajectory_format=conv_data.get('trajectory_format', 'csv'),
        )
        
        # Parse file paths
        files_data = data.get('files', {})
        
        # Resolve paths relative to config file location
        config_dir = path.parent
        
        timing_path = files_data.get('timing')
        if timing_path:
            timing_path = str(config_dir / timing_path)
        
        files = FilePaths(
            gcp_image_coords=str(config_dir / files_data['gcp_image_coords']),
            gcp_geocentric_coords=str(config_dir / files_data['gcp_geocentric_coords']),
            trajectory=str(config_dir / files_data['trajectory']),
            timing=timing_path,
        )
        
        return cls(
            epsg_code=data['epsg_code'],
            camera=camera,
            boresight=boresight,
            lever_arm=lever_arm,
            conventions=conventions,
            files=files,
            validation_threshold=data.get('validation_threshold', 2.0),
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            'epsg_code': self.epsg_code,
            'camera': {
                'fx': self.camera.fx,
                'fy': self.camera.fy,
                'cx': self.camera.cx,
                'cy': self.camera.cy,
                'k1': self.camera.k1,
                'k2': self.camera.k2,
                'k3': self.camera.k3,
                'p1': self.camera.p1,
                'p2': self.camera.p2,
                'image_width': self.camera.image_width,
                'image_height': self.camera.image_height,
            },
            'boresight': {
                'roll': self.boresight.roll,
                'pitch': self.boresight.pitch,
                'yaw': self.boresight.yaw,
            },
            'lever_arm': {
                'x': self.lever_arm.x,
                'y': self.lever_arm.y,
                'z': self.lever_arm.z,
            },
            'conventions': {
                'v_axis_up': self.conventions.v_axis_up,
                'correspondence_format': self.conventions.correspondence_format,
                'trajectory_format': self.conventions.trajectory_format,
            },
            'files': {
                'gcp_image_coords': self.files.gcp_image_coords,
                'gcp_geocentric_coords': self.files.gcp_geocentric_coords,
                'trajectory': self.files.trajectory,
                'timing': self.files.timing,
            },
            'validation_threshold': self.validation_threshold,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {config_path}")
