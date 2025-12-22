"""
GCP Reprojection Validation Package

A Python package to validate the reprojection of 3D Ground Control Points (GCPs)
from geocentric (ECEF) coordinates to image coordinates using camera intrinsics
and trajectory information.

Coordinate System Chain:
    GCP (ECEF) → Local NED → Body Frame → Camera Frame → Image (u,v)

Conventions:
    - Trajectory: WGS84 position with ellipsoidal height, orientation in local NED
    - SBET notation: Forward-Right-Down body frame
    - Camera mount: X-right, Y-back, Z-down

Supported Formats:
    - BINGO format correspondence files
    - CSV format correspondence files
    - SBET and CSV trajectory files
    - Image timing files for trajectory interpolation
"""

from .config import Config, CameraIntrinsics, CoordinateConventions
from .transforms import CoordinateTransformer, TrajectoryPose
from .camera import CameraModel
from .validator import GCPValidator, ValidationReport, run_validation
from .data_loader import DataLoader, GCPMeasurement, GCPCoordinate
from .bingo_parser import BINGOParser, BINGOObservation, parse_bingo_file, parse_timing_file
from .trajectory_interpolator import TrajectoryInterpolator, load_trajectory_interpolator

__version__ = "1.1.0"
__all__ = [
    "Config",
    "CameraIntrinsics",
    "CoordinateConventions",
    "CoordinateTransformer",
    "TrajectoryPose",
    "CameraModel",
    "GCPValidator",
    "ValidationReport",
    "run_validation",
    "DataLoader",
    "GCPMeasurement",
    "GCPCoordinate",
    "BINGOParser",
    "BINGOObservation",
    "parse_bingo_file",
    "parse_timing_file",
    "TrajectoryInterpolator",
    "load_trajectory_interpolator",
]
