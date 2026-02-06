"""
GCP Reprojection Validator module.

This is the main module that orchestrates the validation workflow:
    1. Load configuration and data
    2. For each GCP measurement:
        a. Get GCP 3D coordinates (ECEF)
        b. Get interpolated trajectory pose for the image
        c. Transform GCP to camera frame
        d. Project to image coordinates
        e. Compare with measured coordinates
    3. Generate validation report

The validator computes reprojection errors and provides statistics
to assess the quality of the camera calibration and trajectory.

Supports:
    - BINGO format correspondence files
    - Trajectory interpolation from SBET or CSV files
    - Configurable coordinate conventions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

from .config import Config
from .transforms import CoordinateTransformer
from .camera import CameraModel
from .data_loader import DataLoader, GCPMeasurement

logger = logging.getLogger(__name__)


@dataclass
class ReprojectionResult:
    """Result of reprojecting a single GCP measurement."""
    gcp_id: str
    gcp_name: str
    image_id: int
    measured_u: float
    measured_v: float
    projected_u: float
    projected_v: float
    error: float  # Euclidean distance in pixels
    valid: bool  # True if projection was successful
    
    # Additional debug info
    point_camera: Optional[np.ndarray] = None  # Point in camera frame
    distance_m: Optional[float] = None  # Distance from camera in meters
    u_photo: Optional[float] = None  # Original photo-coordinate U
    v_photo: Optional[float] = None  # Original photo-coordinate V


@dataclass
class ValidationReport:
    """Summary report of the validation process."""
    # Overall statistics
    total_measurements: int = 0
    valid_projections: int = 0
    invalid_projections: int = 0
    
    # Error statistics (pixels)
    mean_error: float = 0.0
    std_error: float = 0.0
    min_error: float = 0.0
    max_error: float = 0.0
    median_error: float = 0.0
    rmse: float = 0.0
    
    # Thresholded results
    threshold: float = 2.0
    points_within_threshold: int = 0
    points_outside_threshold: int = 0
    pass_rate: float = 0.0
    
    # Per-GCP statistics
    gcp_errors: Dict[str, Dict] = field(default_factory=dict)
    
    # Per-image statistics
    image_errors: Dict[str, Dict] = field(default_factory=dict)
    
    # Detailed results
    results: List[ReprojectionResult] = field(default_factory=list)


class GCPValidator:
    """
    Main class for validating GCP reprojection.
    
    Workflow:
        1. Initialize with configuration
        2. Load data (supports BINGO format and trajectory interpolation)
        3. Run validation
        4. Generate report
        
    Example usage:
        config = Config.from_yaml("config.yaml")
        validator = GCPValidator(config)
        validator.load_data()
        report = validator.validate()
        validator.save_report(report, "results.json")
    """
    
    def __init__(self, config: Config):
        """
        Initialize the validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize transformer with boresight and lever arm
        self.transformer = CoordinateTransformer(
            boresight_roll=config.boresight.roll,
            boresight_pitch=config.boresight.pitch,
            boresight_yaw=config.boresight.yaw,
            lever_arm_x=config.lever_arm.x,
            lever_arm_y=config.lever_arm.y,
            lever_arm_z=config.lever_arm.z,
        )
        
        # Initialize camera model
        self.camera = CameraModel(config.camera)
        
        # Data loader with conventions and camera for coordinate conversion
        self.data_loader = DataLoader(
            file_paths=config.files,
            conventions=config.conventions,
            camera=config.camera,
        )
        
        logger.info(f"Validator initialized with EPSG:{config.epsg_code}")
    
    def load_data(self) -> None:
        """Load all input data files (BINGO correspondences, trajectory, etc.)."""
        logger.info("Loading data files...")
        self.data_loader.load_all()
        
        stats = self.data_loader.get_statistics()
        logger.info(f"Loaded {stats['num_measurements']} measurements")
        logger.info(f"Loaded {stats['num_gcps']} GCPs")
        logger.info(f"Loaded {stats['num_images']} images")
        logger.info(f"Loaded {stats['num_poses']} interpolated poses")
    
    def validate_single(
        self,
        measurement: GCPMeasurement,
    ) -> ReprojectionResult:
        """
        Validate a single GCP measurement.
        
        Args:
            measurement: GCP measurement to validate
            
        Returns:
            ReprojectionResult with comparison data
        """
        # Get GCP 3D coordinates
        gcp_coord = self.data_loader.get_gcp_coordinate(measurement.gcp_id)
        if gcp_coord is None:
            logger.warning(f"No coordinates for GCP {measurement.gcp_id}")
            return ReprojectionResult(
                gcp_id=measurement.gcp_id,
                gcp_name=measurement.gcp_name,
                image_id=measurement.image_id,
                measured_u=measurement.u,
                measured_v=measurement.v,
                projected_u=0.0,
                projected_v=0.0,
                error=float('inf'),
                valid=False,
                u_photo=measurement.u_photo,
                v_photo=measurement.v_photo,
            )
        
        # Get trajectory pose for this image (interpolated if timing provided)
        pose = self.data_loader.get_trajectory_pose(measurement.image_id)
        if pose is None:
            logger.warning(f"No trajectory for image {measurement.image_id}")
            return ReprojectionResult(
                gcp_id=measurement.gcp_id,
                gcp_name=measurement.gcp_name,
                image_id=measurement.image_id,
                measured_u=measurement.u,
                measured_v=measurement.v,
                projected_u=0.0,
                projected_v=0.0,
                error=float('inf'),
                valid=False,
                u_photo=measurement.u_photo,
                v_photo=measurement.v_photo,
            )
        
        # Transform GCP from ECEF to camera frame
        gcp_ecef = gcp_coord.as_array()
        point_camera = self.transformer.ecef_to_camera_frame(gcp_ecef, pose)
        
        # Compute distance from camera
        distance = np.linalg.norm(point_camera)
        
        # Project to image coordinates
        projected_u, projected_v, valid = self.camera.project_point(point_camera)
        
        if not valid:
            return ReprojectionResult(
                gcp_id=measurement.gcp_id,
                gcp_name=measurement.gcp_name,
                image_id=measurement.image_id,
                measured_u=measurement.u,
                measured_v=measurement.v,
                projected_u=projected_u,
                projected_v=projected_v,
                error=float('inf'),
                valid=False,
                point_camera=point_camera,
                distance_m=distance,
                u_photo=measurement.u_photo,
                v_photo=measurement.v_photo,
            )
        
        # Compute reprojection error
        error = np.sqrt(
            (projected_u - measurement.u) ** 2 +
            (projected_v - measurement.v) ** 2
        )
        
        return ReprojectionResult(
            gcp_id=measurement.gcp_id,
            gcp_name=measurement.gcp_name,
            image_id=measurement.image_id,
            measured_u=measurement.u,
            measured_v=measurement.v,
            projected_u=projected_u,
            projected_v=projected_v,
            error=error,
            valid=True,
            point_camera=point_camera,
            distance_m=distance,
            u_photo=measurement.u_photo,
            v_photo=measurement.v_photo,
        )
    
    def validate(self) -> ValidationReport:
        """
        Run validation on all GCP measurements.
        
        Returns:
            ValidationReport with statistics and detailed results
        """
        logger.info("Starting validation...")
        
        report = ValidationReport(threshold=self.config.validation_threshold)
        results: List[ReprojectionResult] = []
        
        # Process all measurements
        for measurement in self.data_loader.gcp_measurements:
            result = self.validate_single(measurement)
            results.append(result)
        
        report.results = results
        report.total_measurements = len(results)
        
        # Compute statistics
        self._compute_statistics(report)
        
        logger.info(f"Validation complete: {report.valid_projections}/{report.total_measurements} valid")
        logger.info(f"RMSE: {report.rmse:.3f} pixels")
        logger.info(f"Pass rate: {report.pass_rate:.1%}")
        
        return report
    
    def _compute_statistics(self, report: ValidationReport) -> None:
        """Compute statistics from validation results."""
        valid_results = [r for r in report.results if r.valid]
        invalid_results = [r for r in report.results if not r.valid]
        
        report.valid_projections = len(valid_results)
        report.invalid_projections = len(invalid_results)
        
        if not valid_results:
            return
        
        errors = np.array([r.error for r in valid_results])
        
        # Overall error statistics
        report.mean_error = float(np.mean(errors))
        report.std_error = float(np.std(errors))
        report.min_error = float(np.min(errors))
        report.max_error = float(np.max(errors))
        report.median_error = float(np.median(errors))
        report.rmse = float(np.sqrt(np.mean(errors ** 2)))
        
        # Threshold analysis
        within_threshold = errors <= report.threshold
        report.points_within_threshold = int(np.sum(within_threshold))
        report.points_outside_threshold = len(errors) - report.points_within_threshold
        report.pass_rate = report.points_within_threshold / len(errors)
        
        # Per-GCP statistics
        gcp_results: Dict[str, List[ReprojectionResult]] = {}
        for r in valid_results:
            if r.gcp_id not in gcp_results:
                gcp_results[r.gcp_id] = []
            gcp_results[r.gcp_id].append(r)
        
        for gcp_id, gcp_res in gcp_results.items():
            gcp_errors = [r.error for r in gcp_res]
            report.gcp_errors[gcp_id] = {
                'count': len(gcp_res),
                'mean_error': float(np.mean(gcp_errors)),
                'max_error': float(np.max(gcp_errors)),
                'rmse': float(np.sqrt(np.mean(np.array(gcp_errors) ** 2))),
            }
        
        # Per-image statistics
        image_results: Dict[str, List[ReprojectionResult]] = {}
        for r in valid_results:
            if r.image_id not in image_results:
                image_results[r.image_id] = []
            image_results[r.image_id].append(r)
        
        for image_id, img_res in image_results.items():
            img_errors = [r.error for r in img_res]
            report.image_errors[image_id] = {
                'count': len(img_res),
                'mean_error': float(np.mean(img_errors)),
                'max_error': float(np.max(img_errors)),
                'rmse': float(np.sqrt(np.mean(np.array(img_errors) ** 2))),
            }
    
    def save_report(
        self,
        report: ValidationReport,
        output_path: str,
        include_details: bool = True,
    ) -> None:
        """
        Save validation report to JSON file.
        
        Args:
            report: Validation report to save
            output_path: Path for output JSON file
            include_details: Whether to include per-measurement details
        """
        data = {
            'summary': {
                'total_measurements': report.total_measurements,
                'valid_projections': report.valid_projections,
                'invalid_projections': report.invalid_projections,
                'threshold_pixels': report.threshold,
                'points_within_threshold': report.points_within_threshold,
                'points_outside_threshold': report.points_outside_threshold,
                'pass_rate': report.pass_rate,
            },
            'error_statistics': {
                'mean_error': report.mean_error,
                'std_error': report.std_error,
                'min_error': report.min_error,
                'max_error': report.max_error,
                'median_error': report.median_error,
                'rmse': report.rmse,
            },
            'per_gcp': report.gcp_errors,
            'per_image': report.image_errors,
        }
        
        if include_details:
            data['details'] = [
                {
                    'gcp_id': r.gcp_id,
                    'gcp_name': r.gcp_name,
                    'image_id': r.image_id,
                    'measured_u': r.measured_u,
                    'measured_v': r.measured_v,
                    'projected_u': r.projected_u,
                    'projected_v': r.projected_v,
                    'error': r.error if r.valid else None,
                    'valid': r.valid,
                    'distance_m': r.distance_m,
                    'u_photo': r.u_photo,
                    'v_photo': r.v_photo,
                }
                for r in report.results
            ]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
    
    def save_residuals_csv(
        self,
        report: ValidationReport,
        output_path: str,
    ) -> None:
        """
        Save residuals to CSV for further analysis.
        
        Args:
            report: Validation report
            output_path: Path for output CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'gcp_id', 'gcp_name', 'image_id',
                'measured_u', 'measured_v',
                'projected_u', 'projected_v',
                'residual_u', 'residual_v',
                'error', 'distance_m', 'valid',
                'u_photo', 'v_photo'
            ])
            
            for r in report.results:
                residual_u = r.projected_u - r.measured_u if r.valid else None
                residual_v = r.projected_v - r.measured_v if r.valid else None
                writer.writerow([
                    r.gcp_id, r.gcp_name, r.image_id,
                    r.measured_u, r.measured_v,
                    r.projected_u, r.projected_v,
                    residual_u, residual_v,
                    r.error if r.valid else None,
                    r.distance_m,
                    r.valid,
                    r.u_photo, r.v_photo,
                ])
        
        logger.info(f"Residuals saved to {output_path}")


def run_validation(config_path: str, output_dir: Optional[str] = None) -> ValidationReport:
    """
    Convenience function to run validation from a config file.
    
    Args:
        config_path: Path to YAML configuration file
        output_dir: Optional directory for output files
        
    Returns:
        ValidationReport with results
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config and run validation
    config = Config.from_yaml(config_path)
    validator = GCPValidator(config)
    validator.load_data()
    report = validator.validate()
    
    # Save outputs if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        validator.save_report(report, str(output_path / 'validation_report.json'))
        validator.save_residuals_csv(report, str(output_path / 'residuals.csv'))
    
    return report
