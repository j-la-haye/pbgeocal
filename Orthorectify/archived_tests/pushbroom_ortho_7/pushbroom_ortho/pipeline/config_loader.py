"""
config_loader.py — Load and validate the pipeline YAML configuration.

Reads config.yaml and produces a structured namespace object with all
camera, mounting, trajectory, and processing parameters readily accessible.
Validates required fields and converts units (degrees → radians, lists → numpy).
"""

import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathsConfig:
    sbet: str
    bil_image: str
    bil_header: str
    exposure_times: str
    dsm: str
    output: str


@dataclass
class CRSConfig:
    dsm_epsg: int
    output_epsg: int
    sbet_ellipsoid: str


@dataclass
class SmileConfig:
    """Along-track smile polynomial: θ_at = a·xt² + b·xt + c (degrees)."""
    a: float    # quadratic (smile curvature)
    b: float    # linear (smile tilt)
    c: float    # constant (boresight along-track offset)


@dataclass
class OpticsConfig:
    """
    In-flight focal length and principal point correction.

    f_lab : effective lab focal length [pixels] (derived from LUT mean IFOV)
    f     : in-flight focal length [pixels]  (= f_lab for no correction)
    cx    : across-track PP shift [pixels]   (0 = no shift)
    cy    : along-track  PP shift [pixels]   (0 = no shift)
    """
    f_lab: float
    f: float
    cx: float
    cy: float


@dataclass
class CameraConfig:
    """
    Decoupled pushbroom camera model config.

    Across-track: lab-measured angle LUT (path to CSV)
    Along-track:  smile polynomial coefficients
    In-flight:    focal length and PP correction (f, cx, cy)
    BIL mapping:  first_valid_pixel offset
    """
    angle_lut_path: str
    smile: SmileConfig
    optics: OpticsConfig
    first_valid_pixel: int = 0    # BIL sample offset for LUT pixel 0


@dataclass
class MountingConfig:
    """Lever-arm, boresight, and mounting matrix from Body-NED to Camera."""
    lever_arm: np.ndarray            # (3,) in body frame [m]
    boresight_rad: np.ndarray        # (3,) roll, pitch, yaw in radians
    mounting_matrix: np.ndarray      # (3,3) body → camera


@dataclass
class NewtonConfig:
    max_iterations: int
    convergence_threshold: float
    numerical_dt: float


@dataclass
class VisibilityConfig:
    enabled: bool
    num_ray_samples: int
    height_tolerance: float


@dataclass
class ProcessingConfig:
    output_gsd: float
    tile_size: int
    num_workers: int
    newton: NewtonConfig
    resampling: str
    visibility: VisibilityConfig
    nodata: float


@dataclass
class PipelineConfig:
    paths: PathsConfig
    crs: CRSConfig
    camera: CameraConfig
    mounting: MountingConfig
    processing: ProcessingConfig


def load_config(config_path: str) -> PipelineConfig:
    """
    Parse config.yaml and return a fully validated PipelineConfig.

    Converts:
      - boresight angles from degrees to radians
      - mounting matrix from nested lists to (3,3) numpy array
      - lever arm from dict to (3,) numpy array
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    # --- Paths ---------------------------------------------------------------
    p = raw['paths']
    paths = PathsConfig(
        sbet=p['sbet'],
        bil_image=p['bil_image'],
        bil_header=p['bil_header'],
        exposure_times=p['exposure_times'],
        dsm=p['dsm'],
        output=p['output'],
    )

    # --- CRS -----------------------------------------------------------------
    c = raw['crs']
    crs = CRSConfig(
        dsm_epsg=c['dsm_epsg'],
        output_epsg=c['output_epsg'],
        sbet_ellipsoid=c.get('sbet_ellipsoid', 'WGS84'),
    )

    # --- Camera --------------------------------------------------------------
    cam = raw['camera']
    smile = cam['smile']
    optics = cam.get('optics', {})
    camera = CameraConfig(
        angle_lut_path=raw['paths']['angle_lut'],
        smile=SmileConfig(
            a=float(smile['a']),
            b=float(smile['b']),
            c=float(smile['c']),
        ),
        optics=OpticsConfig(
            f_lab=float(optics.get('f_lab', 1762.2)),
            f=float(optics.get('f', optics.get('f_lab', 1762.2))),
            cx=float(optics.get('cx', 0.0)),
            cy=float(optics.get('cy', 0.0)),
        ),
        first_valid_pixel=int(cam.get('first_valid_pixel', 0)),
    )

    # --- Mounting ------------------------------------------------------------
    m = raw['mounting']
    lever = m['lever_arm']
    bore = m['boresight']
    mounting = MountingConfig(
        lever_arm=np.array([lever['x'], lever['y'], lever['z']], dtype=np.float64),
        boresight_rad=np.deg2rad(np.array(
            [bore['roll'], bore['pitch'], bore['yaw']], dtype=np.float64
        )),
        mounting_matrix=np.array(m['mounting_matrix'], dtype=np.float64),
    )
    assert mounting.mounting_matrix.shape == (3, 3), "Mounting matrix must be 3×3"

    # --- Processing ----------------------------------------------------------
    pr = raw['processing']
    nw = pr['newton']
    vis = pr['visibility']
    processing = ProcessingConfig(
        output_gsd=float(pr['output_gsd']),
        tile_size=int(pr['tile_size']),
        num_workers=int(pr['num_workers']),
        newton=NewtonConfig(
            max_iterations=int(nw['max_iterations']),
            convergence_threshold=float(nw['convergence_threshold']),
            numerical_dt=float(nw['numerical_dt']),
        ),
        resampling=pr.get('resampling', 'bilinear'),
        visibility=VisibilityConfig(
            enabled=bool(vis['enabled']),
            num_ray_samples=int(vis['num_ray_samples']),
            height_tolerance=float(vis['height_tolerance']),
        ),
        nodata=float(pr.get('nodata', 0)),
    )

    return PipelineConfig(
        paths=paths,
        crs=crs,
        camera=camera,
        mounting=mounting,
        processing=processing,
    )
