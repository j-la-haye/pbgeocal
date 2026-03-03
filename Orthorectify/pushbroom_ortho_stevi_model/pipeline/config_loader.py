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
class DistortionConfig:
    """
    Steviapp degree-5 polynomial lens distortion.

    Δx(u) = Σ aᵢ · ((u − w/2) / w)ⁱ     across-track  [pixels]
    Δy(u) = Σ bᵢ · ((u − w/2) / w)ⁱ     along-track   [pixels]
    """
    delta_x: List[float]     # [a0, a1, a2, a3, a4, a5]
    delta_y: List[float]     # [b0, b1, b2, b3, b4, b5]


@dataclass
class CameraConfig:
    """
    Steviapp pushbroom camera model config.

    Pinhole intrinsics:  f, ppx
    Distortion:          Δx(u), Δy(u) polynomials
    BIL mapping:         first_valid_pixel, detector_width
    Angle LUT:           optional, for validation / fallback
    """
    focal_length: float           # f [pixels]
    principal_point: float        # ppx [pixels]
    detector_width: int           # w [pixels] — polynomial normalisation base
    distortion: DistortionConfig
    first_valid_pixel: int = 0
    angle_lut_path: Optional[str] = None


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
    

    # --- Paths ---------------------------------------------------------------
    #smile_raw = raw['camera']['smile']
    optics_raw = raw['camera'].get('optics', {})
    f_val = float(raw['camera'].get('focal_length',1732.6317))
    ppx_val = float(raw['camera'].get('principal_point',628.0674))

    p = raw['paths']
    output = p['output'].format(
        #a=float(smile_raw['a']),
        # b=float(smile_raw['b']),
        # c=float(smile_raw['c']),
        fl=f_val,
        ppx=ppx_val,
        #cx=float(optics_raw.get('cx', 0.0)),
        #cy=float(optics_raw.get('cy', 0.0)),
        gsd=float(raw['processing']['output_gsd']),
    )
    
    paths = PathsConfig(
        sbet=p['sbet'],
        bil_image=p['bil_image'],
        bil_header=p['bil_header'],
        exposure_times=p['exposure_times'],
        dsm=p['dsm'],
        output=output,
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
    dist = cam.get('distortion', {})
    camera = CameraConfig(
        focal_length=float(cam['focal_length']),
        principal_point=float(cam['principal_point']),
        detector_width=int(cam['detector_width']),
        distortion=DistortionConfig(
            delta_x=[float(x) for x in dist.get('delta_x', [0]*6)],
            delta_y=[float(x) for x in dist.get('delta_y', [0]*6)],
        ),
        first_valid_pixel=int(cam.get('first_valid_pixel', 0)),
        angle_lut_path=raw['paths'].get('angle_lut'),
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
