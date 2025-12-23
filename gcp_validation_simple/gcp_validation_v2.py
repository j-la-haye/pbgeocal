#!/usr/bin/env python3
"""
GCP Reprojection Validation Script

A straightforward implementation for validating GCP reprojection from
ECEF coordinates to image coordinates.

Usage:
    python gcp_validation_script.py config.yaml

Input Files (defined in config.yaml):
    - BINGO correspondence file: GCP observations in images
    - GCP ECEF coordinates: 3D positions in Earth-Centered Earth-Fixed
    - Trajectory CSV: Time-stamped poses (lat, lon, height, roll, pitch, yaw)
    - Timing file: Maps image_id to capture time

Coordinate Systems:
    - ECEF: X→0°lon, Y→90°E, Z→North Pole
    - NED: North-East-Down local tangent plane
    - Body: Forward-Right-Down (SBET convention)
    - Camera: X-right, Y-back, Z-down
    - Image: Origin top-left, U-right, V-down

"""

import numpy as np
import pyproj
import yaml
import csv
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from liblibor.map import TangentPlane, Trajectory,log, loadSBET
from liblibor.rotations import *
from pyproj import CRS, Transformer
from scipy.spatial.transform import Rotation as R
from photogrammetry_verify.io_utils import load_config, load_3d_csv, load_timing_file, parse_bingo_file
from photogrammetry_verify.camera import CameraModel
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Package imports
from photogrammetry_verify.camera import CameraModel
from photogrammetry_verify.transforms import Transform
from photogrammetry_verify.geometry import world_to_image
from photogrammetry_verify.geotools import GeoConverter
from photogrammetry_verify.io_utils import load_config, load_3d_csv, load_timing_file, parse_bingo_file
from photogrammetry_verify.trajectory import TrajectoryInterpolator
from photogrammetry_verify.geotools import get_grid_convergence
# Package imports
from liblibor.map import TangentPlane, Trajectory,log, loadSBET
from liblibor.rotations import *
from pyproj import CRS, Transformer
from scipy.spatial.transform import Rotation as R
import project_gcp2img


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float           # Focal length X (pixels)
    fy: float           # Focal length Y (pixels)
    cx: float           # Principal point X (pixels)
    cy: float           # Principal point Y (pixels)
    k1: float = 0.0     # Radial distortion k1
    k2: float = 0.0     # Radial distortion k2
    k3: float = 0.0     # Radial distortion k3
    p1: float = 0.0     # Tangential distortion p1
    p2: float = 0.0     # Tangential distortion p2
    image_width: int = 0
    image_height: int = 0


@dataclass
class Pose:
    """Camera/IMU pose."""
    latitude: float     # Degrees
    longitude: float    # Degrees
    height: float       # Meters (ellipsoidal)
    roll: float         # Degrees
    pitch: float        # Degrees
    yaw: float          # Degrees (heading from North)


@dataclass
class GCPObservation:
    """A GCP observation in an image."""
    gcp_id: int
    gcp_name: str
    image_id: int
    u_photo: float      # BINGO photo-coordinate U
    v_photo: float      # BINGO photo-coordinate V
    u_pixel: float      # Converted pixel coordinate U
    v_pixel: float      # Converted pixel coordinate V


@dataclass
class GCPCoordinate:
    """GCP 3D coordinates in ECEF."""
    gcp_id: int
    x: float            # ECEF X (meters)
    y: float            # ECEF Y (meters)
    z: float            # ECEF Z (meters)


# =============================================================================
# WGS84 CONSTANTS
# =============================================================================

WGS84_A = 6378137.0                     # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563           # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)       # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2   # First eccentricity squared


# =============================================================================
# FILE PARSING
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve file paths relative to config file
    config_dir = Path(config_path).parent
    files = config.get('files', {})
    for key in ['bingo_file', 'gcp_file', 'trajectory_file', 'timing_file']:
        if key in files:
            files[key] = str(config_dir / files[key])
    
    return config


def parse_bingo(filepath: str) -> List[Tuple[int, str, List[Tuple[int, float, float]]]]:
    """
    Parse BINGO correspondence file.
    
    Format:
        gcp_id gcp_name
        image_id U V
        image_id U V
        -99
    
    Returns:
        List of (gcp_id, gcp_name, [(image_id, u, v), ...])
    """
    blocks = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and delimiters
        if not line or line == '-99':
            i += 1
            continue
        
        # Try to parse as header (gcp_id gcp_name)
        parts = line.split(None, 1)
        if len(parts) >= 2:
            try:
                gcp_id = int(parts[0])
                gcp_name = parts[1].strip()
                observations = []
                i += 1
                
                # Read observations until -99
                while i < len(lines):
                    obs_line = lines[i].strip()
                    if obs_line == '-99':
                        i += 1
                        break
                    if not obs_line:
                        i += 1
                        continue
                    
                    obs_parts = obs_line.split()
                    if len(obs_parts) >= 3:
                        try:
                            img_id = int(obs_parts[0])
                            u = float(obs_parts[1])
                            v = float(obs_parts[2])
                            observations.append((img_id, u, v))
                        except ValueError:
                            pass
                    i += 1
                
                if observations:
                    blocks.append((gcp_id, gcp_name, observations))
            except ValueError:
                i += 1
        else:
            i += 1
    
    return blocks


def parse_gcp_file(filepath: str ,epsg=None)-> Dict[int, GCPCoordinate]:
    """
    Parse GCP ECEF coordinates file.
    
    Expected CSV columns: gcp_id, x, y, z
    """
    gcps = {}
    
    enu2ecefTransformer = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)


    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f,fieldnames=['gcp_id','x','y','z'])
        for row in reader:
            if epsg is not None:
                x, y, z = enu2ecefTransformer.transform(float(row['x']), float(row['y']), float(row['z']))
            else:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
            gcp_id = int(row['gcp_id'])
            gcps[gcp_id] = GCPCoordinate(
                gcp_id=gcp_id,
                x=x,
                y=y,
                z=z,
            )
    
    return gcps


def parse_timing_file(filepath: str) -> Dict[int, float]:
    """
    Parse image timing file.
    
    Format: image_id, time (CSV or space-delimited)
    """
    timings = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try comma-separated, then space-separated
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            
            if len(parts) >= 2:
                try:
                    image_id = int(parts[0].strip())
                    time = float(parts[1].strip())
                    timings[image_id] = time
                except ValueError:
                    continue
    
    return timings


def parse_trajectory_file(filepath: str) -> List[Tuple[float, Pose]]:
    """
    Parse trajectory CSV file.
    
    Expected columns: time, latitude, longitude, height, roll, pitch, yaw
    
    Returns:
        List of (time, Pose) sorted by time
    """
    epochs = []
    
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time = float(row['time'])
            pose = Pose(
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                height=float(row['height']),
                roll=float(row['roll']),
                pitch=float(row['pitch']),
                yaw=float(row['yaw']),
            )
            epochs.append((time, pose))
    
    # Sort by time
    epochs.sort(key=lambda x: x[0])
    return epochs


# =============================================================================
# COORDINATE CONVERSIONS
# =============================================================================

def bingo_to_pixel(u_photo: float, v_photo: float, 
                   image_width: int, image_height: int,
                   v_axis_up: bool = True) -> Tuple[float, float]:
    """
    Convert BINGO photo-coordinates to pixel coordinates.
    
    BINGO: Origin at image center
    Pixel: Origin at top-left
    
    Args:
        u_photo: BINGO U coordinate (positive right)
        v_photo: BINGO V coordinate (positive up if v_axis_up=True)
        image_width: Image width in pixels
        image_height: Image height in pixels
        v_axis_up: True if BINGO V is positive upward (photogrammetric)
    
    Returns:
        (u_pixel, v_pixel) with origin at top-left
    """
    # U: shift origin from center to left edge
    u_pixel = u_photo + image_width / 2.0
    
    # V: handle axis direction
    if v_axis_up:
        # Photogrammetric: V positive up → flip for pixel coords (V positive down)
        v_pixel = image_height / 2.0 - v_photo
    else:
        # V already positive down
        v_pixel = v_photo + image_height / 2.0
    
    return u_pixel, v_pixel


def geodetic_to_ecef(lat: float, lon: float, height: float,radians=True) -> np.ndarray:
    """
    Convert geodetic (lat, lon, height) to ECEF (X, Y, Z).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        height: Ellipsoidal height in meters
    
    Returns:
        ECEF coordinates as numpy array [X, Y, Z]
    """
    if not radians:
        lat = np.radians(lat)
        lon = np.radians(lon)
    lat_rad = lat
    lon_rad = lon
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    
    # ECEF coordinates
    X = (N + height) * cos_lat * cos_lon
    Y = (N + height) * cos_lat * sin_lon
    Z = (N * (1 - WGS84_E2) + height) * sin_lat
    
    return np.array([X, Y, Z])


# =============================================================================
# ROTATION MATRICES
# =============================================================================

def rotation_ecef_to_ned(lat: float, lon: float,radians=True) -> np.ndarray:
    """
    Compute rotation matrix from ECEF to NED (local tangent plane).
    
    NED axes at position (lat, lon):
        - North: Points north along meridian
        - East: Points east along parallel
        - Down: Points toward Earth center (nadir)
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
    
    Returns:
        3x3 rotation matrix R such that v_ned = R @ v_ecef
    """
    if not radians:
        lat = np.radians(lat)
        lon = np.radians(lon)
    lat_rad = lat
    lon_rad = lon
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # Each row is a unit vector in ECEF for N, E, D directions
    R = np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],   # North
        [-sin_lon,            cos_lon,             0      ],   # East
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]    # Down
    ])
    
    return R


def rotation_ned_to_body(roll: float, pitch: float, yaw: float,radians=True) -> np.ndarray:
    """
    Compute rotation matrix from NED to Body frame using ZYX Euler angles.
    
    Body frame (SBET convention): Forward-Right-Down
    
    Euler angles:
        - Yaw (ψ): Rotation about Down axis (heading from North)
        - Pitch (θ): Rotation about Right axis (nose up/down)
        - Roll (φ): Rotation about Forward axis (bank left/right)
    
    Order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    
    Args:
        roll: Roll angle in degrees
        pitch: Pitch angle in degrees
        yaw: Yaw/heading angle in degrees
    
    Returns:
        3x3 rotation matrix R such that v_body = R @ v_ned
    """
    if not radians:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    phi = roll
    theta = pitch
    psi = yaw
    
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    
    # ZYX Euler rotation matrix
    R = np.array([
        [cos_psi * cos_theta,
         cos_psi * sin_theta * sin_phi - sin_psi * cos_phi,
         cos_psi * sin_theta * cos_phi + sin_psi * sin_phi],
        
        [sin_psi * cos_theta,
         sin_psi * sin_theta * sin_phi + cos_psi * cos_phi,
         sin_psi * sin_theta * cos_phi - cos_psi * sin_phi],
        
        [-sin_theta,
         cos_theta * sin_phi,
         cos_theta * cos_phi]
    ])
    
    return R


def rotation_body_to_camera() -> np.ndarray:
    """
    Fixed rotation matrix from Body frame to Camera frame.
    
    Body frame:  Forward (X), Right (Y), Down (Z)
    Camera frame: Right (X), Back (Y), Down (Z)
    
    Mapping:
        Camera X (right) = Body Y (right)
        Camera Y (back)  = -Body X (forward)
        Camera Z (down)  = Body Z (down)
    
    Returns:
        3x3 rotation matrix R such that v_camera = R @ v_body
    """
    R = np.array([
        [0,  -1,  0],   # Camera X = Body Y
        [-1, 0,  0],   # Camera Y = -Body X
        [0,  0,  1]    # Camera Z = Body Z
    ])
    return R


# =============================================================================
# TRAJECTORY INTERPOLATION
# =============================================================================

def interpolate_angle(angle1: float, angle2: float, t: float) -> float:
    """
    Interpolate between two angles, handling wrap-around at 360°.
    
    Args:
        angle1: Starting angle in degrees
        angle2: Ending angle in degrees
        t: Interpolation factor (0 = angle1, 1 = angle2)
    
    Returns:
        Interpolated angle in degrees
    """
    diff = angle2 - angle1
    
    # Handle wrap-around (e.g., 350° to 10° should go through 0°)
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    
    return angle1 + t * diff


def interpolate_pose(pose1: Pose, pose2: Pose, t: float) -> Pose:
    """
    Interpolate between two poses.
    
    Args:
        pose1: Starting pose
        pose2: Ending pose
        t: Interpolation factor (0 = pose1, 1 = pose2)
    
    Returns:
        Interpolated Pose
    """
    return Pose(
        latitude=pose1.latitude + t * (pose2.latitude - pose1.latitude),
        longitude=pose1.longitude + t * (pose2.longitude - pose1.longitude),
        height=pose1.height + t * (pose2.height - pose1.height),
        roll=interpolate_angle(pose1.roll, pose2.roll, t),
        pitch=interpolate_angle(pose1.pitch, pose2.pitch, t),
        yaw=interpolate_angle(pose1.yaw, pose2.yaw, t),
    )


def get_pose_at_time(trajectory: List[Tuple[float, Pose]], query_time: float) -> Optional[Pose]:
    """
    Interpolate trajectory to get pose at a specific time.
    
    Args:
        trajectory: List of (time, Pose) sorted by time
        query_time: Time to query
    
    Returns:
        Interpolated Pose, or None if time is outside trajectory range
    """
    if not trajectory:
        return None
    
    times = [t for t, _ in trajectory]
    
    # Check bounds
    if query_time < times[0] or query_time > times[-1]:
        return None
    
    # Binary search for bracketing epochs
    idx = np.searchsorted(times, query_time)
    
    if idx == 0:
        return trajectory[0][1]
    if idx >= len(trajectory):
        return trajectory[-1][1]
    
    # Get bracketing epochs
    t1, pose1 = trajectory[idx - 1]
    t2, pose2 = trajectory[idx]
    
    # Compute interpolation factor
    if t2 == t1:
        return pose1
    
    t = (query_time - t1) / (t2 - t1)
    
    return interpolate_pose(pose1, pose2, t)


# =============================================================================
# MAIN TRANSFORMATION PIPELINE
# =============================================================================

def transform_ecef_to_camera(
    point_ecef: np.ndarray,
    pose: Pose,
    lever_arm: np.ndarray = np.zeros(3),
    boresight: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Transform point(s) from ECEF to camera frame.
    
    Pipeline:
        1. Compute camera position in ECEF (with lever arm)
        2. Compute vector from camera to point in ECEF
        3. Rotate: ECEF → NED → Body → Camera
    
    Args:
        point_ecef: Point coordinates in ECEF, shape (3,) or (N, 3)
        pose: Camera/IMU pose
        lever_arm: Offset from IMU to camera in body frame [forward, right, down]
        boresight: Boresight correction angles (roll, pitch, yaw) in degrees
    
    Returns:
        Point coordinates in camera frame, same shape as input
    """
    # Handle both single point (3,) and multiple points (N, 3)
    is_single_point = point_ecef.ndim == 1
    if is_single_point:
        point_ecef = point_ecef.reshape(1, 3)
    
    # Step 1: Compute camera position in ECEF
    # First get IMU position
    latitude, longitude, height = pose.lla[0], pose.lla[1], pose.lla[2]
    roll, pitch, yaw = pose.rpy[0], pose.rpy[1], pose.rpy[2]
    imu_ecef = geodetic_to_ecef(latitude, longitude, height)
    
    # Build rotation from body to ECEF (for lever arm transformation)
    R_ned_ecef = rotation_ecef_to_ned(latitude, longitude)
    R_body_ned = rotation_ned_to_body(roll, pitch, yaw)
    R_body_ecef = R_body_ned @ R_ned_ecef  # Body ← NED ← ECEF
    R_ecef_body = R_body_ecef.T            # ECEF ← Body (transpose = inverse for rotation)
    
    # Transform lever arm from body to ECEF and add to get camera position
    lever_arm_ecef = R_ecef_body @ lever_arm
    camera_ecef = imu_ecef + lever_arm_ecef
    
    # Step 2: Vector from camera to point in ECEF (N, 3)
    delta_ecef = point_ecef - camera_ecef
    
    # Step 3: Build full rotation chain ECEF → Camera
    # Boresight correction (small rotation applied before body-to-camera)
    R_boresight = rotation_ned_to_body(boresight[0], boresight[1], boresight[2])
    
    # Body to camera (fixed rotation)
    R_cam_body = rotation_body_to_camera()
    
    # Full chain: Camera ← Boresight ← Body ← NED ← ECEF
    R_cam_ecef = R_cam_body @ R_boresight @ R_body_ecef
    
    # Transform points to camera frame: (N, 3) @ (3, 3)^T = (N, 3)
    point_camera = delta_ecef @ R_cam_ecef
    
    # Return original shape
    if is_single_point:
        return point_camera.squeeze(0)
    return point_camera


def project_to_image(
    point_camera: np.ndarray,
    camera: CameraIntrinsics,
    apply_distortion: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D point(s) in camera frame to 2D image coordinates.
    
    Args:
        point_camera: Point(s) in camera frame, shape (3,) or (N, 3)
        camera: Camera intrinsic parameters
        apply_distortion: Whether to apply lens distortion
    
    Returns:
        (u, v, valid): Pixel coordinates and validity flags
            - If input is (3,): returns (float, float, bool)
            - If input is (N, 3): returns (ndarray[N], ndarray[N], ndarray[N] of bool)
    """
    # Handle both single point (3,) and multiple points (N, 3)
    is_single_point = point_camera.ndim == 1
    if is_single_point:
        point_camera = point_camera.reshape(1, 3)
    
    X = point_camera[:, 0]
    Y = point_camera[:, 1]
    Z = point_camera[:, 2]
    
    # Check if points are in front of camera
    valid = Z > 0
    
    # Initialize output arrays
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    # Only process valid points
    if np.any(valid):
        X_valid = X[valid]
        Y_valid = Y[valid]
        Z_valid = Z[valid]
        
        # Perspective projection (normalized coordinates)
        x = X_valid / Z_valid
        y = Y_valid / Z_valid
        
        # Apply distortion if requested
        if apply_distortion:
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            
            # Radial distortion
            radial = 1 + camera.k1 * r2 + camera.k2 * r4 + camera.k3 * r6
            
            # Tangential distortion
            x_d = x * radial + 2 * camera.p1 * x * y + camera.p2 * (r2 + 2 * x * x)
            y_d = y * radial + camera.p1 * (r2 + 2 * y * y) + 2 * camera.p2 * x * y
        else:
            x_d, y_d = x, y
        
        # Apply intrinsics
        u[valid] = camera.fx * x_d + camera.cx
        v[valid] = camera.fy * y_d + camera.cy
    
    # Return in original format
    if is_single_point:
        return u[0], v[0], valid[0]
    return u, v, valid


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def run_validation(config_path: str, verbose: bool = True):
    """
    Run the complete GCP reprojection validation pipeline.
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Print detailed output
    """
    print("=" * 70)
    print("GCP REPROJECTION VALIDATION")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load configuration
    # -------------------------------------------------------------------------
    print("\n[1] Loading configuration...")
    config = load_config(config_path)
    
    # Camera intrinsics
    cam_cfg = config['camera']
    camera_model = CameraModel.from_dict(config['camera'])
    camera = CameraIntrinsics(
        fx=cam_cfg['fx'],
        fy=cam_cfg['fy'],
        cx=cam_cfg['cx'],
        cy=cam_cfg['cy'],
        k1=cam_cfg.get('k1', 0.0),
        k2=cam_cfg.get('k2', 0.0),
        k3=cam_cfg.get('k3', 0.0),
        p1=cam_cfg.get('p1', 0.0),
        p2=cam_cfg.get('p2', 0.0),
        image_width=cam_cfg.get('width', 0),
        image_height=cam_cfg.get('height', 0),
    )
    print(f"    Camera: fx={camera.fx}, fy={camera.fy}, cx={camera.cx}, cy={camera.cy}")
    print(f"    Image size: {camera.image_width} x {camera.image_height}")
    print(f"    Distortion: k1={camera.k1}, k2={camera.k2}, k3={camera.k3}")
    
    # Lever arm
    lever_cfg = config.get('lever_arm', {})
    lever_arm = np.array([
        lever_cfg.get('x', 0.0),  # Forward
        lever_cfg.get('y', 0.0),  # Right
        lever_cfg.get('z', 0.0),  # Down
    ])
    print(f"    Lever arm: [{lever_arm[0]}, {lever_arm[1]}, {lever_arm[2]}] (F-R-D)")
    
    # Boresight
    bore_cfg = config.get('boresight', {})
    boresight = (
        bore_cfg.get('roll', 0.0),
        bore_cfg.get('pitch', 0.0),
        bore_cfg.get('yaw', 0.0),
    )
    print(f"    Boresight: roll={boresight[0]}, pitch={boresight[1]}, yaw={boresight[2]}")
    
    # Conventions
    conv_cfg = config.get('conventions', {})
    v_axis_up = conv_cfg.get('v_axis_up', True)
    print(f"    V-axis up: {v_axis_up}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Load BINGO correspondence file
    # -------------------------------------------------------------------------
    print("\n[2] Loading BINGO correspondence file...")
     # 2. Parse BINGO Correspondence File
    print(f"Parsing BINGO file: {config['files']['bingo_file']}...")
    bingo_data = parse_bingo_file(config['files']['bingo_file'])
    print(f"Found {len(bingo_data)} image blocks.")

    global_errors = []

    #Filter BINGO data to only tiepoint_id < 1000 corresponding to 3D Checkpoints
    bingo_data = [block for block in bingo_data if (int(block['points']['tiepoint_id'].iloc[0]) < 1000)]
    
    # -------------------------------------------------------------------------
    # STEP 3: Load GCP ECEF coordinates
    # -------------------------------------------------------------------------
    print("\n[3] Loading GCP ECEF coordinates...")
    gcp_path = config['files']['gcp_file']
    gcps = parse_gcp_file(gcp_path, epsg=config['project']['epsg'])
    print(f"    Loaded {len(gcps)} GCP coordinates")
    
    if verbose and gcps:
        print("\n    Sample GCP coordinates:")
        for gcp_id, gcp in list(gcps.items())[:3]:
            print(f"      GCP {gcp_id}: X={gcp.x:.3f}, Y={gcp.y:.3f}, Z={gcp.z:.3f}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Load trajectory and timing
    # -------------------------------------------------------------------------
    
    # Load Timing (Image ID -> Time)
    timing_map = load_timing_file(config['files']['timing_file'])

    # Define time span of images
    img_times = np.array([timing_map[img_id]  for img_id in timing_map])
    time_buffer = 3
    img_time_span = [img_times.min()-time_buffer, img_times.max()+time_buffer]
   
    t_start, t_end = img_time_span
    print(f"    Trajectory time range: {t_start:.3f} to {t_end:.3f}")
    
    log("[2/3] Loading SBET data...", verbose=True, force=True)
    # Extract time, lla, rpy from sbet_df
    t,lla,rpy = loadSBET(Path(config['files']['trajectory_file']))
    
    mask = (t >= img_time_span[0]) & (t <= img_time_span[1])
    #tspan = t[mask]
    img_lla = lla[mask,:]
    #rpy = rpy[mask,:]
    lat0 = np.degrees(img_lla[0,0])
    lon0 = np.degrees(img_lla[0,1])
    alt0 = img_lla[0,2]
    print(f"Reference LTP origin: lat: {lat0:.6f} lon: {lon0:.6f} alt: {alt0:.3f}")
    tangentPlane = TangentPlane(lat0, lon0,alt0)
    

    
    trajectory = Trajectory(t, lla, rpy, tangentPlane, img_time_span)
    print(f"    Loaded {len(trajectory.t)} trajectory epochs")
    log("[3/3] Interpolating poses...", verbose=True, force=True)
    # Create coordinate transformer
    
    img_poses = trajectory.interpolate(img_times, config)
    
    
    # print("\n[4] Loading trajectory and timing...")
    
    # timing_path = config['files']['timing_file']
    # timings = parse_timing_file(timing_path)
    # print(f"    Loaded {len(timings)} image timings")
    
    # traj_path = config['files']['trajectory_file']
    # trajectory = parse_trajectory_file(traj_path)
    # 
    
    # if trajectory:
    #     t_start, t_end = trajectory[0][0], trajectory[-1][0]
    #     print(f"    Trajectory time range: {t_start:.3f} to {t_end:.3f}")
    
    # # -------------------------------------------------------------------------
    # # STEP 5: Interpolate poses for each image
    # # -------------------------------------------------------------------------
    # print("\n[5] Interpolating poses for images...")
    
    # image_poses = {}
    # for image_id, time in timings.items():
    #     pose = get_pose_at_time(trajectory, time)
    #     if pose is not None:
    #         image_poses[image_id] = pose
    
    # print(f"    Interpolated poses for {len(image_poses)} images")
    
    # if verbose and image_poses:
    #     print("\n    Sample interpolated poses:")
    #     for img_id, pose in list(image_poses.items())[:3]:
    #         print(f"      Image {img_id}: lat={pose.latitude:.6f}, lon={pose.longitude:.6f}, "
    #               f"h={pose.height:.1f}, r={pose.roll:.2f}, p={pose.pitch:.2f}, y={pose.yaw:.2f}")
    
    # -------------------------------------------------------------------------
    # STEP 6: Run reprojection for each observation
    # -------------------------------------------------------------------------
    print("\n[6] Running reprojection validation...")
    
    results = []
    
    for img_block in bingo_data:
        # Get GCP coordinates
        img_id = img_block['img_id']
        
        # A. Get Timestamp
        if img_id not in timing_map:
            print(f"Warning: Image4329364 ID {img_id} not found in timing file. Skipping.")
            continue
        timestamp = timing_map[img_id]

        
        
        # Get pose for this image
        pose_idx = np.where([pose.t == timestamp for pose in img_poses])[0]
        try:
            # select pose at timestamp from img_poses
            pose_idx = np.where([pose.t == timestamp for pose in img_poses])[0]
            if len(pose_idx) == 0:
                raise ValueError(f"Timestamp {timestamp} not found in interpolated poses")
            pose = img_poses[pose_idx[0]]
            
        except ValueError:
            print(f"Skipping {img_id}: Time {timestamp} out of trajectory bounds.")
            continue
        
        obs = img_block['points']
        # Join on 'tiepoint_id' (obs) vs index (3D)
        valid_obs = obs[obs['tiepoint_id'].astype(int).isin(gcps.keys())]
        
        if valid_obs.empty:
            continue
        
        # 5. Image Projection (BINGO U/V to Pixel U/V)
        # BINGO U is Right, V is UP. Pixel U is Right, V is DOWN.
        obs_u_px = camera_model.K[0, 2] + valid_obs['u_bingo'].values
        obs_v_px = camera_model.K[1, 2] - valid_obs['v_bingo'].values
        obs_px = np.stack([obs_u_px, obs_v_px], axis=1)

        # Localize in Grid Frame
        tie_ids = valid_obs['tiepoint_id'].astype(int).values
        gcp_ecef = np.array([[gcps[tid].x, gcps[tid].y, gcps[tid].z] for tid in tie_ids])

        # Transform GCP to camera frame
        point_camera = transform_ecef_to_camera(
            gcp_ecef, pose, lever_arm, boresight
        )
        
        # Project to image
        proj_u, proj_v, valid_proj = project_to_image(point_camera, camera)

        u_rpj,v_rpj = project_gcp2img.reproject_3d_to_image_full(gcp_ecef,pose,camera_model.K, lever_arm, boresight)
        
        # Process each GCP observation
        for i, (tid, is_valid) in enumerate(zip(tie_ids, valid_proj)):
            if not is_valid:
                print(f"    WARNING: Invalid projection for GCP {tid} in image {img_id} "
                      f"(point behind camera, Z={point_camera[i, 2]:.1f})")
                continue
            
            # Get corresponding observation
            obs_row = valid_obs.iloc[i]
            
            # Compute error
            error = np.sqrt((proj_u[i] - obs_u_px[i]) ** 2 + (proj_v[i] - obs_v_px[i]) ** 2)
            
            results.append({
                'gcp_id': int(tid),
                'gcp_name': str(tid),  # You may want to get the actual name from somewhere
                'image_id': img_id,
                'measured_u': obs_u_px[i],
                'measured_v': obs_v_px[i],
                'projected_u': proj_u[i],
                'projected_v': proj_v[i],
                'residual_u': proj_u[i] - obs_u_px[i],
                'residual_v': proj_v[i] - obs_v_px[i],
                'error': error,
                'distance_m': np.linalg.norm(point_camera[i]),
                'point_camera': point_camera[i].copy(),
            })
    
    print(f"    Processed {len(results)} valid reprojections")
    
    # -------------------------------------------------------------------------
    # STEP 7: Compute and display statistics
    # -------------------------------------------------------------------------
    print("\n[7] Computing statistics...")
    
    if not results:
        print("    ERROR: No valid results to analyze!")
        return
    
    errors = [r['error'] for r in results]
    residuals_u = [r['residual_u'] for r in results]
    residuals_v = [r['residual_v'] for r in results]
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    median_error = np.median(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    
    mean_res_u = np.mean(residuals_u)
    mean_res_v = np.mean(residuals_v)
    std_res_u = np.std(residuals_u)
    std_res_v = np.std(residuals_v)
    
    threshold = config.get('validation_threshold', 3.0)
    within_threshold = sum(1 for e in errors if e <= threshold)
    pass_rate = within_threshold / len(errors) * 100
    
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nTotal observations:     {len(results)}")
    print(f"\nError Statistics (pixels):")
    print(f"  Mean error:           {mean_error:.4f}")
    print(f"  Std deviation:        {std_error:.4f}")
    print(f"  Min error:            {min_error:.4f}")
    print(f"  Max error:            {max_error:.4f}")
    print(f"  Median error:         {median_error:.4f}")
    print(f"  RMSE:                 {rmse:.4f}")
    print(f"\nResidual Statistics (pixels):")
    print(f"  Mean residual U:      {mean_res_u:.4f}")
    print(f"  Mean residual V:      {mean_res_v:.4f}")
    print(f"  Std residual U:       {std_res_u:.4f}")
    print(f"  Std residual V:       {std_res_v:.4f}")
    print(f"\nThreshold Analysis ({threshold} pixels):")
    print(f"  Within threshold:     {within_threshold}/{len(results)}")
    print(f"  Pass rate:            {pass_rate:.1f}%")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 8: Detailed results
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[8] Detailed Results:")
        print("-" * 100)
        print(f"{'GCP':>6} {'Name':<25} {'Image':>6} {'Meas_U':>10} {'Meas_V':>10} "
              f"{'Proj_U':>10} {'Proj_V':>10} {'Error':>8}")
        print("-" * 100)
        
        for r in sorted(results, key=lambda x: x['error'], reverse=True):
            print(f"{r['gcp_id']:>6} {r['gcp_name']:<25} {r['image_id']:>6} "
                  f"{r['measured_u']:>10.2f} {r['measured_v']:>10.2f} "
                  f"{r['projected_u']:>10.2f} {r['projected_v']:>10.2f} "
                  f"{r['error']:>8.3f}")
        
        print("-" * 100)
    
    # -------------------------------------------------------------------------
    # STEP 9: Save results to CSV
    # -------------------------------------------------------------------------
    output_path = Path(config_path).parent / 'validation_results.csv'
    print(f"\n[9] Saving results to {output_path}...")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'gcp_id', 'gcp_name', 'image_id',
            'measured_u', 'measured_v',
            'projected_u', 'projected_v',
            'residual_u', 'residual_v',
            'error', 'distance_m',
            'camera_x', 'camera_y', 'camera_z'
        ])
        for r in results:
            writer.writerow([
                r['gcp_id'], r['gcp_name'], r['image_id'],
                f"{r['measured_u']:.4f}", f"{r['measured_v']:.4f}",
                f"{r['projected_u']:.4f}", f"{r['projected_v']:.4f}",
                f"{r['residual_u']:.4f}", f"{r['residual_v']:.4f}",
                f"{r['error']:.4f}", f"{r['distance_m']:.2f}",
                f"{r['point_camera'][0]:.4f}",
                f"{r['point_camera'][1]:.4f}",
                f"{r['point_camera'][2]:.4f}",
            ])
    
    print("\nDone!")
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python gcp_validation_script.py config.yaml")
        print("\nExample config.yaml:")
        print("""
camera:
  fx: 8000.0
  fy: 8000.0
  cx: 4000.0
  cy: 3000.0
  k1: -0.08
  k2: 0.005
  k3: 0.0
  p1: 0.0
  p2: 0.0
  image_width: 8000
  image_height: 6000

lever_arm:
  x: 0.15    # Forward (meters)
  y: 0.05    # Right (meters)
  z: -0.30   # Down (meters, negative = up)

boresight:
  roll: 0.0   # degrees
  pitch: 0.0  # degrees
  yaw: 0.0    # degrees

conventions:
  v_axis_up: true  # BINGO V-axis positive upward

files:
  bingo_file: correspondences.bingo
  gcp_file: gcp_ecef.csv
  trajectory_file: trajectory.csv
  timing_file: image_timing.csv

validation_threshold: 3.0  # pixels
""")
        sys.exit(1)
    
    config_path = 'gcp_validation_simple/validation_test/config.yaml'

    run_validation(config_path, verbose=True)
