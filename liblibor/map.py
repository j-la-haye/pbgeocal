from logging import config
import sys
import os
import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R, Slerp
from liblibor.rotations import *
from pyproj import CRS, Transformer
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from photogrammetry_verify.geotools import GeoConverter

proj_ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
proj_lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

lla2ecefTransformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
ecef2llaTransformer = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326")

from liblibor.rotations import *

class TangentPlane:
    def __init__(self, lat, lon,alt):
        print(f"Setting up tangent plane at lat: {lat}°, lon: {lon}°")
        self.lat = np.radians(lat)
        self.lon = np.radians(lon)
        self.alt = alt

        self.xyz0 = np.array(lla2ecefTransformer.transform(self.lat,self.lon,self.alt, radians=True)).reshape(1,3)

        self.R_ecef2enu = T_enu_ned() @ R_ned2e(self.lat, self.lon).T
        print("R_ecef2enu:", self.R_ecef2enu)

class Trajectory:
    def __init__(self, t, lla, rpy, tp, t_span=None,radians=True):
        if t_span is None:
            t_span = [t[0], t[-1]]
        
        mask = (t >= t_span[0]) & (t <= t_span[1])
        t = t[mask]
        lla = lla[mask,:]
        rpy = rpy[mask,:]
        
        self.t = t
        self.lla = lla.T
        self.rpy = rpy.T
        self.ecef = np.dstack(lla2ecefTransformer.transform(lla[:, 0], lla[:, 1], lla[:, 2],radians=radians))[0]
        self.xyz = tp.R_ecef2enu @ (self.ecef - tp.xyz0).T
        self.R_ned2body = np.empty((rpy.shape[0],3,3))
        self.R_ned2ecef = np.empty((lla.shape[0],3,3))
        self.R_b2enu = np.empty((rpy.shape[0],3,3))
        self.q_b2m = np.zeros((rpy.shape[0],4))
        
        for i in range(len(t)):
            self.R_ned2body[i] = R_ned2b(rpy[i,0], rpy[i,1], rpy[i,2])
            self.R_ned2ecef[i] = R_ned2e(lla[i,0], lla[i,1])
            self.R_b2enu[i] = R_e2enu(lla[i,0], lla[i,1]) @ self.R_ned2ecef[i] @ self.R_ned2body[i].T
            #self.R_b2enu[i] = tp.R_ecef2enu @ self.R_ned2ecef[i] @ self.R_ned2body[i].T
            self.q_b2m[i,:] = dcm2quat_scipy(self.R_b2enu[i])
            #self.q_b2m[i,:] = dcm2quat(self.R_b2enu[i])
        
    def interpolate(self, timestamps, cfg,customRPY = True):
        
        transformer = Transformer.from_crs(
            4326,
            cfg['project']['epsg'],
        always_xy=False  # Ensures lon, lat order
        )
        # Step 7: Convert camera position to projected coordinates
        ENH_interp = np.empty((len(timestamps), 3))

        xyz_interp = np.empty((len(timestamps), 3))
        lla_interp = np.empty((len(timestamps), 3))
        for i in range(3):
            xyz_interp[:,i] = np.interp(timestamps, self.t, self.xyz[i,:])
            lla_interp[:,i] = np.interp(timestamps, self.t, self.lla[i,:])
        
        E, N, H = transformer.transform(lla_interp[:,0], lla_interp[:,1], lla_interp[:,2],radians=True)
        ENH_interp = np.array([E,N,H]) 

        #R_ned2b_interp = self.slerp_ned2b(timestamps)
        #R_ned2e_interp = self.slerp_ned2e(timestamps).as_matrix()

        
        if customRPY:
            rpy_interp = np.empty((len(timestamps),3))
            for i in range(len(timestamps)):
                rpy_interp[i,:] = rpy_from_R_ned2b(self.R_ned2body[i], as_degrees=False)
        else:
            rpy_interp = self.R_ned2body.as_euler('xyz', degrees=False).T

        poses = []
        for i in range(len(timestamps)):
            poses.append(Pose_std(timestamps[i], lla_interp[i,:], xyz_interp[i,:],rpy_interp[i,:],self.R_ned2ecef[i], self.R_ned2body[i],self.ecef[i,:], ENH_interp[:,i]))

        return poses
    
class Pose:
    def __init__(self, t, lla, xyz, rpy,  R_ned2e):
        self.t = t
        self.lla = lla
        self.xyz = xyz
        self.R_ned2e = R_ned2e
        self.rpy = rpy

class CamPose:
    def __init__(self, data_dict):
        # define np.array for each key in data_dict dictionary list

        self.t = np.array([d['time'] for d in data_dict])
        self.easting = np.array([d['easting'] for d in data_dict])
        self.northing = np.array([d['northing'] for d in data_dict])
        self.ellip_height = np.array([d['ellip_height'] for d in data_dict])
        self.omega = np.array([d['omega'] for d in data_dict])
        self.phi = np.array([d['phi'] for d in data_dict])
        self.kappa = np.array([d['kappa'] for d in data_dict])
        self.lat = np.array([d['lat'] for d in data_dict])
        self.lon = np.array([d['lon'] for d in data_dict])
    def get_field_names(self):
        return ['t', 'easting', 'northing', 'ellip_height', 'omega', 'phi', 'kappa', 'lat', 'lon']

class Pose_std:
    def __init__(self, t, lla, xyz, rpy, R_ned2e,R_ned2b, ecef,ENH=None, std=None, opk=None):
        self.t = t
        self.lla = lla          # [lat, lon, alt]
        self.ecef = ecef
        self.xyz = xyz          # [x, y, z]
        self.rpy = rpy          # [roll, pitch, yaw]
        self.R_ned2e = R_ned2e  # rotation matrix
        self.std = std or {}    # {"lat": ..., "lon": ..., "r": ..., "p": ..., "y": ...}
        self.ENH = ENH if ENH is not None else []    # [E, N, H]
        self.R_ned2b = R_ned2b 
        self.opk = opk or None  # [opk]

def log(msg: str, *, verbose: bool = False, force: bool = False):
    if verbose or force:
        print(msg, file=sys.stderr)
def loadSBET(path):
    """
    Decodes an APPLANIX SNV/SBET file.

    Parameters:
    - settings: path to SBET

    Returns:
    - data: numpy array of processed data

  Input record: 17xdouble=(136 bytes)
       0  time  			sec_of_week 
       1  latitude   		rad
       2  longitude  		rad
       3  altitude       meters
       4  x_wander_vel   m/s
       5  y_wander_vel   m/s
       6  z_wander_vel  	m/s
       7  roll          	radians
       8  pitch         	radians
       9  wander_heading radians
       10 wander angle   radians
       11 x body accel   m/s^2
       12 y body accel   m/s^2
       13 z body accel   m/s^2
       14 x angular rate rad/s
       15 y angular rate rad/s
	   16 z angular rate rad/s					
 This is what is written in the ouput record:
       0   time            sec_of_week
       1   latitude        rad
       2   longitude       rad
       3   altitude        m
       4   roll            rad
       5   pitch           rad
       6  heading         rad 
    """

    try:
        with open(path, "rb") as f:
            print(f"Loading file {path}")
            data = np.fromfile(f, dtype=np.float64).reshape(-1,17)
    except Exception as e:
        errmsg = f"Cannot open file! {str(e)}"
        raise ValueError(errmsg)
        
    return data[:, 0], data[:, 1:4],  np.column_stack((data[:, 7:9], data[:, 9]-data[:, 10]))

def extract_pose_arrays(pose_list):
        t = np.array([p.t for p in pose_list])
        lla = np.array([p.lla for p in pose_list])
        xyz = np.array([p.xyz for p in pose_list])
        rpy = np.array([p.rpy for p in pose_list])
        R_ned2b = np.array([p.R_ned2b for p in pose_list])
        R_ned2e = np.array([p.R_ned2e for p in pose_list])
        ENH = np.array([p.ENH for p in pose_list])
        std = np.array([p.std for p in pose_list])

        return t, lla, xyz, rpy, R_ned2b, R_ned2e, ENH, std

import csv

def convert_to_projected_coords(
    poses: list,
    epsg_code: int,
    height_ref: str = "ellipsoidal",
    lat_col: str = "lat_deg",
    lon_col: str = "lon_deg",
    h_col: str = "alt_m",
    verbose: bool = False
) -> list:
    """
    Convert geographic coordinates (lat/lon/height) to projected or geocentric coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns lat_deg, lon_deg, alt_m.
    epsg_code : int
        EPSG code for target coordinate reference system (e.g., 25833 for ETRS89 / UTM33N).
    height_ref : str
        Height reference ('ellipsoidal', 'orthometric', 'geoid-based'). Currently only used for labeling.
    lat_col, lon_col, h_col : str
        Column names for latitude, longitude, and height.
    verbose : bool
        Print information about the transformation.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of input DataFrame with added columns:
        - x_m, y_m, z_m for geocentric
        - E_m, N_m, H_m for projected
    """
    # Source CRS: WGS84 / ETRS89 geographic
    crs_source = CRS.from_epsg(4979)  # WGS84 3D

    # Target CRS
    crs_target = CRS.from_epsg(epsg_code)
    if verbose:
        log(f"Transforming from {crs_source.to_authority()} to {crs_target.to_authority()}", force=True)

    transformer = Transformer.from_crs(crs_source, crs_target, always_xy=False)

    # Perform transformation
    lla = np.array([p.lla for p in poses])
    

    try:
        X, Y, Z = transformer.transform(lla[:, 0], lla[:, 1], lla[:, 2], radians=False)
    except Exception as e:
        log(f"[ERROR] Coordinate transformation failed: {e}", force=True)
        raise
    # Vectorized assignment of ENH coordinates
    ENH = np.column_stack((X, Y, Z))
    for i, pose in enumerate(poses):
        pose.ENH = tuple(ENH[i])
    
    if crs_target.is_geocentric:
        if verbose:
            log("Converted to geocentric (ECEF-like) coordinates", force=True)
    else:
        if verbose:
            log(f"Converted to projected coordinates in EPSG:{epsg_code}", force=True)
    return poses

def write_pose_list_to_csv(timestamp_img_name,poses, filename):
    """
    Writes a list of Pose_std objects to a CSV file with specific precision:
      - time: 6 digits
      - lat, lon: 14 digits
      - all other numeric fields: 3 digits
    """
    
    # Define column names (adjust as needed)
    fieldnames = ["filename",
        "t", "lat", "lon", "alt",
        "roll", "pitch", "yaw",
        "std_lat", "std_lon", 
        "std_r", "std_p", "std_y",
        "E", "N", "H",
        #"R_ned2e", "R_ned2b",
        "x", "y", "z",
        "o", "phi", "k"
    ]
    
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for p in poses:
            row = {}
            
            # --- Time ---
            row["t"] = f"{p.t:.6f}"
            
            # --- LLA ---
            lat, lon, alt = p.lla
            row["lat"] = f"{lat:.14f}"
            row["lon"] = f"{lon:.14f}"
            row["alt"] = f"{alt:.3f}"

            # --- RPY ---
            roll, pitch, yaw = p.rpy
            row["roll"] = f"{np.rad2deg(roll):.3f}"
            row["pitch"] = f"{q_b2mnp.rad2deg(pitch):.3f}"
            row["yaw"] = f"{np.rad2deg(yaw):.3f}"

            # --- Std dictionary ---
            row["std_lat"] = f"{p.std.get('lat', 0):.3f}"
            row["std_lon"] = f"{p.std.get('lon', 0):.3f}"
            row["std_r"]   = f"{p.std.get('r', 0):.3f}"
            row["std_p"]   = f"{p.std.get('p', 0):.3f}"
            row["std_y"]   = f"{p.std.get('y', 0):.3f}"
            
            # --- ENH ---
            if p.ENH and len(p.ENH) == 3:
                E, N, H = p.ENH
                row["E"] = f"{E:.3f}"
                row["N"] = f"{N:.3f}"
                row["H"] = f"{H:.3f}"
            else:
                row["E"] = row["N"] = row["H"] = ""
            
            # --- XYZ ---
            x, y, z = p.xyz
            row["x"] = f"{x:.3f}"
            row["y"] = f"{y:.3f}"
            row["z"] = f"{z:.3f}"
            
        
            # --- Rotation matrices ---
            #row["R_ned2e"] = ";".join(f"{val:.3f}" for val in sum(p.R_ned2e, []))  # flatten 2D
            #row["R_ned2b"] = ";".join(f"{val:.3f}" for val in sum(p.R_ned2b, []))
            
            # --- OPK ---
            if p.opk.any() and len(p.opk) == 3:
                o,phi,k = p.opk
                row["o"] = f"{o:.3f}"
                row["phi"] = f"{phi:.3f}"
                row["k"] = f"{k:.3f}"
            else:
                row["o"] = row["phi"] = row["k"] = ""
            
            writer.writerow(row)
def write_sbet_to_csv(trajectory, filename,step=1, stem=None,extension=None, type=['sbet','limatch','gps']):
    """
    Writes a list of Pose_std objects to a CSV file with specific precision:
      - time: 6 digits
      - lat, lon: 14 digits
      - all other numeric fields: 3 digits
    Limatch format
    Column 1: time, unit gps seconds of week
    Column 2-4: body frame position in ltp (epsg:), unit 3x meters
    Column 5-8: body frame orientation (epsg:), quaternion w, x, y, z
    """
    # replace filename with extension if provided
    if extension:
        filename = Path(filename).with_stem(stem).with_suffix(extension)
    
    with open(filename, "w", newline="") as csvfile:
        if type == 'sbet':
            fieldnames = [
                "t", "lat", "lon", "alt",
                "roll", "pitch", "yaw",
                "std_lat", "std_lon", 
                "std_r", "std_p", "std_y",
                "E", "N", "H",
                "x", "y", "z",
                "o", "phi", "k"
            ]
        elif type == 'limatch':
            fieldnames = [
                "t",
                "x", "y", "z",
                "qw", "qx", "qy", "qz"
            ]
        if type == 'gps':
            fieldnames = [
                "t", "lat", "lon", "alt"
            ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        trj = trajectory
        # Sample data every 200 entries of trajectory
        for i in range(0, len(trajectory.t), step):
            row = {}
            # --- Time ---
            row["t"] = f"{trj.t[i]:.6f}"
            
            if type == 'sbet':
                # --- LLA ---
                lat, lon, alt = trj.lla[i]
                row["lat"] = f"{lat:.14f}"
                row["lon"] = f"{lon:.14f}"
                row["alt"] = f"{alt:.3f}"
                
                # --- RPY ---
                roll, pitch, yaw = trj.rpy[i]
                row["roll"] = f"{np.rad2deg(roll):.3f}"
                row["pitch"] = f"{np.rad2deg(pitch):.3f}"
                row["yaw"] = f"{np.rad2deg(yaw):.3f}"
                
                # --- Std dictionary ---
                row["std_lat"] = f"{trj.std.get('lat', 0):.3f}"
                row["std_lon"] = f"{trj.std.get('lon', 0):.3f}"
                row["std_r"]   = f"{trj.std.get('r', 0):.3f}"
                row["std_p"]   = f"{trj.std.get('p', 0):.3f}"
                row["std_y"]   = f"{trj.std.get('y', 0):.3f}"
                
                # --- ENH ---
                if trj.ENH and len(trj.ENH) == 3:
                    E, N, H = q_m2btrj.ENH[i]
                    row["E"] = f"{E:.3f}"
                    row["N"] = f"{N:.3f}"
                    row["H"] = f"{H:.3f}"
                else:
                    row["E"] = row["N"] = row["H"] = ""
                
                # --- OPK ---
                if trj.opk.any() and len(trj.opk) == 3:
                    o, phi, k = trj.opk[i]
                    row["o"] = f"{o:.3f}"
                    row["phi"] = f"{phi:.3f}"
                    row["k"] = f"{k:.3f}"
                else:
                    row["o"] = row["phi"] = row["k"] = ""
                
                # --- XYZ ---
                x, y, z = trj.xyz[:, i]
                row["x"] = f"{x:.3f}"
                row["y"] = f"{y:.3f}"
                row["z"] = f"{z:.3f}"
                
            if type == 'limatch':
                # --- XYZ ---
                x, y, z = trj.xyz[:, i]
                row["x"] = f"{x:.3f}"
                row["y"] = f"{y:.3f}"
                row["z"] = f"{z:.3f}"
                
                # --- Quaternion m2b ---
                qw, qx, qy, qz = trj.q_b2m[i,:]
                row["qw"] = f"{qw:.6f}"
                row["qx"] = f"{qx:.6f}"
                row["qy"] = f"{qy:.6f}"
                row["qz"] = f"{qz:.6f}"
            if type == 'gps':
                # --- LLA ---
                lat, lon, alt = trj.lla[:,i]
                row["lat"] = f"{np.degrees(lat):.14f}"
                row["lon"] = f"{np.degrees(lon):.14f}"
                row["alt"] = f"{alt:.3f}"
            
            writer.writerow(row)



@dataclass
class ImageStamp:
    image_id: str
    image_filename: str
    timestamp_ns: int

def load_image_timestamps(exif_root: Path, exif_ext: str, image_exts: List[str], verbose: bool=False) -> List[ImageStamp]:
    stamps = []
    for root, _, files in os.walk(exif_root):
        for fname in files:
            if not fname.lower().endswith(exif_ext.lower()):
                continue
            p = Path(root) / fname
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith("#TIMESTAMP"):
                        rhs = line.split("=", 1)[-1].strip()
                        ns = int(rhs)
                        # Create image filename by replacing exif extension with .jpg
                        image_filename = p.stem + image_exts[0]
                        stamps.append(ImageStamp(p.stem, image_filename, ns))
                        break
    stamps.sort(key=lambda s: s.timestamp_ns)
    log(f"Loaded {len(stamps)} image timestamps", verbose=verbose, force=True)
    return stamps

def write_pose_with_images_to_csv(poses, images, filename,time_tolerance=0.0001,fieldnames=None,coord_type=None,verbose=False):
    """
    Writes a list of Pose_std objects to a CSV, matching each pose with an image by timestamp.
    
    Args:
        poses: list of Pose_std objects
        images: list of objects with attributes image_filename and time_stamp (ns)
        filename: output CSV file path
        time_tolerance: max time difference (sec) allowed to match an image (default 1 ms)
    """

    # Convert image timestamps from ns to seconds
    if coord_type == 'pospac_eo':
        log("Writing POSPAC EO format CSV", verbose=verbose, force=True)
        image_times = poses.t
        #image_files = [images[i]['image_filename'] for i in range(len(filename))]
    else:
        log("Writing default format CSV", verbose=verbose, force=True)     
        image_times = [img.timestamp_ns * 1e-9 for img in images]
    image_files = [img.image_filename for img in images]

    # Sort images by time (for efficient lookup)
    sorted_pairs = sorted(zip(image_times, image_files))
    image_times, image_files = zip(*sorted_pairs) if sorted_pairs else ([], [])

    # CSV columns
    if fieldnames is None:
        if coord_type is None:
            coord_type='default'
            fieldnames = [
                "image_filename",
                "t", "lat", "lon", "alt",
                "std_lat", "std_lon", "std_alt",
                "E", "N", "H",
            "x", "y", "z",
            "roll", "pitch", "yaw",
            "std_r", "std_p", "std_y",
            "o", "phi", "k"
        ]
        elif coord_type=='pospac_eo':
            '''
            'time': time_s,
                    'easting': easting,
                    'northing': northing,
                    'ellip_height': ellip_height,
                    'omega': omega,
                    'phi': phi,
                    'kappa': kappa,
                    'lat': lat,
                    'lon': lon
            '''
            fieldnames = [
                "image_filename",
                "t", "lat", "lon", "ellip_height",
                "easting", "northing", "omega", "phi", "kappa"
            ]
            

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for p in poses:
            row = {}

            # --- Find matching image ---
            if image_times:
                idx = bisect.bisect_left(image_times, p.t)
                closest_idx = None
                min_diff = float("inf")
                for i in [idx - 1, idx, idx + 1]:
                    if 0 <= i < len(image_times):
                        diff = abs(image_times[i] - p.t)
                        if diff < min_diff:
                            min_diff = diff
                            closest_idx = i
                if min_diff <= time_tolerance and closest_idx is not None:
                    row["image_filename"] = image_files[closest_idx]
                else:
                    row["image_filename"] = ""
            else:
                row["image_filename"] = ""

            for field in fieldnames:
                if field not in row:
                    row[field] = ""
            # --- Time ---
            if "t" in fieldnames:
                row["t"] = f"{p.t:.6f}"

            # --- LLA ---
            if poses.lla and len(poses.lla) == 3:
                lat, lon, alt = poses.lla
            elif poses[p].lat is not None and poses[p].lon is not None and poses[p].alt is not None:
                lat, lon, alt = poses[p].lat, poses[p].lon, poses[p].alt
            row["lat"] = f"{lat:.14f}"
            row["lon"] = f"{lon:.14f}"
            row["alt"] = f"{alt:.3f}"
            if p.std:
                row["std_lat"] = f"{p.std.get('lat', 0):.3f}"
                row["std_lon"] = f"{p.std.get('lon', 0):.3f}"
                row["std_alt"] = f"{p.std.get('alt', 0):.3f}"
            
            # --- ENH ---
            if p.ENH and len(p.ENH) == 3:
                E, N, H = p.ENH
                row["E"] = f"{E:.3f}"
                row["N"] = f"{N:.3f}"
                row["H"] = f"{H:.3f}"
            elif "easting" in fieldnames and "northing" in fieldnames and "ellip_height" in fieldnames:
                row["easting"] = row["northing"] = row["ellip_height"] = p.easting, p.northing, p.ellip_height

            # --- XYZ ---
            if "x" in fieldnames and "y" in fieldnames and "z" in fieldnames:
                x, y, z = p.xyz
                row["x"] = f"{x:.3f}"
                row["y"] = f"{y:.3f}"
                row["z"] = f"{z:.3f}"
            # --- RPY ---
            if "roll" in fieldnames and "pitch" in fieldnames and "yaw" in fieldnames:
                roll, pitch, yaw = p.rpy
                row["roll"] = f"{roll:.3f}"
                row["pitch"] = f"{pitch:.3f}"
                row["yaw"] = f"{yaw:.3f}"
            # --- Std dictionary ---
            if "std_r" in fieldnames and "std_p" in fieldnames and "std_y" in fieldnames:
                row["std_r"]   = f"{p.std.get('r', 0):.3f}"
                row["std_p"]   = f"{p.std.get('p', 0):.3f}"
                row["std_y"]   = f"{p.std.get('y', 0):.3f}"

            # --- OPK ---
            if p.opk.any() and len(p.opk) == 3:
                row["o"] = f"{p.opk[0]:.3f}"
                row["phi"] = f"{p.opk[1]:.3f}"
                row["k"] = f"{p.opk[2]:.3f}"
            elif "omega" in fieldnames and "phi" in fieldnames and "kappa" in fieldnames:
                row["o"] = row["phi"] = row["k"] = p.omega, p.phi, p.kappa

            writer.writerow(row)
def load_image_timestamps(exif_root: Path, exif_ext: str, image_exts: List[str], verbose: bool=False) -> List[ImageStamp]:
    stamps = []
    for root, _, files in os.walk(exif_root):
        for fname in files:
            if not fname.lower().endswith(exif_ext.lower()):
                continue
            p = Path(root) / fname
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith("#TIMESTAMP"):
                        rhs = line.split("=", 1)[-1].strip()
                        ns = int(rhs)
                        # Create image filename by replacing exif extension with .jpg
                        image_filename = p.stem + image_exts[0]
                        stamps.append(ImageStamp(p.stem, image_filename, ns))
                        break
    stamps.sort(key=lambda s: s.timestamp_ns)
    log(f"Loaded {len(stamps)} image timestamps", verbose=verbose, force=True)
    return stamps
