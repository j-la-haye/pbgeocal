import pandas as pd
import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
import csv
import pyproj

@dataclass
class BoresightCalibration:
    """Boresight angles and lever arm between IMU and camera."""
    roll: float = 0.0   # degrees
    pitch: float = 0.0  # degrees
    yaw: float = -90.0  # degrees (typical for nadir camera)
    
    # Lever arm: offset from IMU to camera in IMU body frame (meters)
    lever_arm_x: float = 0.0  # forward (along IMU X-axis)
    lever_arm_y: float = 0.0  # right (along IMU Y-axis)
    lever_arm_z: float = 0.0  # down (along IMU Z-axis)


@dataclass
class SBETRecord:
    """Single SBET trajectory record."""
    image_name: str
    time: float  # seconds (since week start or day start)
    lat: float  # degrees
    lon: float  # degrees
    alt: float  # meters above ellipsoid
    roll: float  # degrees
    pitch: float  # degrees
    yaw: float  # degrees (heading)
    

@dataclass
class SBET_Bin_Record:
    """Single Binary SBET trajectory record."""
    time: float  # seconds (since week start or day start)
    lat: float  # degrees
    lon: float  # degrees
    alt: float  # meters above ellipsoid
    roll: float  # degrees
    pitch: float  # degrees
    yaw: float  # degrees (heading)
@dataclass
class CameraPose:
    """Camera pose in projected coordinate system."""
    time: float  # seconds
    E: float  # easting or local X (meters)
    N: float  # northing or local Y (meters)
    H: float  # elevation or local Z (meters)
    x: float  # local X (meters)
    y: float  # local Y (meters)
    z: float  # local Z (meters)
    lat: float  # degrees
    lon: float  # degrees
    alt: float  # meters
    omega: float  # degrees
    phi: float  # degrees
    kappa: float  # degrees
    roll: float  # degrees
    pitch: float  # degrees
    heading: float  # degrees
    image_name: str = None

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

class Config:
    """Configuration for SBET conversion."""
    
    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.sbet_path = config['sbet_path']
        self.output_path = config['img_eo_path']
        self.source_epsg = config.get('source_epsg', 'EPSG:4326')  # WGS84 default
        self.target_epsg = config['target_epsg']  # e.g., 'EPSG:25832' for UTM32N
        
        # Boresight calibration
        boresight = config.get('boresight', {})
        self.boresight = BoresightCalibration(
            roll=boresight.get('roll', 0.0),
            pitch=boresight.get('pitch', 0.0),
            yaw=boresight.get('yaw', -90.0),
            lever_arm_x=boresight.get('lever_arm_x', 0.0),
            lever_arm_y=boresight.get('lever_arm_y', 0.0),
            lever_arm_z=boresight.get('lever_arm_z', 0.0)
        )
        
        # Get central meridian for grid convergence calculation
        self.central_meridian = self._get_central_meridian(self.target_epsg)
    
    @staticmethod
    def _get_central_meridian(epsg_code: str) -> float:
        """Get central meridian for UTM zone from EPSG code."""
        # Extract zone number from common UTM EPSG codes
        epsg_to_cm = {
            'EPSG:25832': 9.0,   # UTM32N ETRS89
            'EPSG:25833': 15.0,  # UTM33N ETRS89
            'EPSG:32632': 9.0,   # UTM32N WGS84
            'EPSG:32633': 15.0,  # UTM33N WGS84
            'EPSG:32631': 3.0,   # UTM31N WGS84
            'EPSG:25831': 3.0,   # UTM31N ETRS89
        }
        
        if epsg_code in epsg_to_cm:
            return epsg_to_cm[epsg_code]
        
        # Try to extract from zone number
        try:
            # Most UTM EPSG codes follow pattern: 326XX or 258XX where XX is zone
            epsg_num = int(epsg_code.split(':')[1])
            zone = epsg_num % 100
            return -183.0 + zone * 6.0
        except:
            raise ValueError(f"Cannot determine central meridian for {epsg_code}. "
                           f"Please add it to the mapping or specify manually.")

def load_csv_points(csv_path):
    """
    Expects CSV with columns: id, x, y, z
    (x, y, z can be Long/Lat/Alt or Easting/Northing/Height)
    """
    df = pd.read_csv(csv_path)
    return df.set_index('id')

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def read_sbet(file_path: str) -> List[SBETRecord]:
    """
    Read SBET data from CSV OR BIN file.
    
    Expected CSV format:
    image_name,lat,lon,alt,roll,pitch,yaw
    IMG_001.jpg,46.5,7.5,500.0,0.1,-0.2,45.3
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file containing SBET data
    
    Returns:
    --------
    List[SBETRecord]
        List of SBET records
    """

    """
    Decodes an APPLANIX SNV/SBET file.

    Parameters:
    - settings: path to SBET

    Bin Returns:
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
    records = []

    if file_path.name.lower().endswith('.out'):
        try:
            with open(file_path, "rb") as f:
                print(f"Loading file {file_path}")
                reader = np.fromfile(f, dtype=np.float64).reshape(-1,17)
                for row in reader:
                    records.append(SBET_Bin_Record(
                        time=float(row[0]%86400),  # sec_of_day
                        lat=float(row[1]),
                        lon=float(row[2]),
                        alt=float(row[3]),
                        roll=float(row[7]),
                        pitch=float(row[8]),
                        yaw=float(row[9])
                ))
        except Exception as e:
            errmsg = f"Cannot open file! {str(e)}"
            raise ValueError(errmsg)
        if (reader[:,0] > 86400).any():
            #convert to sec of day
            reader[:,0] = reader[:,0] % 86400
        
        return reader[:, 0], reader[:, 1:4],  np.column_stack((reader[:, 7:9], reader[:, 9]-reader[:, 10])),records
    
    elif file_path.lower().endswith('.csv'):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(SBETRecord(
                    image_name=row['image_name'],
                    time=float(row['time']),
                    lat=float(row['lat']),
                    lon=float(row['lon']),
                    alt=float(row['alt']),
                    roll=float(row['roll']),
                    pitch=float(row['pitch']),
                    yaw=float(row['yaw'])
                ))
        return records
def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_3d_csv(csv_path):
    """Expects columns: id, x, y, z"""
    return pd.read_csv(csv_path, names=['id', 'x', 'y', 'z']).set_index('id')

def load_timing_file(csv_path):
    """
    Expects CSV with columns: id, time
    Returns a dictionary mapping img_id -> timestamp
    """
    df = pd.read_csv(csv_path, names=['id', 'time'], dtype={'id': str},delimiter=' ')
    return pd.Series(df.time.values, index=df.id).to_dict()

def load_av4_timing(img_path):
    """
    Expects CSV with columns: id, time
    Returns a dictionary mapping img_id -> timestamp
    """
    # for all subdirs in img_path, using glob recursive search for file with .timing and sort and concatenate to return single dataframe
    import glob
    import os       
    timing_files = glob.glob(os.path.join(img_path, '**', '*.timing'), recursive=True)
    
    for timing_file in timing_files:
        # append all timing files into a single dataframe
        df_list = []
        for timing_file in timing_files:
            df_temp = pd.read_csv(timing_file, names=['time'], dtype={'time': float},delimiter=' ')
            df_list.append(df_temp)
    df = pd.concat(df_list, ignore_index=True)
        #timing_dict.update(pd.Series(df.time.values, index=df.id).to_dict())
    # ensure times are sorted in ascending order
    times = df['time'].values
    times.sort()
    return times


def parse_bingo_file(filepath):
    """
    Parses BINGO format correspondences.
    Returns a list of dictionaries: [{'img_id': ..., 'points': df}, ...]
    """
    images_data = []
    current_img_id = None
    current_img_name = None
    current_points = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            parts = line.split()

            # Check for end of block
            if parts[0] == '-99':
                if current_img_id is not None:
                    # Save the accumulated block
                    df_pts = pd.DataFrame(current_points, columns=['tiepoint_id', 'u_bingo', 'v_bingo'])
                    images_data.append({
                        'img_id': current_img_id,
                        'img_name': current_img_name,
                        'points': df_pts
                    })
                # Reset
                current_img_id = None
                current_points = []
                continue

            # Identify if line is Header (img_id img_name) or Data (tie_id U V)
            # Heuristic: Headers usually have non-numeric second column or look specific
            # BINGO usually puts integer ID first. 
            # If we are not in a block, this must be a header.
            if current_img_id is None:
                current_img_id = parts[0]
                # Contact rest as image name
                current_img_name = " ".join(parts[1:]) if len(parts) > 1 else ""
            else:
                # We are in a block, parse measurements
                # tiepoint_id U V
                if len(parts) >= 3:
                    tie_id = parts[0]
                    u = float(parts[1])
                    v = float(parts[2])
                    current_points.append((tie_id, u, v))
                    
    return images_data

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