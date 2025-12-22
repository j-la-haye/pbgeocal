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
# Package imports
from photogrammetry_verify.camera import CameraModel
from photogrammetry_verify.transforms import Transform
from photogrammetry_verify.geometry import world_to_image
from photogrammetry_verify.geotools import GeoConverter
from liblibor.map import TangentPlane, Trajectory,log
from liblibor.rotations import *
from photogrammetry_verify.io_utils import *
from pyproj import CRS, Transformer

def run_bingo_verification(config_path):
    # 1. Load Configuration & Static Data
    cfg = load_config(config_path)
    
    # Load 3D Ground Truth
    df_3d = load_3d_csv(cfg['project']['point_cloud_csv'])
    
    # Load Timing (Image ID -> Time)
    timing_map = load_timing_file(cfg['project']['timing_csv'])

    # Define time span of images
    img_times = np.array([timing_map[img_id]  for img_id in timing_map])
    time_buffer = 3
    img_time_span = [img_times.min()-time_buffer, img_times.max()+time_buffer]

    
    log("[2/3] Loading SBET data...", verbose=True, force=True)
    # Extract time, lla, rpy from sbet_df
    t,lla,rpy,_ = read_sbet(Path(cfg['project']['trajectory']))
    
    mask = (t >= img_time_span[0]) & (t <= img_time_span[1])
    #tspan = t[mask]
    img_lla = lla[mask,:]
    #rpy = rpy[mask,:]
    lat0 = np.degrees(img_lla[0,0])
    lon0 = np.degrees(img_lla[0,1])
    alt0 = img_lla[0,2]
    print(f"Reference LTP origin: lat: {lat0:.6f} lon: {lon0:.6f} alt: {alt0:.3f}")
    tangentPlane = TangentPlane(lat0, lon0,alt0)
    

    #t,lla,rpy = loadSBET(sbet_path)
    trajectory_ltp = Trajectory(t, lla, rpy, tangentPlane, img_time_span)
    
    log("[3/3] Interpolating poses...", verbose=True, force=True)
    # Create coordinate transformer
    
    img_poses = trajectory_ltp.interpolate(img_times, cfg)
    
    
    # Initialize Trajectory Interpolator
    #print("Initializing Trajectory Interpolator...")
    #traj_interp = TrajectoryInterpolator(cfg['project']['trajectory_csv'])

    # Initialize Camera & Mount
    camera = CameraModel.from_dict(cfg['camera'])
    T_body_cam = Transform(
        cfg['mount']['lever_arm'], 
        cfg['mount']['rotation_euler']
    )
    
    # Initialize Geo Tools
    geo = GeoConverter(cfg['project']['epsg'])

    # 2. Parse BINGO Correspondence File
    print(f"Parsing BINGO file: {cfg['project']['bingo_file']}...")
    bingo_data = parse_bingo_file(cfg['project']['bingo_file'])
    print(f"Found {len(bingo_data)} image blocks.")

    global_errors = []

    #Filter BINGO data to only tiepoint_id < 1000 corresponding to 3D Checkpoints
    bingo_data = [block for block in bingo_data if (int(block['points']['tiepoint_id'].iloc[0]) < 1000)]

    # 3. Process Each Image Block
    for img_block in bingo_data:
        img_id = img_block['img_id']
        
        # A. Get Timestamp
        if img_id not in timing_map:
            print(f"Warning: Image4329364 ID {img_id} not found in timing file. Skipping.")
            continue
        timestamp = timing_map[img_id]

        # B. Interpolate Pose (Linear Pos + SLERP Rot)
        try:
            # select pose at timestamp from img_poses
            #pose_idx = np.where(img_times == timestamp)[0]
            pose_idx = np.where([pose.t == timestamp for pose in img_poses])[0]
            if len(pose_idx) == 0:
                raise ValueError(f"Timestamp {timestamp} not found in interpolated poses")
            pose = img_poses[pose_idx[0]]
            #img_poses = trajectory_ltp.interpolate(np.array([timestamp]), cfg)
            #pos_interp, rot_interp = traj_interp.get_pose_at_time(timestamp)
        except ValueError:
            print(f"Skipping {img_id}: Time {timestamp} out of trajectory bounds.")
            continue
            
        # Create NED -> Body Transform for this specific time
        # Convert Scipy Rotation to Euler for our Transform class, or modify Transform to accept Rotation object
        # Here we just re-wrap it:
        from scipy.spatial.transform import Rotation as R
        T_ned_body = Transform([0,0,0], [0,0,0]) # Placeholder init
        T_ned_body.t = pose.ENH
        T_ned_body.r = R.from_matrix(pose.R_ned2b)

        # C. Filter 3D points
        # Match BINGO tiepoint_ids with 3D CSV ids
        df_obs = img_block['points']
        # Join on 'tiepoint_id' (obs) vs index (3D)
        valid_obs = df_obs[df_obs['tiepoint_id'].astype(int).isin(df_3d.index)]
        
        if valid_obs.empty:
            continue

        # Extract Arrays
        tie_ids = valid_obs['tiepoint_id'].astype(int).values
        # BINGO Coords: U (Right), V (Up/Down?) relative to Principal Point
        bingo_u = valid_obs['u_bingo'].values
        bingo_v = valid_obs['v_bingo'].values
        
        # D. Convert BINGO Coords to Image Pixels (Top-Left Origin)
        # Assumption: BINGO U is X-Right, BINGO V is Y-Up (standard Cartesian)
        # Pixel U = cx + Bingo_U
        # Pixel V = cy - Bingo_V
        obs_u_px = camera.K[0, 2] + bingo_u
        obs_v_px = camera.K[1, 2] - bingo_v 
        obs_px = np.stack([obs_u_px, obs_v_px], axis=1)

        xyz_3d = df_3d.loc[tie_ids][['x', 'y', 'z']].values

        # 1. Skip the geo.get_local_enu_points logic entirely
        # Just use the Swiss coordinates directly from the CSV
        xyz_world = df_3d.loc[tie_ids][['x', 'y', 'z']].values 

        # 2. Define the Camera Pose in Swiss Grid
        # T_world_body.t is the Swiss ENH of the platform
        T_world_body = Transform(pose.ENH, [0,0,0]) 
        T_world_body.r = R.from_matrix(pose.R_ned2b)

       
        # Localize ENU around the drone's current position
        platform_pos_global = np.array([pose.ENH])  # lat, lon, alt
        
        # Note: If trajectory CSV is already Local NED, skip geo conversion. 
        # Assuming Trajectory CSV is Geo (Lat/Lon/Alt) or ECEF? 
        # *CRITICAL*: The 'pos_interp' must match the units of 'geo.to_ecef'.
        #lon, lat, _ = geo.to_wgs84_transformer.transform(pose.ENH[0], pose.ENH[1], pose.ENH[2])
        # Assuming Trajectory CSV contains Lat/Lon/Alt for this example:
        
        lon, lat, _ = geo.to_wgs84_transformer.transform(pose.ENH[0], pose.ENH[1], pose.ENH[2])
        platform_ecef = geo.to_ecef(platform_pos_global.reshape(1,3))[0]

        # Convert 3D GCPs to Local ENU
        world_ecef = geo.to_ecef(xyz_3d)
        world_enu = geo.get_local_enu_points(world_ecef, platform_ecef, (lon, lat))
        
        # ENU -> NED
        world_ned = world_enu @ T_enu_ned() #[:, [1, 0, 2]]
        #world_ned[:, 2] *= -1

        # Project
        #proj_uv, _ = world_to_image(world_ned, T_ned_body, T_body_cam, camera)

         # 3. Project
        # geometry.world_to_image will handle (P_world - T_world_body.t)
        proj_uv, _ = world_to_image(xyz_world, T_world_body, T_body_cam, camera)
                # E. Rigorous Projection (Geo -> ENU -> NED -> Cam)
        
        
        # Errors
        residuals = obs_px - proj_uv
        errors = np.linalg.norm(residuals, axis=1)
        
        print(f"Image {img_id} (t={timestamp:.4f}): {len(errors)} pts, RMSE={np.sqrt(np.mean(errors**2)):.2f} px")
        global_errors.extend(errors)

    print(f"\nTotal RMSE: {np.sqrt(np.mean(np.array(global_errors)**2)):.4f} px")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    config_path = "photogrammetry_verify/config_tie_points.yaml"
    #args = parser.parse_args()
    #args.config = config_path if args.config is None else args.config
    run_bingo_verification(config_path)