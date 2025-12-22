import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Package imports
from photogrammetry_verify.camera import CameraModel
from photogrammetry_verify.transforms import Transform
from photogrammetry_verify.geometry import world_to_image
from photogrammetry_verify.geotools import GeoConverter, TangentPlane, Trajectory,log
from photogrammetry_verify.io_utils import *

def run_automated_verification(config_path):
    # 1. Load Global Settings
    cfg = load_config(config_path)
    sbet_path = Path(cfg["sbet_path"])

    # 2. Load Data Tables
    # Expected columns:
    # 3d_coords: id, x, y, z
    # tie_points: id,t, u, v, frame_id
    # trajectory: frame_id, x, y, z, roll, pitch, yaw
    df_3d = pd.read_csv(cfg['project']['point_cloud_csv']).set_index('id')
    df_obs = pd.read_csv(cfg['project']['observations_csv'])
    df_traj = pd.read_csv(cfg['project']['trajectory_csv']).set_index('frame_id')
    
    
    # Define time span of images
    img_times = np.array([s.timestamp_ns / 1e9 for s in stamps])
    time_buffer = 3
    img_time_span = [img_times.min()-time_buffer, img_times.max()+time_buffer]

    log("[2/3] Loading SBET data...", verbose=True, force=True)
    # Extract time, lla, rpy from sbet_df
    t,lla,rpy,_ = read_sbet(sbet_path)
    
    mask = (t >= img_time_span[0]) & (t <= img_time_span[1])
    #tspan = t[mask]
    img_lla = lla[mask,:]
    #rpy = rpy[mask,:]
    lat0 = np.degrees(img_lla[0,0])
    lon0 = np.degrees(img_lla[0,1])
    alt0 = img_lla[0,2]
    print(f"Reference LTP origin: lat: {lat0:.6f} lon: {lon0:.6f} alt: {alt0:.3f}")
    tangentPlane = TangentPlane(lat0, lon0,0)
    

    #t,lla,rpy = loadSBET(sbet_path)
    trajectory_ltp = Trajectory(t, lla, rpy, tangentPlane, img_time_span)
    
    log("[3/3] Interpolating poses...", verbose=True, force=True)
    # Create coordinate transformer
        transformer = Transformer.from_crs(
            config.source_epsg,
            config.target_epsg,
        always_xy=True  # Ensures lon, lat order
        )
    img_poses = trajectory_ltp.interpolate(img_times)


    # 3. Setup Camera & Mount
    camera = CameraModel.from_dict(cfg['camera'])
    T_body_cam = Transform(
        cfg['mount']['lever_arm'], 
        cfg['mount']['rotation_euler']
    )
    
    geo = GeoConverter(cfg['project']['epsg'])
    
    unique_frames = df_obs['frame_id'].unique()
    print(f"Discovered {len(unique_frames)} frames in observation file.")

    all_residuals = []

    for f_id in unique_frames:
        if f_id not in df_traj.index:
            print(f"Skipping {f_id}: No trajectory data found.")
            continue
            
        # A. Localize Frame (Calculate local ENU/NED at drone position)
        traj_data = df_traj.loc[f_id]
        drone_pos = np.array([[traj_data['x'], traj_data['y'], traj_data['z']]])
        
        # Rigorous geodetic lookup for ENU rotation
        lon, lat, _ = geo.to_wgs84_transformer.transform(drone_pos[:,0], drone_pos[:,1], drone_pos[:,2])
        drone_ecef = geo.to_ecef(drone_pos)[0]
        
        # Trajectory Pose: NED -> Body
        T_ned_body = Transform([0,0,0], [traj_data['roll'], traj_data['pitch'], traj_data['yaw']])

        # B. Join Image observations with 3D points
        frame_obs = df_obs[df_obs['frame_id'] == f_id]
        joined = frame_obs.join(df_3d, on='id', how='inner', rsuffix='_3d')
        
        if joined.empty:
            continue

        ids = joined['id'].values
        uv_obs = joined[['u', 'v']].values
        xyz_3d = joined[['x', 'y', 'z']].values

        # C. Rigorous Coordinate Conversion (Global -> ECEF -> Local NED)
        world_ecef = geo.to_ecef(xyz_3d)
        world_enu = geo.get_local_enu_points(world_ecef, drone_ecef, (lon[0], lat[0]))
        
        # Map ENU [E, N, U] to NED [N, E, -U]
        world_ned = world_enu[:, [1, 0, 2]]
        world_ned[:, 2] *= -1

        # D. Project and Calculate Residuals
        uv_proj, _ = world_to_image(world_ned, T_ned_body, T_body_cam, camera)
        
        # Calculate residuals (Observed - Projected)
        res_uv = uv_obs - uv_proj
        errors = np.linalg.norm(res_uv, axis=1)

        # E. Store for Export
        for i in range(len(ids)):
            all_residuals.append({
                'frame_id': f_id,
                'point_id': ids[i],
                'u_obs': uv_obs[i, 0],
                'v_obs': uv_obs[i, 1],
                'u_proj': uv_proj[i, 0],
                'v_proj': uv_proj[i, 1],
                'u_res': res_uv[i, 0],
                'v_res': res_uv[i, 1],
                'error_mag': errors[i]
            })

    # 6. Global Summary and Export
    if all_residuals:
        df_results = pd.DataFrame(all_residuals)
        rmse = np.sqrt(np.mean(np.square(df_results['error_mag'])))
        
        # Save results
        output_path = Path(config_path).parent / "residuals_report.csv"
        df_results.to_csv(output_path, index=False)
        
        print("\n" + "="*50)
        print(f"VERIFICATION COMPLETE")
        print(f"Global RMSE: {rmse:.4f} pixels")
        print(f"Report saved to: {output_path}")
        print("="*50)
        
        # Quick diagnostic: are errors systematic?
        mean_u_res = df_results['u_res'].mean()
        mean_v_res = df_results['v_res'].mean()
        print(f"Mean Residuals: U={mean_u_res:.2f}, V={mean_v_res:.2f}")
        if abs(mean_u_res) > 1.0 or abs(mean_v_res) > 1.0:
            print(">> ADVICE: Significant mean shift detected. Check your Boresight/Mount rotation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="Path to config.yaml")
    config_path = "config_tie_points.yaml"
    args = parser.parse_args()
    run_automated_verification(args.config if args.config else config_path)