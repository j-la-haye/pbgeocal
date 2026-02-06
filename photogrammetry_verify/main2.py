import argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from photogrammetry_verify.camera import CameraModel
from photogrammetry_verify.transforms import Transform
from photogrammetry_verify.geometry import world_to_image
from photogrammetry_verify.io_utils import load_config, load_3d_csv, load_timing_file, parse_bingo_file
from photogrammetry_verify.trajectory import TrajectoryInterpolator
from photogrammetry_verify.geotools import get_grid_convergence

def run_rigorous_verification(config_path):
    cfg = load_config(config_path)
    epsg = cfg['project']['epsg']
    
    # Load Data
    df_3d = load_3d_csv(cfg['project']['point_cloud_csv'])
    timing_map = load_timing_file(cfg['project']['timing_csv'])
    traj_interp = TrajectoryInterpolator(cfg['project']['trajectory_csv'])
    camera = CameraModel.from_dict(cfg['camera'])
    
    # BINGO parsing
    bingo_data = parse_bingo_file(cfg['project']['bingo_file'])
    
    all_results = []

    for img_block in bingo_data:
        img_id = img_block['img_id']
        if img_id not in timing_map: continue
        
        # 1. Interpolate Pose
        t = timing_map[img_id]
        pos_grid, rot_true_body = traj_interp.get_pose_at_time(t) # pos_grid = [E, N, H]
        
        # 2. General Grid Convergence
        # We need lon/lat to find convergence at this specific spot
        # (Assuming your interpolator can provide lon/lat or you back-project pos_grid)
        lon, lat = traj_interp.get_lon_lat_at_time(t) 
        gamma = get_grid_convergence(epsg, lon, lat)
        
        # 3. Create Rotation: Grid -> True -> Body
        R_grid_true = R.from_euler('z', gamma, degrees=True)
        R_grid_body = R_grid_true * rot_true_body
        
        # 4. Filter and Project
        df_obs = img_block['points']
        valid_obs = df_obs[df_obs['tiepoint_id'].isin(df_3d.index)]
        if valid_obs.empty: continue

        # Localize in Grid Frame
        xyz_world = df_3d.loc[valid_obs['tiepoint_id']][['x', 'y', 'z']].values
        p_rel_grid = xyz_world - pos_grid
        
        # Transform to Body
        p_body = R_grid_body.inv().apply(p_rel_grid) - np.array(cfg['mount']['lever_arm'])
        
        # Transform to Camera (Apply Boresight from Config)
        T_body_cam = R.from_euler(cfg['mount']['rotation_order'], 
                                  cfg['mount']['rotation_euler'], degrees=True)
        p_cam = T_body_cam.apply(p_body)
        
        # 5. Image Projection (BINGO U/V to Pixel U/V)
        # BINGO U is Right, V is UP. Pixel U is Right, V is DOWN.
        obs_u_px = camera.K[0, 2] + valid_obs['u_bingo'].values
        obs_v_px = camera.K[1, 2] - valid_obs['v_bingo'].values
        obs_px = np.stack([obs_u_px, obs_v_px], axis=1)

        # Final Projection using Pinhole Logic
        depths = p_cam[:, 2]
        proj_u = camera.K[0,0] * (p_cam[:,0]/depths) + camera.K[0,2]
        proj_v = camera.K[1,1] * (p_cam[:,1]/depths) + camera.K[1,2]
        proj_uv = np.stack([proj_u, proj_v], axis=1)

        # 6. Error Calculation
        err = np.linalg.norm(obs_px - proj_uv, axis=1)
        all_results.extend(err)
        print(f"Img {img_id}: Mean Err {np.mean(err):.2f}px | Depth: {np.mean(depths):.1f}m")

    print(f"\nFinal RMSE: {np.sqrt(np.mean(np.square(all_results))):.4f} px")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rigorous verification of photogrammetry data.")
    parser.add_argument("config_path", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    run_rigorous_verification(args.config_path)