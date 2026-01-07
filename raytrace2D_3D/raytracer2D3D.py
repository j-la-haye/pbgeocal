import numpy as np
import yaml
import csv
from pyproj import Transformer
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from photogrammetry_verify.io_utils import load_timing_file, parse_gcp_file,parse_bingo_file
from liblibor.map import TangentPlane, Trajectory,Pose_std,log, loadSBET
from pathlib import Path

# ==========================================
# 1. Camera Model (Optics)
# ==========================================
class CameraModel:
    def __init__(self, cfg):
        self.fx = cfg['focal_length_px']
        self.fy = cfg['focal_length_px']
        self.cx = cfg['principal_point'][0]
        self.cy = cfg['principal_point'][1]
        self.width = cfg['image_size'][0]
        self.height = cfg['image_size'][1]

        # Construct Intrinsic Matrix K
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ])
        self.K_inv = np.linalg.inv(self.K)

# ==========================================
# 2. Camera Projector (Physics & Geodesy)
# ==========================================
class CameraProjector:
    def __init__(self, camera_model: CameraModel, cfg_mount):
        self.cam = camera_model
        
        # Load mounting params
        self.lever_arm = np.array(cfg_mount['lever_arm'])
        
        # Build Rotation Matrices
        # 1. Boresight (Small corrections)
        br, bp, by = cfg_mount['boresight_rpy_rad']
        self.R_boresight = self._build_rotation_matrix(br, bp, by)
        
        # 2. Nominal Mount (Design)
        mr, mp, my = np.radians(cfg_mount['nominal_mount_rpy_deg'])
        self.R_mount = self._build_rotation_matrix(mr, mp, my)
        
        # Combined Body -> Camera rotation
        self.R_body_to_cam = self.R_mount @ self.R_boresight
        
        # Coordinate Transformer
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    def _build_rotation_matrix(self, r, p, y):
        """Z-Y-X Euler sequence."""
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        R_z = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        R_y = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        R_x = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        return R_x @ R_y @ R_z

    def get_pose_matrices(self, sbet_pose):
        """Computes Camera Center (ECEF) and Rotation (ECEF->NED->Body)."""
        # 1. IMU Position LLA -> ECEF
        imu_ecef = np.array(self.transformer.transform(
            np.degrees(sbet_pose.lla[1]), 
            np.degrees(sbet_pose.lla[0]), 
            sbet_pose.lla[2]
        ))
        
        # 2. ECEF -> NED Rotation
        lat, lon = sbet_pose.lla[0], sbet_pose.lla[1]
        clat, slat, clon, slon = np.cos(lat), np.sin(lat), np.cos(lon), np.sin(lon)
        R_ecef_to_ned = np.array([
            [-slat * clon, -slat * slon,  clat],
            [-slon,         clon,         0   ],
            [-clat * clon, -clat * slon, -slat]
        ])
        
        # 3. NED -> Body Rotation
        R_ned_to_body = self._build_rotation_matrix(
            sbet_pose.rpy[0], sbet_pose.rpy[1], sbet_pose.rpy[2]
        )
        
        return imu_ecef, R_ecef_to_ned, R_ned_to_body

    def get_ray_in_ecef(self, uv, sbet_pose):
        """Returns Camera Center and Unit Ray Vector in ECEF."""
        imu_ecef, R_ecef_to_ned, R_ned_to_body = self.get_pose_matrices(sbet_pose)
        
        # Chain: ECEF -> NED -> Body -> Camera
        # Inverse: Camera -> Body -> NED -> ECEF
        R_body_to_ecef = R_ecef_to_ned.T @ R_ned_to_body.T
        
        # Camera Center = IMU + Rotated Lever Arm
        cam_center_ecef = imu_ecef + (R_body_to_ecef @ self.lever_arm)
        
        # Ray in Camera Frame
        uv_homog = np.array([uv[0], uv[1], 1.0])
        ray_cam = self.cam.K_inv @ uv_homog
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Ray in ECEF
        # ray_ecef = R_body_to_ecef * R_cam_to_body * ray_cam
        # Note: R_cam_to_body is Transpose of R_body_to_cam
        ray_body = self.R_body_to_cam.T @ ray_cam
        ray_ecef = R_body_to_ecef @ ray_body
        
        return cam_center_ecef, ray_ecef

    def project(self, pt_ecef, sbet_pose):
        """Projects ECEF point to (u,v)."""
        imu_ecef, R_ecef_to_ned, R_ned_to_body = self.get_pose_matrices(sbet_pose)
        
        # Vector IMU -> Point
        v_ecef = pt_ecef - imu_ecef
        
        # Rotation Chain: ECEF -> NED -> Body
        v_body = R_ned_to_body @ (R_ecef_to_ned @ v_ecef)
        
        # Apply Lever Arm & Mount
        v_camera = self.R_body_to_cam @ (v_body - self.lever_arm)
        
        if v_camera[2] <= 0: return None
        
        uv_homog = self.cam.K @ (v_camera / v_camera[2])
        return uv_homog[0], uv_homog[1]

# ==========================================
# 3. Utilities (Math & Parsing)
# ==========================================
def triangulate_n_views(centers, rays):
    """Least Squares Intersection of N lines in 3D."""
    mat_sum = np.zeros((3, 3))
    vec_sum = np.zeros(3)
    
    for C, v in zip(centers, rays):
        # I - v*vT (Projection onto subspace orthogonal to v)
        P_orth = np.eye(3) - np.outer(v, v)
        mat_sum += P_orth
        vec_sum += P_orth @ C
        
    try:
        return np.linalg.solve(mat_sum, vec_sum)
    except np.linalg.LinAlgError:
        return None

def parse_sbet_csv(filepath, cfg_sbet):
    """Loads SBET text file into a dict of {image_id: pose_dict}."""
    poses = {}
    c = cfg_sbet
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=',') # Adjust delimiter if needed
        for line in reader:
            if not line: continue
            # Basic parsing - adjust indices based on file structure
            try:
                img_id = int(float(line[c['image_id_col']]))
                poses[img_id] = {
                    'lat': float(line[c['lat_col']]),
                    'lon': float(line[c['lon_col']]),
                    'alt': float(line[c['alt_col']]),
                    'roll': float(line[c['roll_col']]),
                    'pitch': float(line[c['pitch_col']]),
                    'yaw': float(line[c['yaw_col']]),
                }
            except ValueError:
                continue
    return poses

def parse_bingo(filepath):
    """Parses BINGO .dat file."""
    blocks = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == '-99':
            i += 1
            continue
        
        parts = line.split(None, 1)
        if len(parts) >= 2:
            try:
                gcp_id = int(parts[0])
                gcp_name =" ".join(parts[1:]) if len(parts) > 1 else "" #parts[1].strip()
                observations = []
                i += 1
                while i < len(lines):
                    obs_line = lines[i].strip()
                    if obs_line == '-99':
                        i += 1; break
                    if not obs_line:
                        i += 1; continue
                    
                    obs_parts = obs_line.split()
                    if len(obs_parts) >= 3:
                        observations.append((int(obs_parts[0]), float(obs_parts[1]), float(obs_parts[2])))
                    i += 1
                if observations:
                    blocks.append((gcp_id, gcp_name, observations))
            except ValueError:
                i += 1
        else:
            i += 1
    return blocks

# ==========================================
# 2. Parsers (UPDATED BINGO PARSER)
# ==========================================
def parse_bingo_inverted(filepath):
    """
    Parses BINGO file where blocks are by IMAGE.
    Groups observations by TIE POINT ID.
    
    Structure:
    IMG_ID IMG_NAME
    TP_ID  U  V
    -99
    
    Returns:
        List of (tp_id, tp_name, [(img_id, u, v), ...])
    """
    # Dictionary to group observations: { tp_id: [(img_id, u, v), ...] }
    grouped_points = defaultdict(list)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    current_img_id = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Check for End of Block
        if line == '-99':
            current_img_id = None # Reset
            i += 1
            continue
            
        parts = line.split()
        
        # Heuristic: Header lines usually have strings (names), Observation lines are numbers.
        # Header: "1 L4_UVT + 69_3_bldgs" (parts > 1, second part is string)
        # Observation: "61 431.97 4.65812" (parts == 3, all numbers)
        
        is_header = False
        try:
            # If line starts with int, and has >=2 parts, check if it's a header
            # Usually observation lines are exactly 3 floats/ints. 
            # Header lines have a Name string.
            if len(parts) >= 2:
                # Try parsing as observation first
                tp_id = int(parts[0])
                u = float(parts[1])
                
                # If we have 3 parts and they are all numbers, it's an observation
                if len(parts) == 3:
                     v = float(parts[2])
                     if current_img_id is not None:
                         grouped_points[tp_id].append((current_img_id, u, v))
                     else:
                         # We found an observation but we aren't inside a block? 
                         # This might happen if file format is slightly off or we misidentified header.
                         pass 
                else:
                     # If length != 3, or parsing failed, assume it is Header
                     is_header = True
        except ValueError:
            # If conversion to float fails, it contains text -> Must be Header
            is_header = True
            
        if is_header:
            try:
                current_img_id = int(parts[0])
                # We ignore the rest of the header (image name) for now
            except ValueError:
                pass # Malformed header?

        i += 1
        
    # Convert dict to list format for the pipeline
    # Format: [(gcp_id, gcp_name, observations), ...]
    # We generate a generic name since BINGO doesn't give names to tie points in this format
    output_blocks = []
    for tp_id, obs_list in grouped_points.items():
        output_blocks.append((tp_id, f"TP_{tp_id}", obs_list))
        
    return output_blocks

def parse_timing_file(filepath, fmt):
    """Returns dict: {image_id: timestamp}"""
    times = {}
    with open(filepath, 'r') as f:
        # Simple split handles both space and comma usually, or refine based on fmt
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if not parts: continue
            try:
                img_id = int(parts[fmt['image_id_col']])
                t = float(parts[fmt['time_col']])
                times[img_id] = t
            except ValueError:
                continue
    return times

# ==========================================
# 4. Main Execution Pipeline
# ==========================================
def main(config_path="config.yaml"):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print("--- Initializing Pipeline ---")
    
    # 2. Initialize Models
    cam_model = CameraModel(cfg['camera'])
    projector = CameraProjector(cam_model, cfg['mounting'])
    
    # 3. Load Data
    # -------------------------------------------------------------------------
    # STEP 3: Load GCP ECEF coordinates
    # -------------------------------------------------------------------------
    print("\n[3] Loading GCP ECEF coordinates...")
    gcp_path = cfg['paths']['gcp_file']
    gcp_dict = parse_gcp_file(gcp_path, epsg=cfg['project']['epsg'])
    print(f"    Loaded {len(gcp_dict)} GCP coordinates")

     # Load Timing (Image ID -> Time)
    timing_map = load_timing_file(cfg['paths']['timing_file'])

    # Define time span of images
    img_times = np.array([timing_map[img_id]  for img_id in timing_map])
    time_buffer = 3
    img_time_span = [img_times.min()-time_buffer, img_times.max()+time_buffer]
   
    t_start, t_end = img_time_span
    print(f"Trajectory time range: {t_start:.3f} to {t_end:.3f}")
    
    
    
    if cfg['project'].get('parse_sbet', True):
        log("[2/3] Loading SBET data...", verbose=True, force=True)
        # Extract time, lla, rpy from sbet_df
        #print(f"Loading SBET from {cfg['paths']['sbet_file']}...")
        t,lla,rpy = loadSBET(Path(cfg['paths']['sbet_file']))
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
        
        img_poses = trajectory.interpolate(img_times, cfg)
        # write poses to csv for debugging
        with open(cfg['paths']['poses_file'], "w") as f:
            f.write("time,lat,lon,alt,roll,pitch,yaw\n")
            for pose in img_poses:
                f.write(f"{pose.t},{pose.lla[0]},{pose.lla[1]},{pose.lla[2]},{pose.rpy[0]},{pose.rpy[1]},{pose.rpy[2]}\n")
        print(f"    Interpolated {len(img_poses)} image poses")
    else:
        # Load poses from defined csv path
        poses_csv = np.loadtxt(cfg['paths']['poses_file'], delimiter=',',skiprows=1)
        t,lla,rpy = poses_csv[:,0], poses_csv[:,1:4], poses_csv[:,4:7]
        mask = (t >= img_time_span[0]) & (t <= img_time_span[1])
        img_lla = lla[mask,:]
        lat0 = np.degrees(img_lla[0,0])
        lon0 = np.degrees(img_lla[0,1])
        alt0 = img_lla[0,2]
        print(f"Reference LTP origin: lat: {lat0:.6f} lon: {lon0:.6f} alt: {alt0:.3f}")
        tangentPlane = TangentPlane(lat0, lon0,alt0)
        poses = Trajectory(t, lla, rpy, tangentPlane, img_time_span)
        transformer = Transformer.from_crs(
            4326,
            cfg['project']['epsg'],
        always_xy=False  # Ensures lon, lat order
        )
        # Step 7: Convert camera position to projected coordinates
        # Use the actual trajectory length, not img_times length
        n_poses = poses.lla.shape[1]  # Number of poses in the trajectory
        E, N, H = transformer.transform(poses.lla[0,:], poses.lla[1,:], poses.lla[2,:],radians=True)
        ENH = np.array([E,N,H])
        img_poses = []
        for i in range(n_poses):
            img_poses.append(Pose_std(poses.t[i], poses.lla[:,i], poses.xyz[:,i], poses.rpy[:,i], poses.R_ned2ecef[i], poses.R_ned2body[i], poses.ecef[i,:], ENH[:,i]))

    print(f"Loading BINGO from {cfg['paths']['bingo_file']}...")
    bingo_data = parse_bingo_inverted(cfg['paths']['bingo_file'])

    #Filter BINGO data to only tiepoint_id < 1000 corresponding to 3D Checkpoints
    bingo_data = [block for block in bingo_data if (block[0] < 1000)]
    
    results = []
    # 4. Process Observations
    # Statistics accumulators
    total_3d_error = 0.0
    gcp_match_count = 0

    for gcp_id, gcp_name, observations in bingo_data:
        centers, rays, valid_obs = [], [], []

        # 1. Gather Rays
        for img_id, u, v in observations:
            # A. Get Time for Image
            # Try both int and str keys to handle type mismatch
            timestamp =  timing_map.get(str(img_id))
            if timestamp is None:
                continue

            # B. Interpolate Pose from SBET
            # Get pose for this image
            try:
            # select pose at timestamp from img_poses
                pose_idx = np.where([pose.t == timestamp for pose in img_poses])[0]
                if len(pose_idx) == 0:
                    raise ValueError(f"Timestamp {timestamp} not found in interpolated poses")
                pose = img_poses[pose_idx[0]]
        
            except ValueError:
                print(f"Skipping {img_id}: Time {timestamp} out of trajectory bounds.")
                continue
            
            if pose is None:
                print(f"Warning: Time {t} out of SBET bounds.")
                continue

            C, r = projector.get_ray_in_ecef((u, -v), pose)
            centers.append(C)
            rays.append(r)
            valid_obs.append({'u': u, 'v': -v, 'pose': pose})
        
        if len(centers) < 2:
            print(f"Skipping GCP {gcp_id}: Not enough views ({len(centers)})")
            continue
            
        # 2. Triangulate
        est_ecef = triangulate_n_views(centers, rays)
        if est_ecef is None: continue
        
        # 3. Calculate Reprojection Error (Pixel Error)
        px_errors = []
        for vo in valid_obs:
            proj_uv = projector.project(est_ecef, vo['pose'])
            if proj_uv:
                px_errors.append(np.linalg.norm(np.array([vo['u'], vo['v']]) - np.array(proj_uv)))
        rmse_px = np.sqrt(np.mean(np.array(px_errors)**2)) if px_errors else 0.0
        print(f"GCP {gcp_id}: Triangulated with {len(valid_obs)} views, RMSE: {rmse_px:.2f} px")
        # 4. Calculate 3D Error against Ground Truth (if exists)
        error_3d_m = -1.0 
        dx, dy, dz = 0.0, 0.0, 0.0
        
        if gcp_id in gcp_dict:
            true_ecef = np.array([gcp_dict[gcp_id].x, gcp_dict[gcp_id].y, gcp_dict[gcp_id].z])
            diff = est_ecef - true_ecef
            dx, dy, dz = diff
            error_3d_m = np.linalg.norm(diff)
            
            total_3d_error += error_3d_m
            gcp_match_count += 1
            
        # Store Result
        results.append([gcp_id, gcp_name, len(valid_obs), rmse_px, error_3d_m, dx, dy, dz, *est_ecef])

    # 5. Report
    print(f"\n--- Evaluation Report ---")
    print(f"{'TP ID':<8} {'Views':<6} {'RMSE(px)':<10} {'3D Err(m)':<10} {'dX':<8} {'dY':<8} {'dZ':<8}")
    print("-" * 75)
    
    # Print matches first
    matched_results = [r for r in results if r[4] != -1.0]
    for res in matched_results:
        print(f"{res[0]:<8} {res[2]:<6} {res[3]:<10.2f} {res[4]:<10.3f} {res[5]:<8.2f} {res[6]:<8.2f} {res[7]:<8.2f}")
    
    if gcp_match_count > 0:
        mean_3d_err = total_3d_error / gcp_match_count
        print("-" * 75)
        print(f"Mean 3D Error over {gcp_match_count} GCPs: {mean_3d_err:.3f} meters")
    else:
        print("\nNo Tie Points matched the provided GCP IDs.")

    # Save CSV
    try:
        out_path = cfg['paths']['output_report']
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['TiePoint_ID', 'Name', 'Views', 'RMSE_px', '3D_Error_m', 'dX', 'dY', 'dZ', 'Est_X', 'Est_Y', 'Est_Z'])
            writer.writerows(results)
        print(f"\nDetailed report saved to {out_path}")
    except KeyError:
        print("Output path not defined in config.")
if __name__ == "__main__":
    main("raytrace2D_3D/config_addlidar.yaml")
