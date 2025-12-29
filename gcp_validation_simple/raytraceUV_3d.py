import numpy as np
from pyproj import Transformer
from gcp_validation_script import parse_bingo

class CameraProjector:
    def __init__(self, K, lever_arm, nominal_mount_rpy_deg, boresight_rpy_rad):
        """
        Initializes the projector with static camera calibration and mounting parameters.
        
        Args:
            K: (3,3) Intrinsic matrix
            lever_arm: (3,) [x, y, z] offset from IMU to Camera in Body frame
            nominal_mount_rpy_deg: (3,) Nominal mount angles in degrees
            boresight_rpy_rad: (3,) Boresight correction angles in radians
        """
        self.K = np.array(K)
        # Ensure lever_arm is a column vector for broadcasting (3, 1)
        self.lever_arm = np.array(lever_arm).reshape(3, 1)
        
        # 1. Pre-calculate the static Body-to-Camera rotation
        self.R_boresight = self._build_rotation_matrix(*boresight_rpy_rad)
        
        m_r, m_p, m_y = np.radians(nominal_mount_rpy_deg)
        self.R_mount = self._build_rotation_matrix(m_r, m_p, m_y)
        
        # Total rotation from Body to Camera frame
        self.R_body_to_cam = self.R_mount @ self.R_boresight
        
        # Initialize the ECEF transformer
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    def _build_rotation_matrix(self, r, p, y):
        """Standard Z-Y-X (Yaw, Pitch, Roll) rotation matrix."""
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        return R_roll @ R_pitch @ R_yaw

    def get_ned_to_camera_params(self, sbet_pose):
        """Calculates transform matrices for a specific pose."""
        # 1. Camera Center in ECEF
        imu_ecef = np.array(self.transformer.transform(
            np.degrees(sbet_pose['lon']), 
            np.degrees(sbet_pose['lat']), 
            sbet_pose['alt']
        )).reshape(3, 1) # Column vector
        
        # 2. ECEF to NED Rotation
        lat, lon = sbet_pose['lat'], sbet_pose['lon']
        clat, slat, clon, slon = np.cos(lat), np.sin(lat), np.cos(lon), np.sin(lon)
        R_ecef_to_ned = np.array([
            [-slat * clon, -slat * slon,  clat],
            [-slon,         clon,         0   ],
            [-clat * clon, -clat * slon, -slat]
        ])
        
        # 3. NED to Body
        R_ned_to_body = self._build_rotation_matrix(
            sbet_pose['roll'], sbet_pose['pitch'], sbet_pose['yaw']
        )
        
        return imu_ecef, R_ecef_to_ned, R_ned_to_body

    def project(self, pt_ecef, sbet_pose):
        """Single point projection (wrapper for batch)."""
        # Reshape input to (1, 3) then batch project
        pt_arr = np.array(pt_ecef).reshape(1, 3)
        uv, valid = self.project_batch(pt_arr, sbet_pose)
        if valid[0]:
            return uv[0, 0], uv[0, 1]
        return None

    def project_batch(self, pts_ecef, sbet_pose, return_mask=False):
        """
        Projects a list of 3D ECEF points into the 2D image plane.

        Args:
            pts_ecef: (N, 3) array of ECEF points.
            sbet_pose: dict containing pose info.
            return_mask: If True, returns the validity mask (Z > 0).

        Returns:
            uv_coords: (N, 2) array of pixel coordinates. 
                       Points behind camera will have values, but are invalid.
            valid_mask: (N,) boolean array indicating which points are in front of camera.
        """
        # Ensure input is Transposed to (3, N) for matrix mult
        # pts_ecef input is (N, 3) -> Transpose to (3, N)
        P_ecef = np.array(pts_ecef).T 
        
        imu_ecef, R_ecef_to_ned, R_ned_to_body = self.get_ned_to_camera_params(sbet_pose)
        
        # 1. Vector from IMU to Points (Broadcasting imu_ecef across N columns)
        # (3, N) - (3, 1)
        v_ecef = P_ecef - imu_ecef
        
        # 2. Rotate ECEF -> NED -> Body
        # Combine rotations for efficiency: R_ecef_to_body = R_ned_to_body * R_ecef_to_ned
        R_ecef_to_body = R_ned_to_body @ R_ecef_to_ned
        
        # v_body = R * v_ecef
        v_body = R_ecef_to_body @ v_ecef
        
        # 3. Apply Lever Arm and Rotate to Camera Frame
        # v_body (3, N) - lever_arm (3, 1)
        v_body_shifted = v_body - self.lever_arm
        
        v_camera = self.R_body_to_cam @ v_body_shifted
        
        # 4. Perspective Projection
        # Extract Z component (Depth)
        Z = v_camera[2, :]
        
        # Create validity mask (Z > 0)
        valid_mask = Z > 0
        
        # Prepare Output Array (N, 2)
        uv_coords = np.zeros((P_ecef.shape[1], 2))
        
        # Avoid division by zero for invalid points by replacing 0/negative Z with 1 (temporary)
        # We will filter them out via the mask anyway.
        Z_safe = np.where(valid_mask, Z, 1.0)
        
        # Project: K @ (v_camera / Z)
        # v_camera is (3, N). We divide each row by Z_safe.
        v_normalized = v_camera / Z_safe
        
        # Homogeneous multiply: (3, 3) @ (3, N) -> (3, N)
        uv_homog = self.K @ v_normalized
        
        # Transpose result back to (N, 2) for standard output format
        uv_coords = uv_homog[:2, :].T
        
        if return_mask:
             return uv_coords, valid_mask
             
        # Optional: Set invalid points to NaN
        uv_coords[~valid_mask] = np.nan
        
        return uv_coords, valid_mask

def get_camera_to_world_transform(sbet_pose, lever_arm, boresight):
    """Computes Camera Center in ECEF and Rotation from Camera to ECEF."""
    # 1. IMU ECEF
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    imu_ecef = np.array(transformer.transform(np.degrees(sbet_pose['lon']), 
                                              np.degrees(sbet_pose['lat']), 
                                              sbet_pose['alt']))

    # 2. Rotation Matrices (Same as previous steps)
    lat, lon = sbet_pose['lat'], sbet_pose['lon']
    clat, slat, clon, slon = np.cos(lat), np.sin(lat), np.cos(lon), np.sin(lon)
    R_ecef_to_ned = np.array([
        [-slat * clon, -slat * slon,  clat],
        [-slon,         clon,         0   ],
        [-clat * clon, -clat * slon, -slat]
    ])
    
    # NED to Body
    r, p, y = sbet_pose['roll'], sbet_pose['pitch'], sbet_pose['yaw']
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    R_ned_to_body = R_roll @ R_pitch @ R_yaw
    
    # 3. Camera Center in ECEF
    # Position = IMU_ECEF + R_body_to_ecef * lever_arm
    R_body_to_ecef = R_ecef_to_ned.T @ R_ned_to_body.T
    cam_center_ecef = imu_ecef + R_body_to_ecef @ np.array(lever_arm)

    # 4. Camera Orientation to ECEF
    br, bp, by = boresight
    cbr, sbr, cbp, sbp, cby, sby = np.cos(br), np.sin(br), np.cos(bp), np.sin(bp), np.cos(by), np.sin(by)
    R_bore = (np.array([[1, 0, 0], [0, cbr, sbr], [0, -sbr, cbr]]) @ 
              np.array([[cbp, 0, -sbp], [0, 1, 0], [sbp, 0, cbp]]) @ 
              np.array([[cby, sby, 0], [-sby, cby, 0], [0, 0, 1]]))
    
    R_mount = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    
    # Total rotation: Camera -> Body -> NED -> ECEF
    R_cam_to_ecef = R_body_to_ecef @ R_bore.T @ R_mount.T
    
    return cam_center_ecef, R_cam_to_ecef

def intersect_rays(p1, v1, p2, v2):
    """Finds the mid-point of the shortest segment between two 3D lines."""
    # Line 1: p1 + s*v1 | Line 2: p2 + t*v2
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    n = np.cross(v1, v2)
    if np.linalg.norm(n) < 1e-7: return None # Parallel rays
    
    # Solve for s and t using linear algebra
    rhs = p2 - p1
    lhs = np.array([v1, -v2, n]).T
    s, t, dist_along_n = np.linalg.solve(lhs, rhs)
    
    pt1 = p1 + s * v1
    pt2 = p2 + t * v2
    intersection_point = (pt1 + pt2) / 2
    miss_distance = np.linalg.norm(pt1 - pt2)
    
    return intersection_point, miss_distance

def confirm_tie_point_3d(uv1, pose1, uv2, pose2, K, lever_arm, boresight):
    """
    Computes the 3D intersection and error for a pair of 2D tie points.
    """
    # Inverse Intrinsic
    K_inv = np.linalg.inv(K)
    
    # Get World Pose for both cameras
    c1, R1 = get_camera_to_world_transform(pose1, lever_arm, boresight)
    c2, R2 = get_camera_to_world_transform(pose2, lever_arm, boresight)
    
    # Back-project 2D to 3D Rays in ECEF
    ray1_cam = K_inv @ np.array([uv1[0], uv1[1], 1.0])
    ray2_cam = K_inv @ np.array([uv2[0], uv2[1], 1.0])
    
    ray1_ecef = R1 @ ray1_cam
    ray2_ecef = R2 @ ray2_cam
    
    # Intersect
    result = intersect_rays(c1, ray1_ecef, c2, ray2_ecef)
    if result is None: return None
    
    return {
        "ecef_pt": result[0],
        "miss_distance_m": result[1]
    }



# Assuming the CameraProjector class from the previous step exists.
# We add this method to it (or subclass it).

class RayCastingProjector(CameraProjector):
    def get_ray_in_ecef(self, uv, sbet_pose):
        """
        Calculates the Optical Center (C) and Unit Ray Direction (v) in ECEF.
        """
        # 1. Get Camera Center and Rotation Matrices
        imu_ecef, R_ecef_to_ned, R_ned_to_body = self.get_ned_to_camera_params(sbet_pose)
        
        # Calculate full rotation chain: Camera -> Body -> NED -> ECEF
        # Note: We need the inverse of the projection chain (Cam -> ECEF)
        # R_cam_to_body = R_body_to_cam.T
        R_body_to_ecef = R_ecef_to_ned.T @ R_ned_to_body.T
        
        # Calculate actual Camera Center in ECEF (IMU + Lever Arm rotation)
        cam_center_ecef = imu_ecef + (R_body_to_ecef @ self.lever_arm)
        
        # 2. Back-project UV to Camera Frame Ray
        # [u, v, 1] = K * [x, y, z]  =>  [x, y, z] = K_inv * [u, v, 1]
        uv_homog = np.array([uv[0], uv[1], 1.0])
        K_inv = np.linalg.inv(self.K)
        ray_cam = K_inv @ uv_homog
        
        # Normalize to unit vector
        ray_cam /= np.linalg.norm(ray_cam)
        
        # 3. Rotate Ray to ECEF
        # Ray_ecef = R_body_to_ecef * R_cam_to_body * ray_cam
        # recall self.R_body_to_cam is Body->Cam. So Cam->Body is its Transpose.
        ray_body = self.R_body_to_cam.T @ ray_cam
        ray_ecef = R_body_to_ecef @ ray_body
        
        # Flatten for consistency
        return cam_center_ecef.flatten(), ray_ecef.flatten()
def triangulate_n_views(centers, rays):
    """
    Finds the 3D point closest to N lines (Least Squares).
    
    Args:
        centers: List of (3,) arrays (Camera Centers)
        rays: List of (3,) arrays (Unit Ray Vectors)
        
    Returns:
        (3,) array: The optimal 3D intersection point.
    """
    # We solve for point P that minimizes sum of squared distances to lines.
    # Formulation: (I - v*v.T) * (P - C) = 0 for each view
    
    mat_sum = np.zeros((3, 3))
    vec_sum = np.zeros(3)
    
    for C, v in zip(centers, rays):
        # Projection matrix onto the subspace orthogonal to v
        # I - v*vT
        P_orth = np.eye(3) - np.outer(v, v)
        
        mat_sum += P_orth
        vec_sum += P_orth @ C
        
    # Solve linear system: mat_sum * P = vec_sum
    # Use pseudo-inverse for stability (though mat_sum usually rank 3 for N>=2)
    try:
        P_est = np.linalg.solve(mat_sum, vec_sum)
        return P_est
    except np.linalg.LinAlgError:
        return None
def evaluate_bingo_file(bingo_path, sbet_data, projector):
    """
    Parses BINGO file and evaluates 3D intersection for all landmarks.
    
    Args:
        bingo_path: Path to .dat file.
        sbet_data: Dict {image_id: {'lat':..., 'lon':..., 'alt':..., 'r':..., 'p':..., 'y':...}}
        projector: Instance of RayCastingProjector.
    """
    # 1. Parse Data
    # Blocks format: [(gcp_id, gcp_name, [(image_id, u, v), ...]), ...]
    landmarks = parse_bingo(bingo_path)
    
    results = []
    
    print(f"Evaluating {len(landmarks)} landmarks...")
    
    for gcp_id, gcp_name, observations in landmarks:
        
        centers = []
        rays = []
        valid_obs = []
        
        # 2. Gather Rays for this Landmark
        for obs in observations:
            # Note: User provided parser uses 'landmark_id' var name for the first column 
            # of the observation line, but logically this is the Image ID.
            image_id = obs[0] 
            u, v = obs[1], obs[2]
            
            if image_id in sbet_data:
                pose = sbet_data[image_id]
                center, ray = projector.get_ray_in_ecef((u, v), pose)
                
                centers.append(center)
                rays.append(ray)
                valid_obs.append({'img_id': image_id, 'u': u, 'v': v, 'pose': pose})
            else:
                print(f"Warning: Image ID {image_id} not found in SBET data.")

        # Need at least 2 rays to triangulate
        if len(centers) < 2:
            continue
            
        # 3. Triangulate 3D Point (N-View Intersection)
        p_est_ecef = triangulate_n_views(centers, rays)
        
        if p_est_ecef is None:
            continue
            
        # 4. Compute Reprojection Errors
        landmark_errors = []
        for obs in valid_obs:
            # Project estimated 3D point back to this image
            proj_uv = projector.project(p_est_ecef, obs['pose'])
            
            if proj_uv is not None:
                orig_uv = np.array([obs['u'], obs['v']])
                calc_uv = np.array(proj_uv)
                dist_err = np.linalg.norm(orig_uv - calc_uv)
                landmark_errors.append(dist_err)
        
        # 5. Store Statistics
        if landmark_errors:
            rmse = np.sqrt(np.mean(np.array(landmark_errors)**2))
            results.append({
                'gcp_id': gcp_id,
                'gcp_name': gcp_name,
                'ecef_x': p_est_ecef[0],
                'ecef_y': p_est_ecef[1],
                'ecef_z': p_est_ecef[2],
                'rmse_px': rmse,
                'num_rays': len(valid_obs),
                'max_error_px': np.max(landmark_errors)
            })

    return results

# --- Example Usage ---
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

# 1. Setup Projector (Use your specific parameters)
# Note: Ensure you use the RayCastingProjector class defined above
projector = RayCastingProjector(
    K=[[3000, 0, 1920], [0, 3000, 1080], [0, 0, 1]], 
    lever_arm=[0.1, -0.05, 0.2], 
    nominal_mount_rpy_deg=[0, 0, -90], 
    boresight_rpy_rad=[0.0, 0.0, 0.0]
)

# 2. Run Evaluation
# Assuming 'sbet_dictionary' is already loaded
results = evaluate_bingo_file("observations.dat", sbet_dictionary, projector)

# 3. Print Report
print(f"{'GCP Name':<15} | {'Rays':<5} | {'RMSE (px)':<10} | {'Status'}")
print("-" * 50)

total_rmse_accum = 0
count = 0

for res in results:
    status = "OK" if res['rmse_px'] < 3.0 else "HIGH ERROR"
    print(f"{res['gcp_name']:<15} | {res['num_rays']:<5} | {res['rmse_px']:<10.2f} | {status}")
    
    total_rmse_accum += res['rmse_px']**2
    count += 1

if count > 0:
    total_rmse = np.sqrt(total_rmse_accum / count)
    print("-" * 50)
    print(f"Total Dataset RMSE: {total_rmse:.2f} px")