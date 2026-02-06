import numpy as np
from scipy.spatial.transform import Rotation as R

class CameraModel:
    def __init__(self, fx, fy, cx, cy, distortion=None):
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        self.dist = distortion  # Assuming [k1, k2, p1, p2, k3] standard OpenCV

class Transform:
    """
    Represents a rigid body transform (Rotation + Translation).
    Convention: T_a_b transforms a point from frame b to frame a.
    P_a = R_ab * P_b + t_ab
    """
    def __init__(self, rotation_quat, translation):
        # rotation_quat: [x, y, z, w]
        self.r = R.from_quat(rotation_quat)
        self.t = np.array(translation)

    def apply(self, points_b):
        """Transform points from frame b to frame this."""
        # points_b shape: (N, 3)
        return self.r.apply(points_b) + self.t

    def inverse(self):
        """Returns the inverse transform (T_b_a)."""
        inv_r = self.r.inv()
        inv_t = -inv_r.apply(self.t)
        # Convert back to init format
        return Transform(inv_r.as_quat(), inv_t)

def project_world_to_image(points_world, traj_pose_wb, mount_cb, camera):
    """
    Projects 3D world points to 2D pixel coordinates.
    
    Args:
        points_world: (N, 3) array of 3D coordinates.
        traj_pose_wb: Transform T_world_body (Body pose in World).
        mount_cb: Transform T_body_cam (Camera pose in Body/Mount).
        camera: CameraModel object.
        
    Returns:
        points_image: (N, 2) projected pixel coordinates.
    """
    # 1. World -> Body Frame
    # T_body_world is inverse of T_world_body
    T_bw = traj_pose_wb.inverse()
    points_body = T_bw.apply(points_world)

    # 2. Body -> Camera Frame
    # T_cam_body is inverse of T_body_cam (Mount)
    T_cb = mount_cb.inverse()
    points_cam = T_cb.apply(points_body)

    # 3. Project to Normalized Image Plane (z=1)
    # Check for points behind camera
    depths = points_cam[:, 2]
    valid_mask = depths > 0
    
    # Perspective division
    x_norm = points_cam[:, 0] / depths
    y_norm = points_cam[:, 1] / depths

    # 4. Apply Distortion (Simple Radial/Tangential model approximation)
    if camera.dist is not None:
        k1, k2, p1, p2, k3 = camera.dist
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2**3
        
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        x_dist = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
        y_dist = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
        
        x_norm, y_norm = x_dist, y_dist

    # 5. Apply Intrinsics (K matrix)
    u = camera.K[0, 0] * x_norm + camera.K[0, 2]
    v = camera.K[1, 1] * y_norm + camera.K[1, 2]

    return np.stack([u, v], axis=1), valid_mask

def ray_trace_distance(points_image, points_world, traj_pose_wb, mount_cb, camera):
    """
    Calculates distance between the 3D point and the ray back-projected from the image pixel.
    """
    # 1. Back-project pixel to ray in Camera Frame
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    
    # Undistortion should ideally happen here, skipping for simple ray direction
    u, v = points_image[:, 0], points_image[:, 1]
    ray_x = (u - cx) / fx
    ray_y = (v - cy) / fy
    ray_z = np.ones_like(ray_x)
    
    rays_cam = np.stack([ray_x, ray_y, ray_z], axis=1)
    # Normalize ray vectors
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)

    # 2. Transform Rays: Camera -> Body -> World
    # Rotate rays (Translation doesn't affect direction vector, only origin)
    rays_body = mount_cb.r.apply(rays_cam)
    rays_world = traj_pose_wb.r.apply(rays_body)

    # 3. Calculate Camera Center in World Frame
    # C_w = T_wb * (T_bc * [0,0,0]) -> effectively T_wb.t + R_wb * T_bc.t
    cam_origin_in_body = mount_cb.t
    cam_origin_in_world = traj_pose_wb.apply(cam_origin_in_body)

    # 4. Compute Distance from Point to Line
    # Vector from Camera Center to 3D Point
    vec_cam_to_point = points_world - cam_origin_in_world
    
    # Cross product of (Ray Direction) and (Vector to Point) gives area of parallelogram
    # Height (distance) = Area / Base (norm of ray direction, which is 1)
    cross_prod = np.cross(rays_world, vec_cam_to_point)
    distances = np.linalg.norm(cross_prod, axis=1)
    
    return distances

# --- Usage Example ---

def verify_constraints():
    print("--- Starting Constraint Verification ---\n")

    # 1. Setup Mock Data (Replace with your actual data)
    
    # Camera Intrinsics (Example: 2000x2000 image, 50mm lens)
    cam = CameraModel(fx=1696.41, fy=0.5, cx=619.5, cy=0.5, distortion=[0, 0, 0, 0, 0])

    # Trajectory (Body Pose in World)
    # Vehicle is at World (100, 100, 500) looking roughly down
    traj_rot = R.from_euler('xyz', [180, 0, 0], degrees=True).as_quat() # Looking down
    traj_pos = [100, 100, 500] 
    T_world_body = Transform(traj_rot, traj_pos)

    # Mount/Boresight (Camera Pose in Body)
    # Camera is mounted with a slight offset and rotation relative to platform center
    mount_rot = R.from_euler('xyz', [0, 0, 2], degrees=True).as_quat() # 2 deg yaw offset
    mount_pos = [0.1, 0.0, -0.2] # 10cm offset
    T_body_cam = Transform(mount_rot, mount_pos)

    # Tie Points (Observation and Ground Truth)
    # Let's say we have a 3D point directly below the camera
    world_pts = np.array([
        [100, 100, 0],   # Directly below
        [120, 120, 10]   # Slightly off-center and elevated
    ])
    
    # Fake "Observed" Image Coordinates (Simulating observations)
    # In a real scenario, you load these from your dataset
    # Here we generate "perfect" ones and add noise to test the verifier
    obs_uv, _ = project_world_to_image(world_pts, T_world_body, T_body_cam, cam)
    obs_uv[0] += [2.0, -1.5] # Add synthetic error to point 1

    # 2. Run Verification
    
    # Check 1: Reprojection Error (3D -> 2D)
    proj_uv, valid_mask = project_world_to_image(world_pts, T_world_body, T_body_cam, cam)
    residuals = obs_uv - proj_uv
    reproj_error = np.linalg.norm(residuals, axis=1)

    print("### Test 1: Reprojection Consistency (3D -> 2D)")
    print(f"{'Point ID':<10} | {'Observed (px)':<15} | {'Projected (px)':<15} | {'Error (px)':<10}")
    print("-" * 60)
    for i, err in enumerate(reproj_error):
        obs_str = f"{obs_uv[i,0]:.1f}, {obs_uv[i,1]:.1f}"
        proj_str = f"{proj_uv[i,0]:.1f}, {proj_uv[i,1]:.1f}"
        print(f"{i:<10} | {obs_str:<15} | {proj_str:<15} | {err:.4f}")
    
    rmse = np.sqrt(np.mean(reproj_error**2))
    print(f"\nTotal RMSE: {rmse:.4f} pixels")
    if rmse > 5.0:
        print(">> WARNING: High Reprojection Error. Check Extrinsics or Intrinsics.")
    else:
        print(">> PASS: Reprojection within acceptable bounds.")

    print("\n" + "="*30 + "\n")

    # Check 2: Ray Casting Distance (2D -> 3D)
    # This checks if the 3D point lies on the ray generated by the pixel
    ray_dists = ray_trace_distance(obs_uv, world_pts, T_world_body, T_body_cam, cam)
    
    print("### Test 2: Ray Intersection Consistency (2D -> 3D)")
    print(f"{'Point ID':<10} | {'Ray Dist (units)':<15}")
    print("-" * 30)
    for i, dist in enumerate(ray_dists):
        print(f"{i:<10} | {dist:.4f}")

    if np.mean(ray_dists) > 1.0: # Threshold depends on your scene scale
        print("\n>> WARNING: 3D points are far from observed rays.")
    else:
        print("\n>> PASS: 3D points lie sufficiently close to viewing rays.")

if __name__ == "__main__":
    verify_constraints()
