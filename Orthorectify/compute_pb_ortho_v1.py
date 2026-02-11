import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from pyproj import Transformer
from scipy.interpolate import interp1d, RegularGridInterpolator, RectBivariateSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

# ==========================================
# 1. Sensor & Distortion Model
# ==========================================
class SensorModel:
    def __init__(self, params_lens, params_smile, width):
        """
        params_lens: [f, cx, cy, k1, k2, k3, p1, p2]
        params_smile: [s0, s1, s2, s3, s4] (Polynomial coeffs for sensor shape)
        width: Image width in pixels
        """
        self.f, self.cx, self.cy = params_lens[:3]
        self.dist_coeffs = params_lens[3:] # k1, k2, k3, p1, p2
        self.smile_coeffs = params_smile
        self.width = width

    def brown_projection(self, vec_cam):
        """Projects 3D camera vector (x, y, z) to Distorted Pixel Coordinates (u, v)."""
        x, y, z = vec_cam
        # Avoid division by zero
        z = np.where(z == 0, 1e-6, z)
        
        # Normalized coordinates
        xn = x / z
        yn = y / z
        
        # Radial Distortion
        r2 = xn**2 + yn**2
        k1, k2, k3, p1, p2 = self.dist_coeffs
        rad = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)
        
        # Tangential Distortion
        dx = 2*p1*xn*yn + p2*(r2 + 2*xn**2)
        dy = p1*(r2 + 2*yn**2) + 2*p2*xn*yn
        
        xd = xn * rad + dx
        yd = yn * rad + dy
        
        # Project to Pixel Plane
        u = self.f * xd + self.cx
        v = self.f * yd + self.cy
        return u, v

    def get_smile_offset(self, u_pixel):
        """Calculates physical sensor v-offset (the 'smile') for a given column."""
        # Simple polynomial evaluation
        # Ensure u_pixel is normalized if your calibration requires it!
        # Here assuming coeffs are for raw pixel indices.
        return np.polyval(self.smile_coeffs[::-1], u_pixel)

# ==========================================
# 2. Core Geometry Engine
# ==========================================
class OrthoEngine:
    def __init__(self, traj_data, exposure_times, sensor, mount, dsm_path, bil_path):
        self.sensor = sensor
        self.exposure_times = exposure_times
        self.t_start = exposure_times[0]
        self.t_end = exposure_times[-1]
        
        # --- Coordinate Transformers ---
        # WGS84 (Lat/Lon) -> ECEF (X/Y/Z)
        self.geo2ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
        # ECEF -> UTM (Output Projection)
        self.ecef2utm = Transformer.from_crs("epsg:4978", "epsg:32633", always_xy=True)
        # UTM -> ECEF
        self.utm2ecef = Transformer.from_crs("epsg:32633", "epsg:4978", always_xy=True)

        # --- Trajectory Interpolation ---
        print("Building Trajectory Interpolators...")
        # Position (ECEF)
        x, y, z = self.geo2ecef.transform(traj_data['lon'], traj_data['lat'], traj_data['height'])
        self.pos_interp = interp1d(traj_data['time'], np.stack([x, y, z], axis=1), axis=0, kind='cubic', bounds_error=False, fill_value="extrapolate")
        
        # Attitude (Roll, Pitch, Yaw) + Lat/Lon for Local Tangent Plane
        # Storing [Roll, Pitch, Yaw, Lat, Lon]
        att_stack = np.stack([traj_data['roll'], traj_data['pitch'], traj_data['yaw'], traj_data['lat'], traj_data['lon']], axis=1)
        self.att_interp = interp1d(traj_data['time'], att_stack, axis=0, kind='cubic', bounds_error=False, fill_value="extrapolate")

        # Mean Speed (for Newton-Raphson derivative)
        vel_vec = np.diff(np.stack([x, y, z], axis=1), axis=0)
        dt = np.diff(traj_data['time'])
        self.mean_speed = np.mean(np.linalg.norm(vel_vec, axis=1) / dt)

        # --- Mounting Matrices ---
        # Lever Arm (Body Frame: Forward, Right, Down)
        self.lever_arm = np.array(mount['lever_arm'])
        # Rotation: Camera -> Body (Mount * Boresight)
        r_bore = R.from_euler('xyz', mount['boresight'], degrees=True).as_matrix()
        r_mount = np.array(mount.get('matrix', np.eye(3)))
        self.r_cam2body = r_mount @ r_bore

        # --- Spatial Index (Time-Space Index) ---
        print("Building Time-Space Index...")
        self.build_tsi()

        # --- DSM & Image ---
        self.dsm_path = dsm_path
        self.bil_path = bil_path
        self.bil_shape = (len(exposure_times), sensor.width)

    def build_tsi(self, step=500):
        """Pre-calculates camera positions for fast initial time lookup."""
        sample_t = self.exposure_times[::step]
        if len(sample_t) == 0: raise ValueError("No exposure times found.")
        
        pos = self.pos_interp(sample_t)
        att = self.att_interp(sample_t)
        
        # Calculate actual camera center (apply lever arm)
        cam_centers = []
        for i in range(len(sample_t)):
            r, p, y, lat, lon = att[i]
            r_b2e = self.get_body2ecef(r, p, y, lat, lon)
            cam_centers.append(pos[i] + r_b2e @ self.lever_arm)
            
        self.tsi_tree = cKDTree(np.array(cam_centers))
        self.tsi_times = sample_t

    def get_body2ecef(self, r, p, y, lat, lon):
        """Constructs Body(IMU) -> ECEF Rotation Matrix."""
        # 1. Body -> NED
        r_b2n = R.from_euler('xyz', [r, p, y], degrees=True).as_matrix()
        
        # 2. NED -> ECEF
        sl, cl = np.sin(np.radians(lat)), np.cos(np.radians(lat))
        slo, clo = np.sin(np.radians(lon)), np.cos(np.radians(lon))
        
        r_n2e = np.array([
            [-sl*clo, -slo, -cl*clo],
            [-sl*slo,  clo, -cl*slo],
            [ cl,      0,   -sl    ]
        ])
        return r_n2e @ r_b2n

    def ground_to_image(self, ground_ecef):
        """
        Maps ECEF Ground Point -> (Line, Sample).
        Returns (-1, -1) if out of bounds.
        """
        # 1. Initial Guess
        _, idx = self.tsi_tree.query(ground_ecef)
        t = self.tsi_times[idx]
        
        # 2. Newton-Raphson
        for _ in range(5):
            if t < self.t_start or t > self.t_end: return -1, -1
            
            # Interpolate state
            pos_imu = self.pos_interp(t)
            att = self.att_interp(t) # [r, p, y, lat, lon]
            
            # Rotations & Lever Arm
            r_b2e = self.get_body2ecef(*att)
            cam_center = pos_imu + r_b2e @ self.lever_arm
            r_c2e = r_b2e @ self.r_cam2body
            
            # Vector Ground -> Camera (in Camera Frame)
            vec_cam = r_c2e.T @ (ground_ecef - cam_center)
            
            # Project
            u, v_proj = self.sensor.brown_projection(vec_cam)
            
            # Check against Smile (Physical Sensor Shape)
            v_sensor = self.sensor.get_smile_offset(u)
            
            # Error (in meters approx)
            scale = vec_cam[2] / self.sensor.f
            err_m = (v_proj - v_sensor) * scale
            
            # Update t
            dt = err_m / self.mean_speed
            t -= dt
            
            if abs(dt) < 1e-5: break

        # 3. Convert Time to Line Index
        # Assuming linear exposure times for fast lookup, else use searchsorted
        # line = (t - t_start) * fps
        line = np.searchsorted(self.exposure_times, t)
        
        if 0 <= line < self.bil_shape[0] and 0 <= u < self.sensor.width:
            return line, u
        return -1, -1

# ==========================================
# 3. Tile Processor (Worker)
# ==========================================
def process_tile(engine, window, transform, output_shape):
    """
    Processing logic for a single tile.
    Uses Sparse Grid Interpolation for speed.
    """
    h, w = window.height, window.width
    
    # --- 1. Load DSM for this tile ---
    # (In production, pass shared memory or read window from DSM file)
    with rasterio.open(engine.dsm_path) as dsm:
        # Read DSM window matching the output tile (approximate mapping required)
        # For simplicity here, we assume 1:1 match or read full coverage
        # *Production*: Use dsm.read(1, window=dsm_window)
        dsm_data = dsm.read(1, window=window, boundless=True)

    # --- 2. Create Sparse Grid ---
    step = 20 # Calculate geometry every 20 pixels
    grid_y = np.arange(0, h + step, step)
    grid_x = np.arange(0, w + step, step)
    
    map_lines = np.full((len(grid_y), len(grid_x)), -1.0)
    map_pixels = np.full((len(grid_y), len(grid_x)), -1.0)
    
    valid_mask = np.zeros_like(map_lines, dtype=bool)

    # --- 3. Solve Geometry on Sparse Grid ---
    for i, r in enumerate(grid_y):
        for j, c in enumerate(grid_x):
            # Clip to window size
            r_safe = min(r, h-1)
            c_safe = min(c, w-1)
            
            # Pixel Center -> UTM
            mx, my = transform * (c_safe + window.col_off, r_safe + window.row_off)
            
            # UTM -> ECEF
            z = dsm_data[r_safe, c_safe] # Get height
            gx, gy, gz = engine.utm2ecef.transform(mx, my, z)
            
            # Ray Trace
            l, p = engine.ground_to_image(np.array([gx, gy, gz]))
            
            if l != -1:
                map_lines[i, j] = l
                map_pixels[i, j] = p
                valid_mask[i, j] = True

    # --- 4. Interpolate Sparse Grid to Full Resolution ---
    # Create interpolators (Coordinate Look-Up Table - CLUT)
    # Filter invalid points or fill nearest to avoid holes at edges
    # (Simplified: assume valid coverage for this snippet)
    
    # We use RectBivariateSpline for high-speed interpolation
    # Note: map_lines contains floating point line indices (time)
    interp_l = RectBivariateSpline(grid_y, grid_x, map_lines, kx=1, ky=1)
    interp_p = RectBivariateSpline(grid_y, grid_x, map_pixels, kx=1, ky=1)
    
    # Generate full grid coords
    mesh_y, mesh_x = np.mgrid[0:h, 0:w]
    full_l = interp_l.ev(mesh_y, mesh_x)
    full_p = interp_p.ev(mesh_y, mesh_x)
    
    # --- 5. Resample from BIL ---
    # Determine bounds to read from disk
    l_min, l_max = int(np.min(full_l)), int(np.max(full_l)) + 2
    
    # Safety checks
    if l_max < 0 or l_min >= engine.bil_shape[0]:
        return np.zeros((h, w), dtype='uint16')

    l_min = max(0, l_min)
    l_max = min(engine.bil_shape[0], l_max)
    
    # Read chunk from BIL
    # Memmap slice is efficient
    bil_mmap = np.memmap(engine.bil_path, dtype='uint16', mode='r', shape=engine.bil_shape)
    chunk = np.array(bil_mmap[l_min:l_max, :]) # Load into RAM
    
    # Adjust line indices relative to chunk
    rel_l = full_l - l_min
    
    # Bilinear Sampling using scipy.ndimage.map_coordinates
    # Input coordinates must be (row, col)
    # Order=1 is bilinear
    coords = np.array([rel_l.ravel(), full_p.ravel()])
    sampled = map_coordinates(chunk, coords, order=1, mode='nearest', prefilter=False)
    
    return sampled.reshape((h, w)).astype('uint16')

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # --- Configuration ---
    dsm_file = "site_dsm.tif"
    bil_file = "flight_line_01.bil"
    sbet_file = "trajectory.txt" # CSV with time,lat,lon,h,r,p,y
    output_file = "ortho_result.tif"
    
    # Calibration Data
    cam_params = [5000.0, 3000.0, 0.0, 1e-5, 0.0, 0.0, 0.0, 0.0] # f, cx, cy, k...
    smile_params = [0, 0, 0, 0, 0] # Flat sensor
    mount_cfg = {
        'lever_arm': [0.2, 0.1, 0.5], # Fwd, Right, Down
        'boresight': [0.01, -0.02, 0.5] # R, P, Y
    }

    # Load Trajectory
    traj = pd.read_csv(sbet_file)
    # Generate synthetic exposure times (e.g., 300 Hz)
    times = np.linspace(traj['time'].iloc[0], traj['time'].iloc[-1], int(len(traj)*3))

    # Initialize Engine
    sensor = SensorModel(cam_params, smile_params, width=6000)
    engine = OrthoEngine(traj, times, sensor, mount_cfg, dsm_file, bil_file)

    # Setup Output
    with rasterio.open(dsm_file) as dsm:
        profile = dsm.profile
        profile.update(dtype='uint16', count=1, compress='lzw')
        
        # Define Tiling
        tile_size = 1024
        windows = [win for _, win in dsm.block_windows(1)] # Or custom grid
        
        # Prepare arguments for parallel workers
        tasks = []
        for win in windows:
            # Adjust window to be multiple of tile_size or handle edges
            tasks.append((engine, win, dsm.transform, (win.height, win.width)))

    # Run Parallel Processing
    print(f"Processing {len(tasks)} tiles with {mp.cpu_count()} cores...")
    
    # Note: 'engine' is pickled to workers. Ensure it's read-only/thread-safe.
    # For very large objects, consider using shared memory or re-init in worker.
    with rasterio.open(output_file, 'w', **profile) as dst:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Map wrapper function
            # We use a lambda or separate function to unpack args
            results = executor.map(wrapper_process, tasks)
            
            for win, data in zip(windows, results):
                dst.write(data, 1, window=win)

def wrapper_process(args):
    # Unpack arguments for the worker
    engine, win, transform, shape = args
    return process_tile(engine, win, transform, shape)

if __name__ == "__main__":
    main()