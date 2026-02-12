import yaml
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import cKDTree
import os
from liblibor.map import TangentPlane, Trajectory,Pose_std,log, loadSBET
from photogrammetry_verify.io_utils import load_av4_timing
from pathlib import Path
import spectral.io.envi as envi
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Global worker variable ---
# This holds the pipeline instance for each process
worker_pipeline = None

def init_worker(config_path):
    """Initializes the pipeline on each worker process."""
    global worker_pipeline
    worker_pipeline = OrthoPipeline(config_path)

def process_window_worker(window):
    """The function executed by each core."""
    # window is a rasterio.windows.Window object
    data = worker_pipeline.process_tile(window)
    return window, data

# ==========================================
# 1. Camera Model (Optics)
# ==========================================
class CameraModel:
    def __init__(self, cfg_cam):
        self.fx = cfg_cam['focal_length_px']
        self.fy = cfg_cam['focal_length_px']
        self.cx = cfg_cam['principal_point'][0]
        self.cy = cfg_cam['principal_point'][1]
        self.width = cfg_cam['image_size'][0]
        self.height = cfg_cam['image_size'][1]
        
        # Distortion Coefficients
        self.k = np.array(cfg_cam.get('k', [0,0,0]))
        self.p = np.array(cfg_cam.get('p', [0,0]))

    def project(self, points_cam):
        """
        Project 3D camera-frame points (Nx3) to 2D pixel coordinates (Nx2).
        Includes Brown-Conrady Distortion.
        """
        z = points_cam[:, 2]
        # Avoid division by zero
        z[z == 0] = 1e-6
        
        x_n = points_cam[:, 0] / z
        y_n = points_cam[:, 1] / z

        # Distortion Model
        r2 = x_n**2 + y_n**2
        r4 = r2**2
        r6 = r2**3
        rad = 1 + self.k[0]*r2 + self.k[1]*r4 + self.k[2]*r6
        
        dx = 2*self.p[0]*x_n*y_n + self.p[1]*(r2 + 2*x_n**2)
        dy = self.p[0]*(r2 + 2*y_n**2) + 2*self.p[1]*x_n*y_n
        
        x_d = x_n * rad + dx
        y_d = y_n * rad + dy
        
        # Apply Intrinsics
        u = self.fx * x_d + self.cx
        v = self.fy * y_d + self.cy
        
        return np.stack([u, v], axis=1)

# ==========================================
# 2. Orthorectification Engine
# ==========================================
class OrthoPipeline:
    def __init__(self, config_path):
        print(f"Initializing Pipeline with {config_path}")
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.cam_model = CameraModel(self.cfg['camera'])
        self.setup_transforms()
        self.setup_mounting()
        self.load_and_interpolate_SBET()
        self.load_dsm()
        
        # --- Spectral IO Integration ---
        self.img_path = self.cfg['paths']['image_file']
        # Automatically find the .hdr associated with the image file
        hdr_path = os.path.splitext(self.img_path)[0] + '.hdr'
        
        print(f"Opening ENVI file: {self.img_path}")
        # envi.open returns a metadata object
        self.envi_obj = envi.open(hdr_path, self.img_path)
        
        # Create a memory-mapped array. 
        # Spectral Python ALWAYS maps this as (Lines, Samples, Bands) 
        # regardless of whether the file is BIL, BIP, or BSQ.
        self.img_data = self.envi_obj.open_memmap(writable=False)
        
        self.n_lines = self.envi_obj.nrows
        self.n_samples = self.envi_obj.ncols
        self.n_bands = self.envi_obj.nbands
        
        print(f"Data ready: {self.n_lines} lines, {self.n_samples} samples, {self.n_bands} bands.")

    def setup_transforms(self):
        epsg_out = self.cfg['project'].get('epsg_out', 32632)
        print(f"Setting up transforms for EPSG:{epsg_out}")
        self.geo_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
        self.local_to_ecef = Transformer.from_crs(f"epsg:{epsg_out}", "epsg:4978", always_xy=True)

    def setup_mounting(self):
        m = self.cfg['mounting']
        self.lever_arm_body = np.array(m['lever_arm'])
        
        # 1. Load Mounting Matrix (3x3)
        # Reshape flat list [r11, r12... r33] into (3,3)
        r_mount = np.array(m['mounting_matrix']).reshape(3, 3)
        
        # 2. Load Boresight (Degrees)
        # Create rotation from Euler angles
        r_bore = R.from_euler('xyz', m['boresight_rpy_deg'], degrees=True).as_matrix()
        
        # 3. Composite Rotation: Camera -> Body
        # R_total = R_mount * R_boresight
        self.R_cam_to_body = r_mount @ r_bore

    def load_and_interpolate_SBET(self):
        """
        Loads SBET and Timing. 
        Crucial Step: Interpolates SBET attributes specifically to the exact 
        timestamp of every image scanline.
        """
        print("Loading Trajectory & Timing...")

        av4_timing = load_av4_timing(self.cfg['paths']['timing_file'])
        # Check if input is in 10 microseconds and convert to seconds if needed
        if av4_timing.max() > 1e8:  # Likely in 10 microseconds
            img_times = av4_timing / 1e5  # Convert to seconds
        else:
            img_times = av4_timing
        self.line_times = img_times
        # Define time span of images
        #img_times = np.array([timing_map[img_id]  for img_id in timing_map])
        time_buffer = 100
        img_time_span = [img_times.min()-time_buffer, img_times.max()+time_buffer]
    
        t_start, t_end = img_time_span
        print(f"Trajectory time range: {t_start:.3f} to {t_end:.3f}")

        if self.cfg['project'].get('parse_sbet', True):
            log("[2/3] Loading SBET data...", verbose=True, force=True)
            # Extract time, lla, rpy from sbet_df
            print(f"Loading SBET from {self.cfg['paths']['sbet_file']}...")
            t,lla,rpy = loadSBET(Path(self.cfg['paths']['sbet_file']))
            mask = (t >= img_time_span[0]) & (t <= img_time_span[1])
            tspan = t[mask]
            img_lla = lla[mask,:]
            img_rpy = rpy[mask,:]
            lat0 = np.degrees(img_lla[0,0])
            lon0 = np.degrees(img_lla[0,1])
            alt0 = img_lla[0,2]
            print(f"Reference LTP origin: lat: {lat0:.6f} lon: {lon0:.6f} alt: {alt0:.3f}")
            tangentPlane = TangentPlane(lat0, lon0,alt0)
            
            trajectory = Trajectory(t, lla, rpy, tangentPlane, img_time_span)
            print(f"    Loaded {len(trajectory.t)} trajectory epochs")
            log("[3/3] Interpolating poses...", verbose=True, force=True)
        # Create coordinate transformer

        # Write trajectory within image time span to CSV for debugging
            with open(self.cfg['paths']['trajectory_file'], "w") as f:
                f.write("time,lat,lon,alt,roll,pitch,yaw\n")
                for i in range(len(trajectory.t)):
                    f.write(f"{trajectory.t[i]},{trajectory.lla[0,i]},{trajectory.lla[1,i]},{trajectory.lla[2,i]},{trajectory.rpy[0,i]},{trajectory.rpy[1,i]},{trajectory.rpy[2,i]}\n")
            print(f"    Wrote trajectory to {self.cfg['paths']['trajectory_file']}")
        # Interpolate poses at image times

            # write t,lla,rpy to csv downsampled every 100th for debugging
            with open(self.cfg['paths']['trajectory_file'] + "_downsampled.csv", "w") as f:
                f.write("time,lat,lon,alt,roll,pitch,yaw\n")
                for i in range(0, len(t),100):
                    f.write(f"{t[i]},{lla[i,0]},{lla[i,1]},{lla[i,2]},{rpy[i,0]},{rpy[i,1]},{rpy[i,2]}\n")
            # Compute interpolated poses at image times
            img_poses = trajectory.interpolate(img_times, self.cfg,customRPY=False)
            # write poses to csv for debugging
            with open(self.cfg['paths']['poses_file'], "w") as f:
                f.write("time,lat,lon,alt,roll,pitch,yaw\n")
                for pose in img_poses:
                    f.write(f"{pose.t},{pose.lla[0]},{pose.lla[1]},{pose.lla[2]},{pose.rpy[0]},{pose.rpy[1]},{pose.rpy[2]}\n")
            print(f"    Interpolated {len(img_poses)} image poses")
        else:
            # Load poses from defined csv path
            poses_csv = np.loadtxt(self.cfg['paths']['poses_file'], delimiter=',',skiprows=1)
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
                self.cfg['project']['epsg_out'],
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

        # 2. Position Interpolation (Translational Manifold)
        #sb_x, sb_y, sb_z = self.geo_to_ecef.transform(trajectory.lla[0,:], trajectory.lla[1,:], trajectory.lla[2,:])
        #interp_pos = interp1d(trajectory.t, np.stack([sb_x, sb_y, sb_z], axis=1), axis=0, kind='cubic', fill_value="extrapolate")
        
        # create ecef posisions from img_poses
        self.line_pos_ecef = np.array([pose.ecef for pose in img_poses])
        self.line_geo = np.array([pose.lla[:2] for pose in img_poses]) # lat, lon for each line

        line_rots = np.array([pose.R_ned2b for pose in img_poses]) # Rotation from NED to Body for each line
        
    
        # G. Pre-calculate R_body_to_ecef for every line
        print("Constructing Rotation Matrices...")
        self.line_R_b2e = self.compute_all_line_rotations(line_rots, self.line_geo)
        
        # H. Compute Mean Speed (for Time-Error conversion)
        d_pos = np.diff(self.line_pos_ecef, axis=0)
        d_time = np.diff(self.line_times)
        speeds = np.linalg.norm(d_pos, axis=1) / d_time
        self.mean_speed = np.nanmean(speeds)
        
        # 7. Spatial Indexing
        step = 50
        self.tsi_tree = cKDTree(self.line_pos_ecef[::step])
        self.tsi_times = self.line_times[::step]

    def load_and_interpolate_trajectory(self):
        """
        Loads SBET and Timing. 
        Uses Slerp for attitude interpolation to respect Lie Algebra.
        """
        print("Loading Trajectory & Timing...")
        
        # A. Load Timing (Exact UTC seconds for each line)
        timing_df = pd.read_csv(self.cfg['paths']['timing_file'], sep=r'\s+', header=None, names=['id', 'time'])
        self.line_times = timing_df['time'].values
        
        # B. Load SBET
        sbet = pd.read_csv(self.cfg['paths']['sbet_file'], sep=r'\s+', header=None, 
                           names=['time', 'lat', 'lon', 'height', 'roll', 'pitch', 'yaw'])
        
        # C. Position Interpolation (Translational Manifold)
        sb_x, sb_y, sb_z = self.geo_to_ecef.transform(sbet['lon'].values, sbet['lat'].values, sbet['height'].values)
        interp_pos = interp1d(sbet['time'], np.stack([sb_x, sb_y, sb_z], axis=1), axis=0, kind='cubic', fill_value="extrapolate")
        
        # D. Attitude Interpolation (Rotation Manifold SO(3) via Slerp)
        # Create Rotation objects from SBET RPY
        sbet_rotations = R.from_euler('xyz', sbet[['roll', 'pitch', 'yaw']].values, degrees=True)
        slerp = Slerp(sbet['time'].values, sbet_rotations)
        
        # E. Geodetic Interpolation (for local NED construction)
        interp_geo = interp1d(sbet['time'], sbet[['lat', 'lon']].values, axis=0, kind='linear', fill_value="extrapolate")

        # F. Interpolate Exact Poses for Every Scanline
        print(f"Interpolating manifold-consistent poses for {len(self.line_times)} lines...")
        self.line_pos_ecef = interp_pos(self.line_times)
        line_rots = slerp(self.line_times)
        self.line_geo = interp_geo(self.line_times)

        # G. Pre-calculate R_body_to_ecef for every line
        print("Constructing Rotation Matrices...")
        self.line_R_b2e = self.compute_all_line_rotations(line_rots, self.line_geo)
        
        # H. Compute Mean Speed (for Time-Error conversion)
        d_pos = np.diff(self.line_pos_ecef, axis=0)
        d_time = np.diff(self.line_times)
        speeds = np.linalg.norm(d_pos, axis=1) / d_time
        self.mean_speed = np.nanmean(speeds)
        
        # I. Spatial Indexing
        step = 50
        print(f"Building KDTree (step={step})...")
        # Use center of scanline approximation for index
        self.tsi_tree = cKDTree(self.line_pos_ecef[::step])
        self.tsi_times = self.line_times[::step]
        print("Trajectory Ready.")

    def compute_all_line_rotations(self, line_rots, line_geo):
        """
        Computes R_b2e = R_ned2ecef * R_body2ned for all lines efficiently.
        """
        lats = np.radians(line_geo[:, 0])
        lons = np.radians(line_geo[:, 1])
        
        sl, cl = np.sin(lats), np.cos(lats)
        slo, clo = np.sin(lons), np.cos(lons)
        
        # Construct NED to ECEF matrices (N, 3, 3)
        zero = np.zeros_like(sl)
        # Transpose logic: np.array structure is (3, 3, N), need (N, 3, 3)
        R_n2e = np.array([
            [-sl*clo, -slo, -cl*clo],
            [-sl*slo,  clo, -cl*slo],
            [ cl,      zero, -sl    ]
        ]).transpose(2, 0, 1)
        
        # R_body2ned from Slerp
        R_b2n = np.array([Rmat.as_matrix() for Rmat in line_rots])
        
        # Final R_body2ecef
        return R_n2e @ R_b2n

    def load_dsm(self):
        print("Loading DSM...")
        with rasterio.open(self.cfg['paths']['dsm_file']) as src:
            data = src.read(1)
            # RegularGridInterpolator setup
            x = np.linspace(src.bounds.left, src.bounds.right, src.width)
            y = np.linspace(src.bounds.bottom, src.bounds.top, src.height)
            
            # Flip UD because image coords are typically Top-Left origin
            data_flipped = np.flipud(data)
            self.dsm_interp = RegularGridInterpolator((y, x), data_flipped, bounds_error=False, fill_value=np.nan)
            self.dsm_transform = src.transform
            self.dsm_src = src # Keep handle for profile

    def ground_to_image_mod(self, points_ecef):
        """
        Finds the (Line, Sample) for ground points ECEF.
        """
        # 1. Initial Guess via KDTree
        _, idxs_sub = self.tsi_tree.query(points_ecef)
        current_line_indices = idxs_sub * 50 # matches step used in load_trajectory
        current_line_indices = np.clip(current_line_indices, 0, len(self.line_times)-1)

        # 2. Iterative Refinement
        for _ in range(5):
            idx_int = current_line_indices.astype(int)
            
            pos_body = self.line_pos_ecef[idx_int] # (N, 3)
            R_b2e = self.line_R_b2e[idx_int]       # (N, 3, 3)
            
            # Camera Center = Pos + R_b2e * Lever
            lever_rot = R_b2e @ self.lever_arm_body
            cam_centers = pos_body + lever_rot
            
            # Vector Ground -> Camera
            vec_global = points_ecef - cam_centers
            
            # Transform to Camera Frame
            # R_c2e = R_b2e @ R_cam_to_body
            R_c2e = R_b2e @ self.R_cam_to_body
            vec_cam = np.einsum('nij,nj->ni', R_c2e.transpose(0, 2, 1), vec_global)
            
            # Project to Pixels
            uv = self.cam_model.project(vec_cam) # (N, 2) [u, v]
            
            # Error Check (along-track v)
            v_error_meters = vec_cam[:, 1]
            dt = v_error_meters / self.mean_speed
            
            # Convert time delta to index delta
            avg_line_dur = (self.line_times[-1] - self.line_times[0]) / len(self.line_times)
            d_index = dt / avg_line_dur
            
            current_line_indices = current_line_indices - d_index
            current_line_indices = np.clip(current_line_indices, 0, len(self.line_times)-1)
            
            if np.mean(np.abs(d_index)) < 0.1: # Convergence
                break
                
        return np.stack([current_line_indices, uv[:, 0]], axis=1)

    def process_tile_old(self, window):
        # 1. Create Output Grid
        transform = self.dsm_transform 
        x_off, y_off = window.col_off, window.row_off
        w, h = window.width, window.height
        
        # Grid Coordinates
        xs = np.arange(x_off, x_off + w) * transform.a + transform.c
        ys = np.arange(y_off, y_off + h) * transform.e + transform.f
        xx, yy = np.meshgrid(xs, ys)
        
        # Flatten
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        
        # 2. Get Z from DSM
        # RegularGridInterpolator takes (y, x)
        flat_z = self.dsm_interp((flat_y, flat_x))
        
        # Filter NaNs
        valid_mask = ~np.isnan(flat_z)
        if not np.any(valid_mask):
            return np.zeros((h, w), dtype='uint8')
        
        # 3. Convert valid pixels to ECEF
        fx, fy, fz = flat_x[valid_mask], flat_y[valid_mask], flat_z[valid_mask]
        ecef_x, ecef_y, ecef_z = self.local_to_ecef.transform(fx, fy, fz)
        pts_ecef = np.stack([ecef_x, ecef_y, ecef_z], axis=1)
        
        # 4. Solve Geometry
        # Returns (Line, Sample)
        img_coords = self.ground_to_image(pts_ecef)
        
        # 5. Sampling
        lines = np.round(img_coords[:, 0]).astype(int)
        samps = np.round(img_coords[:, 1]).astype(int)
        
        # Boundary Checks
        H_img, W_img = self.img_shape
        in_bounds = (lines >= 0) & (lines < H_img) & (samps >= 0) & (samps < W_img)
        
        # 6. Fill Output Array
        out_flat = np.zeros(len(flat_x), dtype='uint16')
        
        if self.img_data is not None:
             valid_indices = np.where(valid_mask)[0][in_bounds]
             # Vectorized read from Memmap
             vals = self.img_data[lines[in_bounds], samps[in_bounds]]
             out_flat[valid_indices] = vals
             
        return out_flat.reshape(h, w)

    def ground_to_image(self, points_ecef):
        _, idxs_sub = self.tsi_tree.query(points_ecef)
        cur_idx = np.clip(idxs_sub * 50, 0, self.n_lines - 1).astype(float)
        avg_dt = (self.line_times[-1] - self.line_times[0]) / self.n_lines

        for _ in range(5):
            idx_int = cur_idx.astype(int)
            R_c2e = np.einsum('nij,jk->nik', self.line_R_b2e[idx_int], self.R_cam_to_body)
            cam_centers = self.line_pos_ecef[idx_int] + np.einsum('nij,j->ni', self.line_R_b2e[idx_int], self.lever_arm_body)
            vec_cam = np.einsum('nji,nj->ni', R_c2e, points_ecef - cam_centers)
            uv = self.cam_model.project(vec_cam)
            
            cur_idx -= (vec_cam[:, 1] / self.mean_speed) / avg_dt
            cur_idx = np.clip(cur_idx, 0, self.n_lines - 1)
            
        return np.stack([cur_idx, uv[:, 0]], axis=1)
    
    def process_tile(self, window):
        transform = self.dsm_transform
        xs = np.arange(window.col_off, window.col_off + window.width) * transform.a + transform.c
        ys = np.arange(window.row_off, window.row_off + window.height) * transform.e + transform.f
        xx, yy = np.meshgrid(xs, ys)
        
        flat_x, flat_y = xx.ravel(), yy.ravel()
        flat_z = self.dsm_interp((flat_y, flat_x))
        valid = ~np.isnan(flat_z)
        
        out_image = np.zeros((self.n_bands, window.height, window.width), dtype=self.img_data.dtype)
        if not np.any(valid): return out_image

        ecef_pts = np.stack(self.local_to_ecef.transform(flat_x[valid], flat_y[valid], flat_z[valid]), axis=1)
        img_coords = self.ground_to_image(ecef_pts)
        
        l_idx = np.round(img_coords[:, 0]).astype(int)
        s_idx = np.round(img_coords[:, 1]).astype(int)
        
        in_bounds = (l_idx >= 0) & (l_idx < self.n_lines) & (s_idx >= 0) & (s_idx < self.n_samples)
        
        valid_pixel_locs = np.where(valid)[0][in_bounds]
        
        # --- Sampling with Spectral Memmap ---
        # Shape is (Lines, Samples, Bands)
        # Pulling [N_valid_pixels, Bands]
        pixel_values = self.img_data[l_idx[in_bounds], s_idx[in_bounds], :]
        
        # Transpose to [Bands, N_valid_pixels] for easier output mapping
        pixel_values = pixel_values.T

        for b in range(self.n_bands):
            band_flat = out_image[b].ravel()
            band_flat[valid_pixel_locs] = pixel_values[b]
            out_image[b] = band_flat.reshape(window.height, window.width)
                
        return out_image


if __name__ == "__main__":
    #pipeline = OrthoPipeline("Orthorectify/ortho_config.yaml")
    
    config_path = "Orthorectify/ortho_config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Use a temporary instance just to get dimensions/metadata
    print("Pre-calculating metadata...")
    meta_pipeline = OrthoPipeline(config_path)
    num_cores = cfg['processing'].get('num_cores', 4)
    
    # Define output profile based on DSM and BIL metadata
    with rasterio.open(cfg['paths']['dsm_file']) as dsm_src:
        out_profile = dsm_src.profile.copy()
        out_profile.update(
            count=meta_pipeline.n_bands,
            dtype=meta_pipeline.img_data.dtype,
            driver='GTiff',
            nodata=0,
            blockxsize=cfg['processing'].get('block_size', 512),
            blockysize=cfg['processing'].get('block_size', 512),
            tiled=True
        )
        windows = [window for ij, window in dsm_src.block_windows()]

    # Run Parallel Processing
    print(f"Launching {num_cores} cores with {len(windows)} tiles...")
    
    with rasterio.open(cfg['paths']['output_ortho'], 'w', **out_profile) as dst:
        # Progress bar
        pbar = tqdm(total=len(windows), desc="Orthorectifying", unit="tile")
        
        # Executor
        with ProcessPoolExecutor(max_workers=num_cores, initializer=init_worker, initargs=(config_path,)) as executor:
            # Map windows to workers
            futures = {executor.submit(process_window_worker, win): win for win in windows}
            
            for future in as_completed(futures):
                window, ortho_stack = future.result()
                
                # Write to disk (Single-threaded writing is safer for GeoTIFF)
                dst.write(ortho_stack, window=window)
                pbar.update(1)
        
        pbar.close()

    print(f"\nProcessing Complete. File saved to: {cfg['paths']['output_ortho']}")