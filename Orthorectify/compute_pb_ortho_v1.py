import yaml
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from liblibor.map import TangentPlane, Trajectory,Pose_std,log, loadSBET
from photogrammetry_verify import load_av4_timing
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp

# ==========================================
# 1. Camera Model (Unchanged)
# ==========================================
class CameraModel:
    def __init__(self, cfg_cam):
        self.fx = cfg_cam['focal_length_px']
        self.fy = cfg_cam['focal_length_px']
        self.cx = cfg_cam['principal_point'][0]
        self.cy = cfg_cam['principal_point'][1]
        self.width = cfg_cam['image_size'][0]
        self.height = cfg_cam['image_size'][1]
        
        # Distortion
        self.k = np.array(cfg_cam.get('k', [0,0,0]))
        self.p = np.array(cfg_cam.get('p', [0,0]))

    def project(self, points_cam):
        """Projects 3D camera-frame points to 2D pixel coordinates."""
        z = points_cam[:, 2]
        # Avoid division by zero
        z[z == 0] = 1e-6
        
        x_n = points_cam[:, 0] / z
        y_n = points_cam[:, 1] / z

        # Distortion (Brown-Conrady)
        r2 = x_n**2 + y_n**2
        r4 = r2**2
        r6 = r2**3
        rad = 1 + self.k[0]*r2 + self.k[1]*r4 + self.k[2]*r6
        
        dx = 2*self.p[0]*x_n*y_n + self.p[1]*(r2 + 2*x_n**2)
        dy = self.p[0]*(r2 + 2*y_n**2) + 2*self.p[1]*x_n*y_n
        
        x_d = x_n * rad + dx
        y_d = y_n * rad + dy
        
        u = self.fx * x_d + self.cx
        v = self.fy * y_d + self.cy
        
        return np.stack([u, v], axis=1)

# ==========================================
# 2. Orthorectification Engine (Modified)
# ==========================================
class OrthoPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.cam_model = CameraModel(self.cfg['camera'])
        self.setup_transforms()
        self.setup_mounting()
        
        # 1. Load Data & Interpolate Poses for EVERY Scanline
        self.load_and_interpolate_trajectory()
        self.load_dsm()
        
        # 2. Open Image
        self.img_path = Path(self.cfg['paths']['image_file'])
        # Shape: (Lines, Samples)
        self.img_shape = (len(self.line_times), self.cam_model.width)
        try:
            self.img_data = np.memmap(self.img_path, dtype='uint16', mode='r', shape=self.img_shape)
        except Exception as e:
            print(f"Warning: Memmap failed ({e}).")

    def setup_transforms(self):
        epsg_out = self.cfg['project'].get('epsg_out', 4979)
        self.geo_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
        self.local_to_ecef = Transformer.from_crs(f"epsg:{epsg_out}", "epsg:4978", always_xy=True)

    def setup_mounting(self):
        m = self.cfg['mounting']
        self.lever_arm_body = np.array(m['lever_arm']) # Offset in Body Frame
        
        # Rotations: Camera <- Body
        r_nominal = R.from_euler('xyz', m['nominal_mount_rpy_deg'], degrees=True).as_matrix()
        r_bore = R.from_euler('xyz', m['boresight_rpy_rad'], degrees=False).as_matrix()
        self.R_cam_to_body = r_nominal @ r_bore

    def load_and_interpolate_trajectory(self):
        """
        Loads SBET and Timing. 
        Crucial Step: Interpolates SBET attributes specifically to the exact 
        timestamp of every image scanline.
        """
        print("Loading Trajectory & Timing...")

        av4_timing = load_av4_timing(self.cfg['paths']['img_files'])
        img_times = av4_timing / 1e5  # Convert to seconds if needed

        # Define time span of images
        #img_times = np.array([timing_map[img_id]  for img_id in timing_map])
        time_buffer = 1000
        img_time_span = [img_times.min()-time_buffer, img_times.max()+time_buffer]
    
        t_start, t_end = img_time_span
        print(f"Trajectory time range: {t_start:.3f} to {t_end:.3f}")

        if self.cfg['project'].get('parse_sbet', True):
            log("[2/3] Loading SBET data...", verbose=True, force=True)
            # Extract time, lla, rpy from sbet_df
            print(f"Loading SBET from {self.cfg['paths']['sbet_file']}...")
            t,lla,rpy = loadSBET(Path(self.cfg['paths']['sbet_file']))
            mask = (t >= img_time_span[0]) & (t <= img_time_span[1])
            #tspan = t[mask]
            img_lla = lla[mask,:]
            rpy = rpy[mask,:]
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
            img_poses = trajectory.interpolate(img_times, self.cfg)
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
                self.cfg['project']['epsg'],
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

        
        # A. Load Timing (Exact UTC seconds for each line)
        # Assumes file has no header, cols: [line_index, time]
        #timing_df = pd.read_csv(self.cfg['paths']['timing_file'], sep=r'\s+', header=None, names=['id', 'time'])
        #self.line_times = timing_df['time'].values
        
        # B. Load SBET (Trajectory Source)
        # Assumes cols: time, lat, lon, height, roll, pitch, yaw
        #sbet = pd.read_csv(self.cfg['paths']['sbet_file'], sep=r'\s+', header=None, 
        #                   names=['time', 'lat', 'lon', 'height', 'roll', 'pitch', 'yaw'])
        
        # 2. Position Interpolation (Translational Manifold)
        sb_x, sb_y, sb_z = self.geo_to_ecef.transform(trajectory.lla[0,:], trajectory.lla[1,:], trajectory.lla[2,:])
        interp_pos = interp1d(trajectory.t, np.stack([sb_x, sb_y, sb_z], axis=1), axis=0, kind='cubic', fill_value="extrapolate")
        
        # # 3. Attitude Interpolation (Rotation Manifold SO(3))
        # # Construct Rotation objects from SBET RPY (Body to NED)
        # # Sequence 'xyz' or 'zyx' depends on your SBET format; 'xyz' is standard for many IMUs
        # sbet_rotations = R.from_euler('xyz', trajectory.rpy.T, degrees=True)
        
        # # Initialize Slerp
        # slerp = Slerp(trajectory.t, sbet_rotations)
        
        # # 4. Geodetic Interpolation (for local NED construction)
        # # Lat/Lon are coordinates on a sphere/ellipsoid, but for small time steps, 
        # # linear interpolation of the coordinates is acceptable for the R_ned2ecef matrix.
        # interp_geo = interp1d(trajectory.t, trajectory.lla[:2,:].T, axis=0, kind='linear', fill_value="extrapolate")

        # # 5. Pre-compute Exact Poses for Every Scanline
        # print(f"Interpolating manifold-consistent poses for {len(self.line_times)} lines...")
        
        # # Interpolate Position
        self.line_pos_ecef = interp_pos(self.line_times)
        
        # # Interpolate Attitude via Slerp
        # line_rots = slerp(self.line_times)
        # # Extract RPY back to degrees for the solver if needed, 
        # # but we will store the Rotation objects/matrices directly for speed.
        # self.line_rpy = line_rots.as_euler('xyz', degrees=True)
        
        # # Interpolate Geodetic for NED frame
        # self.line_geo = interp_geo(self.line_times)

        # # 6. Pre-calculate R_body_to_ecef for every line
        # # This avoids doing Slerp or Euler conversions inside the Newton-Raphson loop.
        # self.line_R_b2e = self.compute_all_line_rotations(line_rots, self.line_geo)
        
        # 7. Spatial Indexing
        step = 50
        self.tsi_tree = cKDTree(self.line_pos_ecef[::step])
        self.tsi_times = self.line_times[::step]

    def compute_all_line_rotations(self, line_rots, line_geo):
        """
        Computes the full Body-to-ECEF rotation matrix for every scanline.
        R_b2e = R_ned2ecef(lat, lon) * R_body2ned(interpolated)
        """
        lats = np.radians(line_geo[:, 0])
        lons = np.radians(line_geo[:, 1])
        
        sl, cl = np.sin(lats), np.cos(lats)
        slo, clo = np.sin(lons), np.cos(lons)
        
        # Construct NED to ECEF matrices (N, 3, 3)
        # Row 1: North, Row 2: East, Row 3: Down
        R_n2e = np.zeros((len(lats), 3, 3))
        R_n2e[:, 0, 0] = -sl * clo
        R_n2e[:, 0, 1] = -slo
        R_n2e[:, 0, 2] = -cl * clo
        R_n2e[:, 1, 0] = -sl * slo
        R_n2e[:, 1, 1] = clo
        R_n2e[:, 1, 2] = -cl * slo
        R_n2e[:, 2, 0] = cl
        R_n2e[:, 2, 1] = 0
        R_n2e[:, 2, 2] = -sl
        
        # R_body2ned from Slerp
        R_b2n = line_rots.as_matrix()
        
        # Final R_body2ecef
        return np.einsum('nij,njk->nik', R_n2e, R_b2n)
        
    print("Trajectory Interpolation Complete.")

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

    def get_rotation_matrices(self, att_data_batch):
        """
        Batch construct R_body_to_ecef for a set of lines.
        att_data_batch: (N, 5) -> [roll, pitch, yaw, lat, lon]
        Returns: (N, 3, 3)
        """
        roll = att_data_batch[:, 0]
        pitch = att_data_batch[:, 1]
        yaw = att_data_batch[:, 2]
        lat = att_data_batch[:, 3]
        lon = att_data_batch[:, 4]
        
        # 1. Body to NED
        r_b2n = R.from_euler('xyz', np.stack([roll, pitch, yaw], axis=1), degrees=True).as_matrix()
        
        # 2. NED to ECEF
        sl, cl = np.sin(np.radians(lat)), np.cos(np.radians(lat))
        slo, clo = np.sin(np.radians(lon)), np.cos(np.radians(lon))
        
        # Construct Rotation Matrices manually to vectorize
        # R_n2e rows:
        # [-sl*clo, -slo, -cl*clo]
        # [-sl*slo,  clo, -cl*slo]
        # [ cl,      0,   -sl    ]
        
        zero = np.zeros_like(sl)
        r_n2e = np.array([
            [-sl*clo, -slo, -cl*clo],
            [-sl*slo,  clo, -cl*slo],
            [ cl,      zero, -sl    ]
        ]).transpose(2, 0, 1) # (N, 3, 3)
        
        return np.einsum('nij,njk->nik', r_n2e, r_b2n)

    def ground_to_image(self, points_ecef):
        """
        Finds the (Line, Sample) for ground points ECEF.
        """
        n_pts = len(points_ecef)
        
        # 1. Initial Guess via KDTree
        _, idxs_sub = self.tsi_tree.query(points_ecef)
        # Map subsampled index back to full resolution index approx
        current_line_indices = idxs_sub * 50 # matches step used in load_trajectory
        
        # Clip to valid range
        current_line_indices = np.clip(current_line_indices, 0, len(self.line_times)-1)

        # 2. Iterative Refinement
        # We iterate on the INDEX, not the TIME, because we have pre-computed arrays.
        for _ in range(5):
            idx_int = current_line_indices.astype(int)
            
            # Pull pre-computed ECEF pose and Manifold-interpolated Rotation
            pos_body = self.line_pos_ecef[idx_int]
            R_b2e = self.line_R_b2e[idx_int]
            
            # Camera Center = Pos + R_b2e * Lever
            lever_rot = np.einsum('nij,j->ni', R_b2e, self.lever_arm_body)
            cam_centers = pos_body + lever_rot
            
            # Combined Rotation: Camera -> ECEF
            R_c2e = np.einsum('nij,jk->nik', R_b2e, self.R_cam_to_body)
            
            # Transform Ground Point to Camera Frame
            vec_global = points_ecef - cam_centers
            vec_cam = np.einsum('nji,nj->ni', R_c2e, vec_global) # Transpose multiply
            
            # Project to Pixels
            uv = self.cam_model.project(vec_cam) # (N, 2) [u, v]
            
            # Calculate Correction
            # In pushbroom, v is the along-track coordinate.
            # Ideally v maps to the center of the sensor (e.g. 0 or 0.5 depending on calib).
            # If v is positive, ground point is "ahead" in image => needs later time/line.
            # We assume 'cy' handles the principal point offset, so we target 0 relative to that.
            
            # Error in meters approx
            v_error_meters = vec_cam[:, 1] # Y is usually along-track in camera frame
            
            # Convert metric error to line index shift
            # Shift = Distance / (Speed * IntegrationTime)
            # But simpler: Distance / Speed -> Time Delta -> Line Delta
            dt = v_error_meters / self.mean_speed
            
            # Convert time delta to index delta
            # We estimate avg line duration
            avg_line_dur = (self.line_times[-1] - self.line_times[0]) / len(self.line_times)
            d_index = dt / avg_line_dur
            
            # Update
            # Note sign: If ground is ahead (v>0), we need to fly forward (increase index)
            current_line_indices = current_line_indices - d_index
            
            # Clamp
            current_line_indices = np.clip(current_line_indices, 0, len(self.line_times)-1)
            
            if np.mean(np.abs(d_index)) < 0.1: # Sub-pixel convergence
                break
                
        return np.stack([current_line_indices, uv[:, 0]], axis=1)

    def process_tile(self, window):
        # (Standard logic to create grid, get Z, call ground_to_image, and sample)
        # ... (Same as previous provided code)
        pass
    def process_tile(self, window):
        """
        Process a single output tile.
        """
        # 1. Create Output Grid
        transform = self.dsm_transform # Use DSM grid or define new Ortho grid
        # For simplicity, using DSM grid definition for output
        
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
        # Note: RegularGridInterpolator takes (y, x)
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
        
        # 5. Sampling (Nearest Neighbor for speed in this demo)
        lines = np.round(img_coords[:, 0]).astype(int)
        samps = np.round(img_coords[:, 1]).astype(int)
        
        # Boundary Checks
        H_img, W_img = self.img_shape
        in_bounds = (lines >= 0) & (lines < H_img) & (samps >= 0) & (samps < W_img)
        
        # 6. Fill Output Array
        out_flat = np.zeros(len(flat_x), dtype='uint16')
        
        # Fetch pixels (Vectorized fetch only works if image is in RAM, 
        # for memmap it's slow with random access. 
        # *Optimization*: In production, sort indices or read blocks.)
        if self.img_data is not None:
             valid_indices = np.where(valid_mask)[0][in_bounds]
             # This step is the bottleneck with memmap. 
             # For true speed, we would iterate over blocks of the Input Image.
             # Here we accept random access overhead.
             vals = self.img_data[lines[in_bounds], samps[in_bounds]]
             out_flat[valid_indices] = vals
             
        return out_flat.reshape(h, w)