import yaml
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from pyproj import Transformer, CRS
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class OrthoEngine:
    def __init__(self, trajectory_data, camera_params, mount_params, dsm_path, bil_path):
        """
        trajectory_data: dict with 'time', 'lat', 'lon', 'height', 'roll', 'pitch', 'yaw'
        camera_params: dict with 'f', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2' (Brown Model)
        mount_params: {'boresight_rot': 3x3 array, 'lever_arm': 3x1 array}
        """
        self.traj = trajectory_data
        self.cam = camera_params
        self.mount = mount_params
        
        # Setup Coordinate Transformers
        self.geo_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
        self.ecef_to_local = Transformer.from_crs("epsg:4978", "epsg:32633", always_xy=True) # Example UTM Zone

        # Interpolate Trajectory (Time-based)
        self.interp_pos = interp1d(self.traj['time'], 
                                   self.geo_to_ecef.transform(self.traj['lon'], self.traj['lat'], self.traj['height']), 
                                   axis=1, kind='cubic')
        self.interp_att = interp1d(self.traj['time'], 
                                   np.stack([self.traj['roll'], self.traj['pitch'], self.traj['yaw']], axis=1), 
                                   axis=0, kind='cubic')

        # Load DSM
        with rasterio.open(dsm_path) as src:
            self.dsm_data = src.read(1)
            self.dsm_transform = src.transform
            self.dsm_res = src.res[0]
            # Create interpolator for Z lookups
            x = np.linspace(src.bounds.left, src.bounds.right, src.width)
            y = np.linspace(src.bounds.top, src.bounds.bottom, src.height)
            self.dsm_interp = RegularGridInterpolator((y[::-1], x), self.dsm_data[::-1, :], bounds_error=False, fill_value=0)

    def apply_distortion(self, u_norm, v_norm):
        """Apply Brown-Conrady Distortion Model"""
        r2 = u_norm**2 + v_norm**2
        k1, k2 = self.cam['k1'], self.cam['k2']
        p1, p2 = self.cam['p1'], self.cam['p2']
        
        # Radial
        dr = (1 + k1*r2 + k2*r2**2)
        # Tangential
        dt_u = p1 * (r2 + 2*u_norm**2) + 2*p2*u_norm*v_norm
        dt_v = p2 * (r2 + 2*v_norm**2) + 2*p1*u_norm*v_norm
        
        return u_norm * dr + dt_u, v_norm * dr + dt_v

    def find_scanline_time(self, ground_ecef, initial_time_guess):
        """
        Iteratively find the exact time a ground point was imaged.
        For pushbroom, the 'v' (along-track) coordinate must be effectively 0 in camera space.
        """
        t = initial_time_guess
        for _ in range(5): # Newton-Raphson or simple iterative refinement
            pos_ecef = self.interp_pos(t)
            att_ned = self.interp_att(t)
            
            # Rotation: Body to ECEF
            r_body_to_ecef = R.from_euler('xyz', att_ned, degrees=True).as_matrix()
            # Rotation: Camera to ECEF (includes boresight)
            r_cam_to_ecef = r_body_to_ecef @ self.mount['boresight_rot']
            
            # Vector from sensor to ground in Camera Frame
            lever_arm_ecef = r_body_to_ecef @ self.mount['lever_arm']
            vec_ecef = ground_ecef - (pos_ecef + lever_arm_ecef)
            vec_cam = r_cam_to_ecef.T @ vec_ecef
            
            # In pushbroom, the target scanline is where the y-offset in camera space is 0
            # We adjust t based on the along-track velocity
            # (Simplified: logic depends on flight speed and frame rate)
            t_offset = vec_cam[1] / 50.0 # 50m/s approx speed
            t += t_offset
            if abs(t_offset) < 1e-6: break
        return t, vec_cam

    def process_tile(self, window):
        """Processes a single spatial tile of the output GeoTIFF"""
        # 1. Define local grid for the tile
        rows = np.arange(window.row_off, window.row_off + window.height)
        cols = np.arange(window.col_off, window.col_off + window.width)
        
        # 2. Map pixels to Local CRS (UTM) -> find Z from DSM
        # 3. For each pixel: find_scanline_time -> apply_distortion -> Sample BIL
        # (Implementation omitted for brevity: involves nested loops or vectorized blocks)
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

def main_pipeline():
    # Setup parallel processing
    num_workers = mp.cpu_count()
    # Define output tiles using rasterio windows
    # ...
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map process_tile across all windows
        pass

# ==========================================
# 3. Main Entry Point
# ==========================================
def main():
    config_path = "config.yaml"
    pipeline = OrthoEngine(config_path)
    
    # Define Output Window (Whole DSM or Tiles)
    # Example: Process center 1000x1000 crop
    w = Window(col_off=0, row_off=0, width=1000, height=1000)
    
    # Run
    print("Processing Tile...")
    result = pipeline.process_tile(w)
    
    # Save
    with rasterio.open(pipeline.cfg['paths']['output_ortho'], 'w', 
                       driver='GTiff', height=w.height, width=w.width, 
                       count=1, dtype='uint16', crs=pipeline.dsm_src.crs, 
                       transform=pipeline.dsm_transform) as dst:
        dst.write(result, 1)
    print("Done.")

if __name__ == "__main__":
    main()
