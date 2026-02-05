import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def brown_projection(params_lens, theta_x, theta_y):
    """
    Standard Brown-Conrady Lens Projection.
    Maps Angles -> Focal Plane Coordinates (u, v)
    """
    f, cx, cy, k1, k2, k3, p1, p2 = params_lens
    
    # 1. Angles to Ideal Image Coordinates
    x_n = np.tan(np.deg2rad(theta_x))
    y_n = np.tan(np.deg2rad(theta_y))
    
    # 2. Distortion Model
    r2 = x_n**2 + y_n**2
    r4 = r2**2
    r6 = r2**3
    rad = 1 + k1*r2 + k2*r4 + k3*r6
    
    dx = 2*p1*x_n*y_n + p2*(r2 + 2*x_n**2)
    dy = p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n
    
    x_d = x_n * rad + dx
    y_d = y_n * rad + dy
    
    # 3. Project to Pixel Plane
    u_proj = f * x_d + cx
    v_proj = f * y_d + cy
    
    return u_proj, v_proj

def smile_model(params_smile, u_norm):
    """
    Sensor Geometry Model (The "Smile").
    Describes the vertical offset of the sensor array as a function of horizontal pixel index.
    """
    s0, s1, s2, s3, s4 = params_smile
    return s0 + s1*u_norm + s2*u_norm**2 + s3*u_norm**3 + s4*u_norm**4

def residuals_separated(params_all, theta_x, theta_y, pixels, u_norm):
    # Split parameters
    params_lens = params_all[:8]   # f, cx, cy, k...
    params_smile = params_all[8:]  # s0, s1...
    
    # 1. Project observed angles through the lens model
    u_pred, v_pred = brown_projection(params_lens, theta_x, theta_y)
    
    # 2. Calculate the "True" physical position of the sensor pixel
    #    Instead of assuming the sensor is at v=0, we calculate its smile position.
    v_sensor_pos = smile_model(params_smile, u_norm)
    
    # 3. Calculate Residuals
    res_u = u_pred - pixels          # Across-track error
    res_v = v_pred - v_sensor_pos    # Along-track error (Lens Proj vs Sensor Shape)
    
    return np.concatenate([res_u, res_v])

# --- Main Execution ---
if __name__ == "__main__":
    # Load Data
    df = pd.read_csv("/media/addLidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/AV4_acrosstrack_PSF_2024_angles_xy.csv")
    pixels = np.arange(len(df))
    theta_x = df['across_track_angle'].values
    theta_y = df['along_track_angle'].values
    
    # Normalize pixel index for stable polynomial fitting
    u_norm = (pixels - np.mean(pixels)) / np.std(pixels)

    # Initial Guesses
    # Lens: [f, cx, cy, k1, k2, k3, p1, p2]
    # Note: cy and s0 are correlated (vertical shift). The optimizer will distribute them.
    x0_lens = [1750.0, 650.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Smile: [s0, s1, s2, s3, s4]
    x0_smile = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    res = least_squares(
        residuals_separated, 
        x0_lens + x0_smile, 
        args=(theta_x, theta_y, pixels, u_norm), 
        method='lm'
    )

    # Output
    rmse = np.sqrt(np.mean(res.fun**2))
    print(f"Total RMSE: {rmse:.6f} pixels")
    
    p_lens = res.x[:8]
    p_smile = res.x[8:]
    
    print("\n--- Calibration Coefficients ---")
    print(f"Focal Length: {p_lens[0]:.4f}")
    print(f"Principal Point: ({p_lens[1]:.4f}, {p_lens[2]:.4f})")
    print(f"Radial Dist: k1={p_lens[3]:.4e}, k2={p_lens[4]:.4e}, k3={p_lens[5]:.4e}")
    print(f"Tangential: p1={p_lens[6]:.4e}, p2={p_lens[7]:.4e}")
    print("\n--- Sensor Smile Model (Poly Coeffs) ---")
    print(f"S0..S4: {p_smile}")
