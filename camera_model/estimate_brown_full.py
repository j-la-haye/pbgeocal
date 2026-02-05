import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def brown_model_full(params, theta_x, theta_y):
    """
    Brown-Conrady projection including Principal Point Y-offset (cy).
    """
    f, cx, cy, k1, k2, k3, p1, p2 = params
    
    # 1. Convert angles to normalized coordinates
    x_n = np.tan(np.deg2rad(theta_x))
    y_n = np.tan(np.deg2rad(theta_y))
    
    # 2. Distortion
    r2 = x_n**2 + y_n**2
    r4 = r2**2
    r6 = r2**3
    rad = (1 + k1*r2 + k2*r4 + k3*r6)
    
    # Tangential (OpenCV convention)
    dx = 2*p1*x_n*y_n + p2*(r2 + 2*x_n**2)
    dy = p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n
    
    x_dist = x_n * rad + dx
    y_dist = y_n * rad + dy
    
    # 3. Project to pixels
    u_pred = f * x_dist + cx
    v_pred = f * y_dist + cy # cy allows the 'smile' to center elsewhere
    
    return u_pred, v_pred

def objective(params, theta_x, theta_y, pix_x):
    u_pred, v_pred = brown_model_full(params, theta_x, theta_y)
    # Residuals: u matches pixel index, v matches physical center (0)
    return np.concatenate([u_pred - pix_x, v_pred - 0.0])

# --- Load and Run ---
df = pd.read_csv("/media/addLidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/AV4_acrosstrack_PSF_2024_angles_xy.csv")
pixels_x = np.arange(len(df))
across = df['across_track_angle'].values
along = df['along_track_angle'].values

# Initial Guess (Include cy)
x0 = [1750, 650, 0, 0, 0, 0, 0, 0] 

res = least_squares(objective, x0, args=(across, along, pixels_x), method='lm')

# Extract Optimized Coeffs
coeffs = dict(zip(['f', 'cx', 'cy', 'k1', 'k2', 'k3', 'p1', 'p2'], res.x))
print(f"Calibration Successful. RMSE: {np.sqrt(np.mean(res.fun**2)):.4f} pixels")
for k, v in coeffs.items():
    print(f"{k}: {v:.6e}")
