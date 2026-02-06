import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mat4py import loadmat
import math
from bisect import bisect_left

def compute_F_len(angles_deg, pixels):
    """
    Computes the focal length (f) and principal point (cx) given a set of 
    angles and corresponding pixel coordinates.

    Assumes a Pinhole Model: pixel = f * tan(angle) + cx

    Args:
        angles_deg (list or np.array): Angles in degrees. 
                                       (Assumes 0 is the optical axis).
        pixels (list or np.array): Corresponding pixel coordinates.

    Returns:
        tuple: (focal_length, principal_point)
    """
    # 1. Prepare Data
    theta = np.deg2rad(angles_deg)
    u = np.array(pixels)

    # 2. Linearize the problem
    # Model: u = f * tan(theta) + cx
    # This is a line equation: y = m * x + b
    # where y = u, x = tan(theta), m = f, b = cx
    x = np.tan(theta)

    # 3. Perform Linear Regression (Least Squares)
    # We define matrix A = [[tan(theta_0), 1], [tan(theta_1), 1], ...]
    A = np.vstack([x, np.ones(len(x))]).T
    
    # Solve for [f, cx]
    params, residuals, rank, s = np.linalg.lstsq(A, u, rcond=None)
    
    f_estimated = params[0]
    cx_estimated = params[1]

    return f_estimated, cx_estimated

def calculate_focal_length(pixel_coords,angles, ppx,image_width):
    cx = bisect_left(angles, 0) + 0.5 #image_width / 2  # Assume principal point is at the center
    
    focal_lengths = []
    for x, angle in zip(pixel_coords, angles):
        angle_rad = np.radians(angle)
        f = (x - cx) / np.tan(angle_rad)
        focal_lengths.append(f)
    
    return np.mean(focal_lengths)

def compute_focal_length(fov_deg, sensor_resolution):
    """
    Compute the focal length in pixels given the field of view (FOV) and sensor resolution.
    
    :param fov_deg: Field of View in degrees
    :param sensor_resolution: Resolution of the sensor in pixels (width or height, depending on FOV)
    :return: Focal length in pixels
    """
    fov_rad = math.radians(fov_deg)  # Convert FOV to radians
    focal_length = (sensor_resolution / 2) / math.tan(fov_rad / 2)
    return focal_length

def brown_distortion_model(angles_across_deg, angles_along_deg, params):
    """
    Projects 3D world angles into 2D pixel coordinates using the Brown-Conrady model.
    
    Args:
        angles_across_deg: Array of across-track angles (theta_x) in degrees.
        angles_along_deg: Array of along-track angles (theta_y) in degrees.
        params: List of parameters [f, cx, k1, k2, k3, p1, p2]
                f: Focal length (in pixels)
                cx: Principal point x-offset (in pixels)
                k1, k2, k3: Radial distortion coefficients
                p1, p2: Tangential distortion coefficients
    
    Returns:
        u_pred: Predicted x-pixel coordinates.
        v_pred: Predicted y-pixel coordinates.
    """
    f, cx, k1, k2, k3, p1, p2 = params

    # 1. Convert angles to normalized ideal image plane coordinates
    #    x = f * tan(theta) / f -> x_n = tan(theta)
    x_n = np.tan(np.deg2rad(angles_across_deg))
    y_n = np.tan(np.deg2rad(angles_along_deg))

    # 2. Calculate radius squared
    r2 = x_n**2 + y_n**2
    r4 = r2**2
    r6 = r2**3

    # 3. Calculate Radial Distortion factor
    #    (1 + k1*r^2 + k2*r^4 + k3*r^6)
    radial_factor = 1 + k1 * r2 + k2 * r4 + k3 * r6

    # 4. Calculate Tangential Distortion
    #    dx = 2*p1*x*y + p2*(r^2 + 2*x^2)
    #    dy = p1*(r^2 + 2*y^2) + 2*p2*x*y
    dx_tangential = 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
    dy_tangential = p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n

    # 5. Apply Distortions
    x_d = x_n * radial_factor + dx_tangential
    y_d = y_n * radial_factor + dy_tangential

    # 6. Project to Pixel Coordinates
    #    u = f * x_d + cx
    #    v = f * y_d + cy (For a linear array, cy is assumed 0 relative to the line center)
    u_pred = f * x_d + cx
    v_pred = f * y_d      

    return u_pred, v_pred

def residuals(params, angles_across, angles_along, pixels_observed):
    """
    Calculates the error between the model prediction and observed data.
    """
    u_pred, v_pred = brown_distortion_model(angles_across, angles_along, params)
    
    # Residual 1: Difference in Horizontal Pixel position
    res_u = u_pred - pixels_observed
    
    # Residual 2: Difference in Vertical position
    # The physical sensor is a 1D line at v=0. 
    # The lab measured an along-track angle (theta_y) for this pixel. 
    # The model must predict that this theta_y maps to v=0.
    res_v = v_pred - 0.0 
    
    # Flatten and combine residuals
    return np.concatenate((res_u, res_v))

def calibrate_linear_array(pixels_observed, angles_across_deg, angles_along_deg):
    """
    Main function to determine Brown coefficients.
    """
    # --- 1. Initial Guesses ---
    # Approximate Focal Length: (Pixel Span / 2) / tan(Half FOV)
    # Assuming center pixel is roughly at index N/2 and 0 angle
    valid_mask = ~np.isnan(pixels_observed) & ~np.isnan(angles_across_deg)
    
    # Simple linear fit for f and cx guess: pixel = f * tan(theta) + cx
    x_ideal_temp = np.tan(np.deg2rad(angles_across_deg[valid_mask]))
    pix_temp = pixels_observed[valid_mask]
    
    # slope ~ f, intercept ~ cx
    A = np.vstack([x_ideal_temp, np.ones(len(x_ideal_temp))]).T
    f_init, cx_init = np.linalg.lstsq(A, pix_temp, rcond=None)[0]
    
    #bisect_left(xt_obs, 0) + 0.5  
    # Initialize distortion coeffs to 0 (no distortion)
    # Params order: [f, cx, k1, k2, k3, p1, p2]
    x0 = [f_init, cx_init, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    print(f"Initial Guess:\n  f={f_init:.2f}, cx={cx_init:.2f}")

    # --- 2. Run Optimization ---
    # Using Levenberg-Marquardt (lm) or Trust Region Reflective (trf)
    result = least_squares(
        residuals, 
        x0, 
        args=(angles_across_deg, angles_along_deg, pixels_observed),
        method='lm' # 'lm' is robust for unconstrained problems
    )

    # --- 3. Extract Results ---
    f_opt, cx_opt, k1, k2, k3, p1, p2 = result.x
    
    return {
        "f": f_opt,
        "cx": cx_opt,
        "k1": k1, "k2": k2, "k3": k3,
        "p1": p1, "p2": p2,
        "rmse": np.sqrt(np.mean(result.fun**2))
    }

# ==========================================
# Example Usage with Mock Data
# ==========================================
if __name__ == "__main__":
    # 1. Generate Dummy Data mimicking a Linear Array
    # Sensor: 2000 pixels, ~30 degree FOV

    # Pixels
    sW = 1280
    px = np.arange(0, 1280, 1)

    # Get CHB measurements
    data = loadmat('/media/addLidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/AV4_acrosstrack_PSF_2024.mat')
    angles = np.array(data['AV4_acrosstrack_PSF']['angles'])[:, 0]

    #write angles to csv
    np.savetxt('/media/addLidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/AV4_acrosstrack_PSF_2024_angles.csv', angles, delimiter=',')

    fwhms = np.array(data['AV4_acrosstrack_PSF']['fwhms'])[:, 0]

    # plot fwhm vs pixel
    # plt.figure(figsize=(10, 5))
    # plt.plot(px, fwhms, label='FWHM (pixels)')
    # plt.xlabel('Pixel Index')
    # plt.ylabel('FWHM (pixels)')
    # plt.title('AVIRIS-4 Across Track PSF FWHM vs Pixel Index')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # select pixels with FWHM > 0
    #num_pixels = len(px)
    valid_indices = np.where(fwhms > 0.025)[0][2:-2]
    
    # plot fwhm filtered
     
    
    #plot angles vs pixel for valid indices
    # plt.figure(figsize=(10, 5))
    # plt.plot(px[valid_indices], angles[valid_indices], label='Measured Across-Track Angle (deg)', color='orange')
    # plt.xlabel('Pixel Index')
    # plt.ylabel('Angle (degrees)')
    # plt.title('AVIRIS-4 Across Track Angle vs Pixel Index (Valid Measurements)')
    # plt.legend()
    # plt.grid()
    # plt.show()  

      # Exclude edges
    FOV = 40.2 # degrees
    pixels_measured = px[valid_indices]
    xt_obs = angles[valid_indices]
    # save xt_obs to csv
    np.savetxt('/media/addLidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/AV4_acrosstrack_PSF_2024_valid_angles.csv', xt_obs, delimiter=',')

    FOV_measured = np.round(xt_obs[-1] - xt_obs[0], 2)
    num_pixels = len(valid_indices)
    #pixels_measured = np.linspace(0, 2000, num_pixels)

    f_est_1 = compute_focal_length(FOV_measured, num_pixels)
    cx_est = bisect_left(xt_obs, 0) + 0.5 
    #f_est_2 = calculate_focal_length(pixels_measured, angles_measured, ppx=num_pixels/2, image_width=num_pixels)
    f_est_3,cx_est = compute_F_len(xt_obs, pixels_measured)
    print(f"Estimated Focal Length from FOV: {f_est_1:.2f} pixels")
    print(f"Estimated Focal Length from Linear Fit: {f_est_3:.2f} pixels")
    
    # True Parameters (to see if we can recover them)
    f_true = f_est_1
    cx_true = cx_est
    k1_true = -0.02 # Barrel distortion
    p1_true = 0.001 # Slight tangential
    
    # Create "Measured" Angles (Inverse problem for data generation)
    # Ideally: pixel = f * tan(theta) + cx
    # We create angles that map to these pixels, adding the distortion effect
    x_normalized = (pixels_measured - cx_true) / f_true
    
    # Adding inverse distortion approximation to generate synthetic "observed angles"
    # (Simplified for data generation purposes)
    r2 = x_normalized**2
    x_distorted = x_normalized * (1 + k1_true * r2) 
    
    # Synthetic Across-track angles (theta_x)
    angles_x_obs = np.rad2deg(np.arctan(x_distorted))
    
    # Synthetic Along-track angles (theta_y) - "Smile" effect
    # Usually a function of x, e.g., bowing at the edges
    angles_y_obs = 0.05 * (x_normalized**2) # Parabolic smile in degrees

    # Along-track fit from DLR measurements
    # v = a * x^2 + b * x + c
    a = -0.0036022
    b = 0.0003831
    c = 0.6025328
    at_obs = a * xt_obs**2 + b*xt_obs + c
    
    # save xt_obs, at_obs to csv
    np.savetxt('/media/addLidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/AV4_acrosstrack_PSF_2024_angles_xy.csv', np.column_stack((xt_obs, at_obs)), delimiter=',', header='across_track_angle,along_track_angle', comments='')    

    # 2. Run Calibration
    print("Running Calibration...")
    coeffs = calibrate_linear_array(pixels_measured, xt_obs, at_obs)
    
    # 3. Output
    print("\nCalibration Results:")
    print("-" * 30)
    print(f"Focal Length (f): {coeffs['f']:.4f} pixels")
    print(f"Principal Point (cx): {coeffs['cx']:.4f} pixels")
    print("-" * 30)
    print(f"Radial k1: {coeffs['k1']:.6e}")
    print(f"Radial k2: {coeffs['k2']:.6e}")
    print(f"Radial k3: {coeffs['k3']:.6e}")
    print("-" * 30)
    print(f"Tangential p1: {coeffs['p1']:.6e}")
    print(f"Tangential p2: {coeffs['p2']:.6e}")
    print("-" * 30)
    print(f"RMSE (Pixel Error): {coeffs['rmse']:.6f}")

    # 1. Generate Dummy Data mimicking a Linear Array
    # Sensor: 2000 pixels, ~30 degree FOV
    num_pixels = 100
    pixels_measured = np.linspace(0, 2000, num_pixels)
    
    # True Parameters (to see if we can recover them)
    f_true = 3500.0
    cx_true = 1000.0
    k1_true = -0.02 # Barrel distortion
    p1_true = 0.001 # Slight tangential
    
    # Create "Measured" Angles (Inverse problem for data generation)
    # Ideally: pixel = f * tan(theta) + cx
    # We create angles that map to these pixels, adding the distortion effect
    x_normalized = (pixels_measured - cx_true) / f_true
    
    # Adding inverse distortion approximation to generate synthetic "observed angles"
    # (Simplified for data generation purposes)
    r2 = x_normalized**2
    x_distorted = x_normalized * (1 + k1_true * r2) 
    
    # Synthetic Across-track angles (theta_x)
    angles_x_obs = np.rad2deg(np.arctan(x_distorted))
    
    # Synthetic Along-track angles (theta_y) - "Smile" effect
    # Usually a function of x, e.g., bowing at the edges
    angles_y_obs = 0.05 * (x_normalized**2) # Parabolic smile in degrees
    
    # 2. Run Calibration
    print("Running Calibration...")
    coeffs = calibrate_linear_array(pixels_measured, angles_x_obs, angles_y_obs)

    # 3. Output
    print("\nCalibration Results:")
    print("-" * 30)
    print(f"Focal Length (f): {coeffs['f']:.4f} pixels")
    print(f"Principal Point (cx): {coeffs['cx']:.4f} pixels")
    print("-" * 30)
    print(f"Radial k1: {coeffs['k1']:.6e}")
    print(f"Radial k2: {coeffs['k2']:.6e}")
    print(f"Radial k3: {coeffs['k3']:.6e}")
    print("-" * 30)
    print(f"Tangential p1: {coeffs['p1']:.6e}")
    print(f"Tangential p2: {coeffs['p2']:.6e}")
    print("-" * 30)
    print(f"RMSE (Pixel Error): {coeffs['rmse']:.6f}")

    # 4. Optional: Validation Plot
    u_fit, v_fit = brown_distortion_model(
        angles_x_obs, angles_y_obs, 
        [coeffs['f'], coeffs['cx'], coeffs['k1'], coeffs['k2'], coeffs['k3'], coeffs['p1'], coeffs['p2']]
    )
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Across Track Fit Residuals")
    plt.scatter(pixels_measured, u_fit - pixels_measured, s=5)
    plt.xlabel("Pixel Index")
    plt.ylabel("Error (pixels)")
    
    plt.subplot(1, 2, 2)
    plt.title("Along Track 'Smile' Correction")
    plt.plot(pixels_measured, angles_y_obs, label="Measured Angle (deg)", color='orange')
    plt.axhline(0, color='k', linestyle='--', label="Ideal Line")
    plt.legend()
    plt.xlabel("Pixel Index")
    plt.show()
