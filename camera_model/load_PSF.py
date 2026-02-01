import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt
from bisect import bisect_left
import numpy as np
import csv
import math
import pandas as pd

def calculate_focal_length(pixel_coords,angles, ppx,image_width):
    cx = ppx #bisect_left(angles, 0) + 0.5 #image_width / 2  # Assume principal point is at the center
    
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

def scaled_px_2_native_px(norm_pixels, sensor_width,ppx):
    """
    rescale pixel coordinates from (-0.5,0.5) to  (0 ,sensor width) coordinates.
    
    :param norm_pixels: Array of normalized pixel coordinates
    :param sensor_width: Width of the sensor in pixels
    :return: Array of pixel coordinates
    """
    return (norm_pixels * sensor_width) + ppx #(sensor_width/2) 

def angle_to_pixel(angle, focal_length, principal_point):
    """
    Convert an angle to a pixel coordinate.
    
    :param angle: View angle in radians
    :param focal_length: Focal length in pixels
    :param principal_point: Principal point (optical center) in pixels
    :return: Pixel coordinate
    """
    return np.tan(np.radians(angle)) * focal_length # + principal_point

def pixel_to_angle(pixel_coord, focal_length, principal_point):
    # Convert pixel coordinate to normalized image coordinate
    normalized_coord = (pixel_coord - principal_point) / focal_length
    
    # Calculate angle using atan2 


    angle = np.arctan2((pixel_coord - principal_point),focal_length)
    
    # Convert angle from radians to degrees
    angle_deg = np.degrees(angle)
    
    return angle_deg

def compute_vertical_pixel_coordinates(vertical_angles, principal_point,focal_length):
    """
    Compute vertical pixel coordinates for given vertical scan angles.
    
    :param vertical_angles: Array of vertical scan angles in radians
    :param focal_length: Focal length in pixels
    :param sensor_width: Width of the sensor in pixels
    :param principal_point: Principal point (optical center) in pixels
    :return: Array of vertical pixel coordinates
    """
    # index of hor_angles where hor_angles is minimum
    #principal_point = ppx # np.argmin(abs(hor_angles)) + 0.5

    #principal_point = 616.5 #bisect_left(hor_angles, 0) + 0.5
    dy = np.zeros(vertical_angles.shape[0])
    
    dy = angle_to_pixel(vertical_angles, focal_length, principal_point)
        #horizontal_pixels[i] = angle_to_pixel(hor_angles[i], focal_length, principal_point)
        #Normalize to image width
        #vertical_pixels[i] = (i - (sensor_width/2)) / sensor_width
        #horizontal_pixels[i] = horizontal_pixels[i] / sensor_width
    
    alta_est = pixel_to_angle(dy, focal_length, principal_point)
    
    
    return dy,alta_est

def compute_vertical_angles(dy_est,focal_length):
    """
    Compute along track angles given dy pixel coordinates.
    """

    alta_est = np.arctan2(dy_est,focal_length)
    at_deg = np.degrees(alta_est)
    
    return alta_est,at_deg

def compute_horizontal_angles( dx_est,scaled_px,sensor_width, ppx,focal_length):
    """
    Compute across track angles given dy pixel coordinates.
    """
    #px_crds = scaled_px_2_native_px(scaled_px, sensor_width,ppx)
    px_crds = np.arange(sensor_width).astype(np.float32)
    
    acta_rad = np.arctan2(dx_est+(px_crds - ppx),focal_length) 
    
   
     # Convert angle from radians to degrees
    acta_deg = np.degrees(acta_rad)
    
    return acta_rad, acta_deg

def compute_horizontal_pixel_coordinates(hor_angles, principal_point,focal_length):
    """
    Compute horizontal pixel coordinates for given horizontal scan angles.
    
    :param horizontal_angles: Array of vertical scan angles in radians
    :param focal_length: Focal length in pixels
    :param sensor_width: Width of the sensor in pixels
    :param principal_point: Principal point (optical center) in pixels
    :return: Array of vertical pixel coordinates
    """

    #principal_point = ppx #np.argmin(abs(hor_angles)) + 0.5
    #principal_point = 616.5 #bisect_left(hor_angles, 0) + 0.5
    #pixel_coords = np.arange(sensor_width)
    horizontal_angles_Est = np.zeros(hor_angles.shape[0])
    dxh = np.zeros(hor_angles.shape)
    
    pixel_indices = np.arange(hor_angles.shape[0])
    horizontal_angles_Est = pixel_to_angle(pixel_indices, focal_length, principal_point)
    dxh = angle_to_pixel(hor_angles, focal_length, principal_point) - (pixel_indices - principal_point)
    
    
    return dxh,horizontal_angles_Est

# Test
#focal_length = 2714  # focal length in pixels
#sensor_width = 1920  # sensor width in pixels
#principal_point = 959.5  # principal point (assumed to be at the center of the sensor)

# Pixels
px = np.arange(1, 1235, 1)

# Get CHB measurements
data = loadmat('/Users/jlahaye/Work/AVIRIS4/AV4_GeoProc/debug/AV4_acrosstrack_PSF_2024.mat')
angles = np.array(data['AV4_acrosstrack_PSF']['angles'])[:, 0]

#write angles to csv
np.savetxt('angles.csv', angles, delimiter=',')

fwhms = np.array(data['AV4_acrosstrack_PSF']['fwhms'])[:, 0]

# Probably it's pixels 30 to 1263 = index 29 to 1262
# Cut to 1234 pixels (pixels that get light)
xt = angles[22:1263]
# Define indices for pixels that get light
xt_idx= np.arange(22, 1263, 1)


# Along-track
a = -0.0036022
b = 0.0003831
c = 0.6025328
at = a * xt**2 + b*xt + c


pixel_coords = np.arange(len(xt))
image_width = len(xt)
angle_test = np.arange(-20.1,20.1,40.2/1280)
full_pixel_coords = ((np.arange(len(angles)) - (len(angles)/2))  / len(angles))
ppx =  np.argmin(abs(xt)) + 0.5

# plt.figure()
# plt.plot(full_pixel_coords[22:1263],angles[22:1263], label='Horizontal angle est')
# plt.plot(full_pixel_coords[22:1263],angle_test[22:1263], label='Horizontal pixel measurements')
# plt.plot(full_pixel_coords[22:1263],angles[22:1263]-angle_test[22:1263], label='Horizontal pixel measurements')
# plt.legend()

norm_pixel_coords = ((pixel_coords - (image_width/2))  / image_width)

# range of values from -21 to 20 of length 1280



# Example usage
#pixel_coords = [100, 200, 300, 400]
#angles = [10, 20, 30, 40]  # in degrees
#image_width = 800
#compute fov as difference between first and last angle in xt
# fov = xt[-1] - xt[0]
# focal_length_mean = calculate_focal_length(pixel_coords,xt,ppx,image_width)
# focal_length = compute_focal_length(fov, image_width)
# print(f"Estimated mean focal length: {focal_length_mean:.2f} pixels")
# print(f"Computed focal length: {focal_length:.2f} pixels")

# dy_fov,alta_est = compute_vertical_pixel_coordinates(at,ppx, focal_length, image_width)
# dy_mflen = compute_vertical_pixel_coordinates(at,ppx, focal_length_mean, image_width)

# acta_est,dx_fov = compute_horizontal_pixel_coordinates(xt,ppx, focal_length, image_width)
# hor_angles_mean,dx_mflen = compute_horizontal_pixel_coordinates(xt,ppx, focal_length_mean, image_width)

# a5,a4,a3,a2,a1,a0= np.polyfit(norm_pixel_coords, dy_fov, deg=5)
# b5,b4,b3,b2,b1,b0 = np.polyfit(norm_pixel_coords, dx_fov, deg=5)

# Optimized acta and alta coefficients
#Focal length: 1696.39x
#Optical-center: 624.885px
#Across-track coefficient (from order 0 to order 5): 2.52384 -73.6277 -9.07736 266.786 -19.2464 93.8364
#Along track coefficient (from order 0 to order 5): 17.8364 3.11131 -165.852 -0.323097 -52.578 1.345
shift_l = -8
shift_r = -1
shift_ppx = np.max([shift_l,shift_r])
pixel_coords_1241 = np.arange(shift_l,1234+shift_r,1)
image_width = 1234
norm_pixel_coords_1241 = ((pixel_coords_1241 - (image_width/2))  / image_width)


#opt_flen = 1696.39
#opt_ppx = 624.885
#opt_dx_poly = [1.2365546226501465,-72.47309112548828,8.986418724060059,307.9082946777344,55.518680572509766,-45.28518295288086]
#opt_dy_poly = [17.88961410522461, 2.622515916824341,-165.19052124023438,-7.4387969970703125,-54.69880676269531,17.86543083190918]

# read optimized coefficients from csv
cam_model_params = '/Volumes/fts-addlidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/Optimized_Model/Mar_25_Latest/steviapp_5th_order_xy_distortion_coefficients_1234_31_3_25.csv'

cam_model  = pd.read_csv(cam_model_params, comment='#')

opt_flen = cam_model['dx'].iloc[0]
#opt_flen = float(opt_flen)
opt_ppx = cam_model['dy'].iloc[0] - 2.2 #(shift_l-shift_r)/2

# reverse the order of the coefficients
opt_dx_poly = list(cam_model['dx'].iloc[1:])#[::-1])
opt_dy_poly = list(cam_model['dy'].iloc[1:])#[::-1])

dx_hat = np.polyval(opt_dx_poly, norm_pixel_coords_1241)
dy_hat = np.polyval(opt_dy_poly, norm_pixel_coords_1241)

# Calculate along-track angles from vertical pixel coordinates
acta_opt_rad,acta_opt_deg = compute_horizontal_angles( dx_hat,norm_pixel_coords_1241,norm_pixel_coords_1241.shape[0], opt_ppx,opt_flen)
alta_opt_rad,alta_opt_deg = compute_vertical_angles( dy_hat,opt_flen)


#a_opt,b_opt,c_opt = np.polyfit(acta_opt_rad, alta_opt_rad, deg=2)

#----------------------------------------------------------------------------------#
# Re-compute 5th order polynomial coefficients for dx and dy for 1241 sensor width #

dy_fov_1241,alta_est_1241 = compute_vertical_pixel_coordinates(alta_opt_deg,opt_ppx, opt_flen)
#dy_mflen = compute_vertical_pixel_coordinates(,ppx, focal_length_mean, image_width)

dx_fov_1241,acta_est_1241 = compute_horizontal_pixel_coordinates(acta_opt_deg,opt_ppx, opt_flen)
#hor_angles_mean,dx_mflen = compute_horizontal_pixel_coordinates(xt,ppx, focal_length_mean, image_width)

a5,a4,a3,a2,a1,a0= np.polyfit(norm_pixel_coords_1241, dy_fov_1241, deg=5)
b5,b4,b3,b2,b1,b0 = np.polyfit(norm_pixel_coords_1241, dx_fov_1241, deg=5)

# write vertical and horizontal fit polynomial coefficients to one csv with 2 columns with header (dxh,dyh) add focal length to header with # comment
# Write vertical and horizontal fit polynomial coefficients to one csv with 2 columns with header (dxh,dyh) add focal length to header with # comment
write=1
if write:
    with open('/Volumes/fts-addlidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/5th_order_xy_distortion_coefficients_sw_{:.1f}_fov_{:.2f}.csv'.format(norm_pixel_coords_1241.shape[0],opt_ppx), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'# Focal length: {opt_flen:.2f} pixels'])
        writer.writerow([f'# Principal point: {opt_ppx:.2f} pixels'])
        writer.writerow(['# order', 'dx', 'dy'])
        coefficients_b = [b5, b4, b3, b2, b1, b0]
        coefficients_a = [a5, a4, a3, a2, a1, a0]
        for i in range(6):
            writer.writerow([f'{5-i}th',coefficients_b[i], coefficients_a[i]])
    
    # with open('/Volumes/fts-addlidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/5th_order_xy_distortion_coefficients_1241_width_opt_fov_{:.2f}.csv'.format(opt_flen), 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([f'# Optimized Focal length: {opt_flen:.2f} pixels'])
    #     writer.writerow(['# order', 'dx', 'dy'])
    #     coefficients_a = opt_dx_poly
    #     coefficients_b = opt_dy_poly
    #     for i in range(6):
    #         writer.writerow([f'{5-i}th',coefficients_a[i], coefficients_b[i]])
        
    # with open('/Volumes/fts-addlidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/2nd_order_angle_distortion_coefficients_1241_sw_opt_flen_{:.2f}_ppx_{:.2f}_.csv'.format(opt_flen,ppx), 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([f'# Focal length: {opt_flen:.2f} pixels'])
    #     coefficients = [a_opt,b_opt,c_opt]
    #     writer.writerow(['#a_opt', 'b_opt', 'c_opt'])
    #     # Write the coefficients to the file with 7 decimal places
    #     writer.writerow([f'{coeff:.7f}' for coeff in coefficients])
        #np.savetxt(csvfile, np.column_stack((a_opt, b_opt,c_opt)), delimiter=',', fmt='%f')

     #write optimized angles(axta_opt and alta_opt) to csv with column header xt,at 
    with open('/Volumes/fts-addlidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/Optimized_Model/April_9_25/optimized_angles_1241_sw_fov_opt_flen_{:.2f}_ppx_{:.2f}_.csv'.format(opt_flen,opt_ppx), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['#xt(rad)', 'at(rad)'])
        # Write the acta_opt in and alta_opt to the file with 8 decimal places
        np.savetxt(csvfile, np.column_stack((-1*acta_opt_rad, alta_opt_rad)), delimiter=',', fmt='%.8f')

        #np.savetxt(csvfile, np.column_stack((acta_opt, alta_opt)), delimiter=',', fmt='%f')


        


# plot vertical pixel fit and original data
plt.figure()
plt.plot(norm_pixel_coords, dy_fov, label='Vertical pixel coordinates')
plt.plot(norm_pixel_coords, np.polyval([a5, a4, a3, a2, a1, a0],norm_pixel_coords), label='Vertical pixel fit')
plt.plot(norm_pixel_coords, dy_fov - np.polyval([a5, a4, a3, a2, a1, a0],norm_pixel_coords), label='Vertical fit diff')
#plt.plot(pixel_coords, at, label='UZH pixel fit')
plt.legend()

# plot horizontal pixel fit and original data
plt.figure()

plt.plot(norm_pixel_coords, dx_fov, label='Horizontal pixel coordinates')
plt.plot(norm_pixel_coords, np.polyval([b5, b4, b3, b2, b1, b0], norm_pixel_coords), label='Horizontal pixel fit')
plt.plot(norm_pixel_coords, dx_fov - np.polyval([b5, b4, b3, b2, b1, b0], norm_pixel_coords), label='Horizontal fit diff')
plt.legend()


plt.figure()
#plt.scatter(acta_est, alta_est,label="estimated")
plt.scatter(xt,at,label="lab angles")
plt.plot(np.linspace(-21, 21, 1000), np.polyval([a,b,c],np.linspace(-21, 21, 1000)), "r", label="opt fit")
plt.legend()

plt.figure()
#plt.scatter(acta_est, alta_est,label="estimated")
plt.scatter(acta_opt, alta_opt,label="optimized")
plt.plot(np.linspace(-21, 21, 1000), np.polyval([a_opt,b_opt,c_opt],np.linspace(-21, 21, 1000)), "r", label="opt fit")
plt.legend()

#plt.figure()
#plt.plot(norm_pixel_coords, xt-hor_angles_est, label='diff angles fov {:.2f}'.format(fov))
#plt.plot(norm_pixel_coords, xt, label='lab angles ')
#plt.plot(norm_pixel_coords, hor_angles_est, label='angles fov {:.2f}'.format(fov))
#plt.plot(norm_pixel_coords, hor_angles_mean, label='angles mean flen {:.2f}'.format(focal_length_mean))
#plt.plot(norm_pixel_coords, xt-hor_angles_mean, label='diff angles mean {:.2f}'.format(focal_length_mean))
#plt.legend()


# Print results
# for i, angle in enumerate(at):
#     print(f"Angle: {angle:.2f}°")
#     print(f"Vertical pixel coordinates: {vertical_pixels[i]/image_width:.6f} ")
#     print()

# plot at




plt.figure()
plt.plot(at)
plt.plot(xt)

# Add 3 degrees to the along-track angles
at = at + 3