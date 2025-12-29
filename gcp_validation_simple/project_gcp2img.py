import numpy as np
from pyproj import Transformer

def reproject_3d_to_image_rad(pt_ecef, sbet_pose, K):
    """
    Reprojects an ECEF coordinate to image u,v.
    
    Args:
        pt_ecef: (3,) array [X, Y, Z] in meters
        sbet_pose: dict with 'lat', 'lon', 'alt' (radians, radians, meters) 
                   and 'roll', 'pitch', 'yaw' (radians)
        K: (3,3) Camera intrinsic matrix
    """
    # 1. Convert Camera LLA to ECEF 
    # pyproj Transformer expects degrees for EPSG:4326
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    lon_deg = np.degrees(sbet_pose['lon'])
    lat_deg = np.degrees(sbet_pose['lat'])
    
    cam_ecef = np.array(transformer.transform(lon_deg, lat_deg, sbet_pose['alt']))

    # 2. Get relative vector in ECEF frame
    v_ecef = pt_ecef - cam_ecef

    # 3. Rotate ECEF to NED (North-East-Down)
    # Uses lat/lon in radians directly
    lat = sbet_pose['lat']
    lon = sbet_pose['lon']
    
    clat, slat = np.cos(lat), np.sin(lat)
    clon, slon = np.cos(lon), np.sin(lon)
    
    # ECEF to NED Rotation Matrix
    R_ecef_to_ned = np.array([
        [-slat * clon, -slat * slon,  clat],
        [-slon,         clon,         0   ],
        [-clat * clon, -clat * slon, -slat]
    ])
    v_ned = R_ecef_to_ned @ v_ecef

    # 4. Rotate NED to Body Frame (Yaw-Pitch-Roll sequence)
    cr, sr = np.cos(sbet_pose['roll']), np.sin(sbet_pose['roll'])
    cp, sp = np.cos(sbet_pose['pitch']), np.sin(sbet_pose['pitch'])
    cy, sy = np.cos(sbet_pose['yaw']), np.sin(sbet_pose['yaw'])

    # Standard aerospace rotation matrices
    R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    
    # Combined: R_body_from_ned = R_roll * R_pitch * R_yaw
    v_body = R_roll @ R_pitch @ R_yaw @ v_ned

    # 5. Body to Camera Mount (X-right, Y-back, Z-down)
    # Mapping based on your specific mount: 
    # Camera X (right) = Body Y
    # Camera Y (down/back) = -Body X
    # Camera Z (optical axis) = Body Z
    R_body_to_cam = np.array([
        [0,  -1,  0],
        [1, 0,  0],
        [0,  0,  1]
    ])
    v_camera = R_body_to_cam @ v_body

    # 6. Perspective Projection (Pinhole Model)
    # Check if point is in front of the camera (Positive Z)
    if v_camera[2] <= 0:
        return None 
        
    # Project 3D Camera coordinates to 2D Image coordinates
    # v_camera / v_camera[2] performs the perspective divide [X/Z, Y/Z, 1]
    uv_homog = K @ (v_camera / v_camera[2])
    
    return uv_homog[0], uv_homog[1]

import numpy as np
from pyproj import Transformer
from liblibor.rotations import *

def reproject_3d_to_image_full(pt_ecef, sbet_pose, K, lever_arm, boresight):
    """
    Args:
        pt_ecef: (3,) array [X, Y, Z] in meters.
        sbet_pose: dict with 'lat', 'lon', 'alt' (rad, rad, m) and 'roll', 'pitch', 'yaw' (rad).
        K: (3,3) Camera intrinsic matrix.
        lever_arm: (3,) array [dx, dy, dz] offset from IMU to Camera in Body frame.
        boresight: (3,) array [dr, dp, dy] angular corrections in radians.
    """
    # 1. IMU LLA to ECEF
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    lon, lat,alt = sbet_pose.lla[1], sbet_pose.lla[0], sbet_pose.lla[2]
    roll,pitch,yaw = sbet_pose.rpy[0], sbet_pose.rpy[1], sbet_pose.rpy[2]
    imu_ecef = np.array(transformer.transform(np.degrees(lon), 
                                              np.degrees(lat), 
                                              alt))

    # 2. Vector in ECEF from IMU to Target
    v_ecef = pt_ecef - imu_ecef

    # 3. ECEF to NED Rotation
    #lat, lon = sbet_pose['lat'], sbet_pose['lon']
    clat, slat = np.cos(lat), np.sin(lat)
    clon, slon = np.cos(lon), np.sin(lon)
    
    R_ecef_to_ned = np.array([
        [-slat * clon, -slat * slon,  clat],
        [-slon,         clon,         0   ],
        [-clat * clon, -clat * slon, -slat]
    ])

    

    v_ned = R_ecef_to_ned @ v_ecef.reshape(3,)

    # 4. NED to IMU Body Frame
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_yaw = np.array([[cy, sy, 0], 
                      [-sy, cy, 0], 
                      [0, 0, 1]])
    R_pitch = np.array([[cp, 0, -sp], 
                        [0, 1, 0], 
                        [sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], 
                       [0, cr, sr], 
                       [0, -sr, cr]])
    R_ned_to_body = R_roll @ R_pitch @ R_yaw

    def R1(r):
        """
        Rotation matrix around the x-axis, r in radians
        """
        return np.array([[1,    0,    0],
                        [0, c(r), -s(r)],
                        [0,s(r), c(r)]])

    def R2(p):
        """
        Rotation matrix around the y-axis, p in radians
        """
        return np.array([[c(p), 0,s(p)],
                        [   0, 1,    0],
                        [-s(p), 0, c(p)]])

    def R3(y):
        """
        Rotation matrix around the z-axis, y in radians
        """
        return np.array([[ c(y), -s(y), 0],
                        [s(y), c(y), 0],
                        [    0,    0, 1]])



    Re2n = R_ned2e(lat,lon).T
    Rn2b = R_ned2b(roll,pitch,yaw)
    Rned2body=R3(yaw)@R2(pitch)@R1(roll)
    Re2b =  Rn2b @ Re2n
    R_b2e = Re2b.T
    
    #R_ned_to_body = R_ned2b(sbet_pose.rpy[0], sbet_pose.rpy[1], sbet_pose.rpy[2])
    
    # Point relative to IMU in Body Frame
    v_body_imu = R_ned_to_body @ v_ned
    #v_body_imu = Rn2b @ v_ned

    # 5. Apply Lever-Arm
    # Vector from Camera to Point = (Vector from IMU to Point) - (Vector from IMU to Camera)
    v_body_cam = v_body_imu + np.array(lever_arm)

    # 6. Apply Boresight and Mounting Rotation
    # Boresight (usually small corrections to the Body frame)
    b_r, b_p, b_y = np.radians(boresight)
    cbr, sbr = np.cos(b_r), np.sin(b_r)
    cbp, sbp = np.cos(b_p), np.sin(b_p)
    cby, sby = np.cos(b_y), np.sin(b_y)
    
    R_bore_yaw = np.array([[cby, sby, 0], [-sby, cby, 0], [0, 0, 1]])
    R_bore_pitch = np.array([[cbp, 0, -sbp], [0, 1, 0], [sbp, 0, cbp]])
    R_bore_roll = np.array([[1, 0, 0], [0, cbr, sbr], [0, -sbr, cbr]])
    R_boresight = R_bore_roll @ R_bore_pitch @ R_bore_yaw

    # Your specific Mounting: Cam_X=Body_Y, Cam_Y=-Body_X, Cam_Z=Body_Z
    R_mount = np.array([
        [0,  -1,  0],
        [1, 0,  0],
        [0,  0,  1]
    ])
    
    # R_mount = np.array([
    #     [1,  0,  0],
    #     [0, 1,  0],
    #     [0,  0,  1]
    # ])
    
    # Final Camera Frame vector
    v_camera = R_mount @ R_boresight @ v_body_cam


    # 7. Projection
    if v_camera[2] <= 0: return None
    uv_homog = K @ (v_camera / v_camera[2])
    
    return uv_homog[0], uv_homog[1]
