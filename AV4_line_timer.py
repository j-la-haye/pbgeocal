import subprocess
import os
import pandas as pd
from scipy.interpolate import splrep, splev
import numpy as np
import glob
import argparse
from bisect import bisect_left
from pathlib import Path

 
def write_csv(input_df,output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
            # Write the column headers to the file
            f.write('#'+'\t'.join(input_df.columns.tolist()) + '\n')
            #imu.iloc[::200,:].to_csv(f, index=False, header=False, sep='\t')
            #new_df.iloc[:, :7].round(6).to_csv(f, index=False, header=True, sep=',')
            input_df.to_csv(f, index=False,header=False,float_format='%.6f', sep=',')    

def read_file(file_path):
    with open(file_path, 'r') as file:
        values = [line.strip() for line in file.readlines()]
    return values


def get_line_files(directory,extension):
    # Function to read all .bin lines files in mission directory and its subdirectories
    # Create a list of paths to .bin files in the subdirectories of director    
    return glob.glob(os.path.join(directory, f"**/*{extension}"), recursive=True)
    
    
   
def AV4_parse_line_times(cpp_file, input_file=None):
    # Compile C++ code to parse AV4 scan line GPS time-stamps (in 10 usec of day)
    # Extract the filename without extension
    filename = os.path.splitext(cpp_file)[0]
    
    # Compile the C++ file
    compile_command = f"g++ -std=c++17 {cpp_file} -o {filename}"
    compile_process = subprocess.run(compile_command, shell=True, capture_output=True, text=True)
    
    if compile_process.returncode != 0:
        print("Compilation failed:")
        print(compile_process.stderr)
        return None
    
    print("Compilation successful.")
    
    # Run the compiled executable
    run_command = f"./{filename}"
    if input_file:
        run_command += f" {input_file}"
    
    run_process = subprocess.run(run_command, shell=True, capture_output=True, text=True)
    
    # Clean up the executable file
    os.remove(filename)
    
    if run_process.returncode != 0:
        print("Program execution failed:")
        print(run_process.stderr)
        return None
    
    # Process the output
    output_lines = run_process.stdout.strip().split('\n')
    data = [line.split() for line in output_lines]
    
    # Create a pandas DataFrame with data in one column and frame number starting from 1 to len(data) in the other column
    frame_numbers = list(range(1, len(data) + 1))
    frame_times = pd.DataFrame({"frame_id": frame_numbers, "GPS_sod(10usec)": [item[0] for item in data]})
    
    return frame_times

def find_traj_index(traj_times, line_time):
	i = 0
	while traj_times[i] < line_time and i < len(traj_times):
		i += 1
	return i

def extract_line_data(in_data, line_times, in_type): 
    
    if in_type == 'traj':
        input_df = pd.read_csv(in_data, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt', 'vx', 'vy','vz','r','p','y'])
        input_times  = input_df['time']
    elif in_type == 'imu':
        input_df = pd.read_csv(in_data, encoding='utf-8', sep='\t',comment='#',names=['time','gyro1','gyro2','gyro3','acc1','acc2','acc3','sensorStatus'])
        input_df = input_df.drop(columns=['sensorStatus'])
        input_df['time'] = input_df['time']*1e5
        
    
    # convert line_times['GPS_sod(10usec)'] to a list and covert each element to a float
    line_times = np.array(line_times['GPS_sod(10usec)'], dtype=np.int32)
    #line_times = [int(time) for time in line_times['GPS_sod(10usec)']]

    #Save line trajectory/imu data 
    idx_min = bisect_left(input_df['time'], line_times[0])
    idx_max = bisect_left(input_df['time'], line_times[-1])
    
    t_idx_min = find_traj_index(input_df['time'], line_times[0])
    t_idx_max = find_traj_index(input_df['time'], line_times[-1])

    traj_min = max(idx_min - 100, 0)
    traj_max = min(idx_max + 100, len(input_df['time'])-1)

    line_data = input_df.loc[traj_min:traj_max] #.to_csv(f, index=False,header=False,float_format='%.6f', sep=',')

    return line_data


def interpolate_line_poses_opt(traj, line_times):

    # Convert line_times to a NumPy array
    line_times = np.array(line_times['GPS_sod(10usec)'], dtype=np.float64)

    # Convert trajectory data to NumPy arrays
    traj_df = pd.read_csv(traj, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt', 'vx', 'vy','vz','r','p','y'])
    traj_times  = np.array(traj_df['time'])
    roll = np.array(traj_df['r'])
    pitch = np.array(traj_df['p'])
    yaw = np.array(traj_df['y'])
    lat = np.array(traj_df['lat'])
    lon = np.array(traj_df['lon'])
    alt = np.array(traj_df['alt'])
    
    # Precompute spline representations
    # spline_roll = splrep(traj_times, roll, k=1, s=0)
    # spline_pitch = splrep(traj_times, pitch, k=1, s=0)
    # spline_yaw = splrep(traj_times, yaw, k=1, s=0)
    # spline_lat = splrep(traj_times, lat, k=1, s=0)
    # spline_lon = splrep(traj_times, lon, k=1, s=0)
    # spline_alt = splrep(traj_times, alt, k=1, s=0)

    def find_traj_index(traj_times, target_time):
        return bisect_left(traj_times, target_time)

    # Initialize result arrays
    roll_inter = np.zeros_like(line_times)
    pitch_inter = np.zeros_like(line_times)
    yaw_inter = np.zeros_like(line_times)
    lat_inter = np.zeros_like(line_times)
    lon_inter = np.zeros_like(line_times)
    alt_inter = np.zeros_like(line_times)

    # Interpolate poses
    for i, time in enumerate(line_times):
        traj_idx = find_traj_index(traj_times, time)
        
        m = max(traj_idx - 10, 0)
        n = min(traj_idx + 10, len(traj_times)-1)
        
        roll_inter[i]= splev(time, splrep(traj_times[m:n], roll[m:n], k=1, s=0), der=0)
        pitch_inter[i] = splev(time, splrep(traj_times[m:n], pitch[m:n], k=1, s=0), der=0)
        yaw_inter[i] = splev(time, splrep(traj_times[m:n], yaw[m:n], k=1, s=0), der=0)
        lat_inter[i] = splev(time, splrep(traj_times[m:n], lat[m:n], k=1, s=0), der=0)
        lon_inter[i] = splev(time, splrep(traj_times[m:n], lon[m:n], k=1, s=0), der=0)
        alt_inter[i] = splev(time, splrep(traj_times[m:n], alt[m:n], k=1, s=0), der=0)
    
    # add line times and interpolated values to a DataFrame
    line_poses = pd.DataFrame({"line_id": list(range(1, len(line_times) + 1)), "GPS_sod(10usec)": line_times, "roll": roll_inter, "pitch": pitch_inter, "yaw": yaw_inter, "lat": lat_inter, "lon": lon_inter, "alt": alt_inter})
    return line_poses

def interpolate_line_poses(traj, line_times): # roll, pitch, yaw, lat, lon, alt):
    # Interpolate line poses from traj using gps time stamps: assumes collumn order of traj_data is 'time', 'lat', 'lon', 'alt' (elps height), 'vx', 'vy','vz','r','p','y'
    roll_inter  = []
    pitch_inter  = []
    yaw_inter  = []
    lat_inter = []
    lon_inter = []
    alt_inter = []
    
    traj_df = pd.read_csv(traj, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt', 'vx', 'vy','vz','r','p','y'])
    traj_times  = traj_df['time']
    roll = traj_df['r']
    pitch = traj_df['p']
    yaw = traj_df['y']
    lat = traj_df['lat']
    lon = traj_df['lon']
    alt = traj_df['alt']

    # convert line_times['GPS_sod(10usec)'] to a list and covert each element to a float
    line_times = [int(time) for time in line_times['GPS_sod(10usec)']]

    # Find the index of the first and last line in the trajectory data and interpolate the poses within that range
    for i in range(len(line_times)):
        traj_idx = find_traj_index(traj_times, line_times[i])
        #traj_stop = find_traj_index(traj_times, line_times[i][-1])
        
        m = max(traj_idx - 10, 0)
        n = min(traj_idx + 10, len(traj_times)-1)
        
        roll_inter.append(splev(line_times[i], splrep(traj_times[m:n], roll[m:n], k=1, s=0), der=0))
        pitch_inter.append(splev(line_times[i], splrep(traj_times[m:n], pitch[m:n], k=1, s=0), der=0))
        yaw_inter.append(splev(line_times[i], splrep(traj_times[m:n], yaw[m:n], k=1, s=0), der=0))
        lat_inter.append(splev(line_times[i], splrep(traj_times[m:n], lat[m:n], k=1, s=0), der=0))
        lon_inter.append(splev(line_times[i], splrep(traj_times[m:n], lon[m:n], k=1, s=0), der=0))
        alt_inter.append(splev(line_times[i], splrep(traj_times[m:n], alt[m:n], k=1, s=0), der=0))

	
    #round lat and lon to 14 decimal places
    lat_inter = [float(f"{val:.14f}") for val in lat_inter]
    lon_inter = [float(f"{val:.14f}") for val in lon_inter]
    # round alt to 10 decimal places
    alt_inter = [float(f"{val:.10f}") for val in alt_inter]
    # round roll, pitch, yaw to 10 decimal places
    roll_inter = [float(f"{val:.10f}") for val in roll_inter]
    pitch_inter = [float(f"{val:.10f}") for val in pitch_inter]
    yaw_inter = [float(f"{val:.10f}") for val in yaw_inter]

    # add line times and interpolated values to a DataFrame
    line_poses = pd.DataFrame({"line_id": list(range(1, len(line_times) + 1)), "GPS_sod(10usec)": line_times, "roll": roll_inter, "pitch": pitch_inter, "yaw": yaw_inter, "lat": lat_inter, "lon": lon_inter, "alt": alt_inter})
    
    return line_poses

def AV4_process_geo_files(in_path,traj_data,imu_data=None,interp_poses = True, out_dir='line_data',extension=".bin"):
    #Write an example usage for this function how to call it from the terminal 

    #Create a list of the paths to all the raw data '.bin' files in the subdirectories of in_path
    if not os.access(in_path, os.R_OK):
        print(f"No read permissions for {in_path}")
        return

    in_path = Path(in_path)

    # check if the input path is a directory
    if not os.path.isdir(in_path):
        print(f"Input path {in_path} is not a directory")
        return
    
    line_files = glob.glob(os.path.join(in_path, f"**/*{extension}"), recursive=True) 
        
    for line in line_files:
        
        #create new 'poses' directory in the line directory to store the interpolated poses
        out_dir = os.path.join(os.path.dirname(line), out_dir)
        
        #check if the output directory exists, if not create it
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        line_times = AV4_parse_line_times("extract_AV4_line_times.cpp", line)
        write_csv(line_times,os.path.join(out_dir, 'line_times.csv'))
        #Write the line times to csv with column name 'frame_time' and use '#' as a comment character to ignore header
        #line_times.to_csv(os.path.join(output_dir, 'line_times.csv'), sep=',', index=False, header=['id','frame_time'], mode='w')

        # Save traj for current line times
        line_traj = extract_line_data(in_data = traj_data,line_times = line_times,in_type='traj')
        write_csv(line_traj,os.path.join(out_dir, 'line_traj.csv'))
        #line_traj.to_csv(os.path.join(output_dir, 'line_traj.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')

        # Save raw-imu for current line times
        if imu_data is not None:
            line_imu = extract_line_data(in_data = imu_data,line_times = line_times,in_type='imu')
            write_csv(line_imu,os.path.join(out_dir, 'line_imu.csv'))
            #line_imu.to_csv(os.path.join(output_dir, 'line_imu.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')

        print(f"Processed times,traj and imu line: {line.split('/')[-1]}")

        # Interpolate Line Poses and save
        if interp_poses:
            line_poses = interpolate_line_poses_opt(traj=traj_data,line_times = line_times)
            write_csv(line_poses,os.path.join(out_dir, 'line_poses.csv'))
            #line_poses.to_csv(os.path.join(output_dir, 'line_poses.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')

        print(f"Interpolated poses for line: {line}")


def main():
    
    
    description = "AVIRIS-4 Produce Geo-Processing Data"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('line_path', help='Path to the line file')
    parser.add_argument('trajectory_path', help='Path to the trajectory file')
    parser.add_argument('imu_path', help='Path to the raw IMU data')
    parser.add_argument('--interp_poses', default=True, type=bool, help='Interpolate poses for line times (default: True)')
    parser.add_argument('--out_dir', default='line_data', help='Output directory (default: line_data)')
    parser.add_argument('--extension', default='.bin', help='File extension (default: .bin)')
   
    args = parser.parse_args()

    if len(args) < 3:
        print('#Example usage: python AV4_process_geo_files [path/to/bin-image/dir] [/path/to/trajectory_file]'\
              '[/path/to/imu_file]--interp_poses False --output_dir custom_output --extension .txt')
    

    # AV4_process_geo_files(in_path,traj_data,imu_data=None,interp_poses = True, output_dir='line_data',extension=".bin"):
    AV4_process_geo_files(
        in_path=args.line_path,
        traj_data=args.trajectory_path,
        imu_data=args.imu_path,
        interp_poses=args.interp_poses,
        out_dir=args.out_dir,
        extension=args.extension
    )


if __name__ == '__main__':
     
    main()



    #raw_image_path = '/Volumes/workspace/common/PROJECTS/AIS/AVIRIS_4/AV4_Missions/24_07_Campaigns/20-7-24-AV4Flights/M002_240720_CHE-Thun/raw_lines/raw_data/L101/101_locked'
    #traj_path= '/Volumes/workspace/common/PROJECTS/AIS/AVIRIS_4/AV4_Missions/24_07_Campaigns/20-7-24-AV4Flights/Atlans_Traj/03_processed/AV4_Thun_Test_2/Atlans_A7-20240720-100407_Thun_sbet_10usec_200Hz.csv'
    #raw_imu_path = '/Volumes/workspace/common/PROJECTS/AIS/AVIRIS_4/AV4_Missions/24_07_Campaigns/20-7-24-AV4Flights/Atlans_Traj/03_processed/AV4_Thun_20_7_24_Traj/THUN-Atlans_A7-20240720-100407_POSTPROCESSING_raw_IMU.txt'
    
    # Function to read values from a text file
    #AV4_process_geo_files(raw_image_path,traj_path,extension='.bin')


    

        