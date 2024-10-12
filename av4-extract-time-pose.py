import configparser
import subprocess
import os
import pandas as pd
from scipy.interpolate import splrep, splev
import numpy as np
import glob
import argparse
from bisect import bisect_left
from pathlib import Path
from Sbet import Sbet

 
def write_csv(input_df,output_file):
    #if not os.path.exists(output_file):
    #    os.makedirs(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
            # Write the column headers to the file
            f.write('#'+'\t'.join(input_df.columns.tolist()) + '\n')
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
    
    print(f"Extracting Aviris-4 frame time stamps from: {input_file}")
    #print(f"Processed times,traj and imu line: {line.split('/')[-1]}")
    
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
    frame_times = pd.DataFrame({"frame_id": frame_numbers, "tod(10usec)": [item[0] for item in data]})
    
    return frame_times

def find_traj_index(traj_times, line_time):
	i = 0
	while traj_times[i] < line_time and i < len(traj_times):
		i += 1
	return i

def extract_line_data(in_data, extracted_times, buffer_size=100,in_type=['traj','imu']): 
    
    if in_type == 'traj':
        #time_10usec,latitude,longitude,altitude,roll,pitch,heading,'x_vel,y_vel,z_vel\n')
        input_df = pd.read_csv(in_data, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt','r','p','y', 'vel_x', 'vel_y','vel_z']) 
    elif in_type == 'imu':
        input_df = pd.read_csv(in_data, encoding='utf-8', sep='\t',comment='#',names=['time','gyro1','gyro2','gyro3','acc1','acc2','acc3','sensorStatus'])
        input_df = input_df.drop(columns=['sensorStatus'])
        input_df['time'] = input_df['time']*1e5
        
    
    # convert line_times['tod(10usec)'] to a list and covert each element to an int

    line_times = np.array(extracted_times['tod(10usec)'], dtype=np.int64)
    #line_times = [int(time) for time in line_times['GPS_sod(10usec)']]

    #Save line trajectory/imu data 
    idx_min = bisect_left(input_df['time'], line_times[0]) - 1
    idx_max = bisect_left(input_df['time'], line_times[-1])
    
    traj_min = max(idx_min - buffer_size, 0)
    traj_max = min(idx_max + buffer_size, len(input_df['time'])-1)

    line_data = input_df.loc[traj_min:traj_max] #.to_csv(f, index=False,header=False,float_format='%.6f', sep=',')

    #assert that line_times[0] > input_df['time'].iloc[idx_min] and line_times[-1] < input_df['time'].iloc[idx_max]
    # Assuming line_times is a list or a Series, and input_df is the DataFrame with the 'time' column
    assert line_times[0] > input_df['time'].iloc[idx_min], "Assertion failed: line_times[0] is not greater than input_df['time'].iloc[idx_min]"
    assert line_times[-1] < input_df['time'].iloc[idx_max], "Assertion failed: line_times[-1] is not less than input_df['time'].iloc[idx_max]"

    return line_data

def interpolate_line_poses_opt(traj, line_times):

    # Convert line_times to a NumPy array
    line_times = np.array(line_times['tod(10usec)'], dtype=np.float64)

    # Convert trajectory data to NumPy arrays
    traj_df = pd.read_csv(traj, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt','r','p','y', 'vel_x', 'vel_y','vel_z'])
    traj_times  = np.array(traj_df['time'])
    roll = np.array(traj_df['r'])
    pitch = np.array(traj_df['p'])
    yaw = np.array(traj_df['y'])
    lat = np.array(traj_df['lat'])
    lon = np.array(traj_df['lon'])
    alt = np.array(traj_df['alt'])
    
    
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
    line_poses = pd.DataFrame({"line_id": list(range(1, len(line_times) + 1)), "tod(10usec)": line_times, "roll": roll_inter, "pitch": pitch_inter, "yaw": yaw_inter, "lat": lat_inter, "lon": lon_inter, "alt": alt_inter})
    return line_poses

def interpolate_line_poses(traj, line_times): # roll, pitch, yaw, lat, lon, alt):
    # Interpolate line poses from traj using gps time stamps: assumes collumn order of traj_data is 'time', 'lat', 'lon', 'alt' (elps height), 'vx', 'vy','vz','r','p','y'
    roll_inter  = []
    pitch_inter  = []
    yaw_inter  = []
    lat_inter = []
    lon_inter = []
    alt_inter = []
    
    traj_df = pd.read_csv(traj, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt','r','p','y', 'vel_x', 'vel_y','vel_z'])
    traj_times  = traj_df['time']
    roll = traj_df['r']
    pitch = traj_df['p']
    yaw = traj_df['y']
    lat = traj_df['lat']
    lon = traj_df['lon']
    alt = traj_df['alt']

    # convert line_times['GPS_sod(10usec)'] to a list and covert each element to a float
    line_times = [int(time) for time in line_times['tod(10usec)']]

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
    line_poses = pd.DataFrame({"line_id": list(range(1, len(line_times) + 1)), "tod(10usec)": line_times, "roll": roll_inter, "pitch": pitch_inter, "yaw": yaw_inter, "lat": lat_inter, "lon": lon_inter, "alt": alt_inter})
    
    return line_poses

def av4_extract_time_pose(in_path,traj_data,imu_data=None,interp_poses = True,parse_sbet=True,sbet_deg=True,buffer_size=1000,extension=".bin",out_traj='Atlans_sbet_NED_tod_10usec.csv'):
    
    if parse_sbet:
        sbet = Sbet(traj_data,sbet_deg)
        sbet_csv_path = sbet_csv_path = traj_data.split('.')[0]+'.csv'
        print(f"Parsing SBET and saving to csv: {sbet_csv_path}")
        sbet.saveSbet2csv(sbet_csv_path)
    else:
        sbet_csv_path = traj_data
    
    in_path = Path(in_path)
    
    # check if the input path is a directory
    if not os.path.isdir(in_path):
        print(f"Input path {in_path} is not a directory")
        return
    # check if the input is readable
    if not os.access(in_path, os.R_OK):
        print(f"No read permissions for {in_path}")
        return

    #Create a list of the paths to all the raw data '.bin' files in the subdirectories of in_path
    line_files = glob.glob(os.path.join(in_path, f"**/*{extension}"), recursive=True) 
    print(f"Found {len(line_files)} line files in {in_path}")
    print(line_files)
        
    for line in line_files:
        print(f"Processing line {line.split('/')[-1]}")
        #create new 'output' directory in the line directory to store files for line times
        times_path = os.path.join(os.path.dirname(line), os.path.splitext(os.path.basename(line))[0] + '_times.csv')
        
        #check if the output directory exists, if not create it
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)
        av4_time_reader = 'extract-av4-line-times.cpp'
        line_times = AV4_parse_line_times(av4_time_reader, line)
        write_csv(line_times,times_path)
        

        # Save traj for current line times
        line_traj = extract_line_data(in_data = sbet_csv_path ,extracted_times = line_times,buffer_size=buffer_size,in_type='traj')
        traj_path = os.path.join(os.path.dirname(line), os.path.splitext(os.path.basename(line))[0] + '_traj.csv')
        write_csv(line_traj,traj_path)
        #line_traj.to_csv(os.path.join(output_dir, 'line_traj.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')

        # Save raw-imu for current line times
        if imu_data is not None:
            line_imu = extract_line_data(in_data = imu_data,extracted_times = line_times,buffer_size=buffer_size,in_type='imu')
            imu_path = os.path.join(os.path.dirname(line), os.path.splitext(os.path.basename(line))[0] + '_imu.csv')
            write_csv(line_imu,imu_path)
            #line_imu.to_csv(os.path.join(output_dir, 'line_imu.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')

        print(f"Processed times,traj and imu line: {line.split('/')[-1]}")

        # Interpolate Line Poses and save
        if interp_poses:
            line_poses = interpolate_line_poses_opt(traj=sbet_csv_path,line_times = line_times)
            imu_path = os.path.join(os.path.dirname(line), os.path.splitext(os.path.basename(line))[0] + '_imu.csv')
            write_csv(line_poses,imu_path)
            #line_poses.to_csv(os.path.join(output_dir, 'line_poses.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')

        print(f"Interpolated poses for line: {line.split('/')[-1]}")


def main(config_file):
    description = "Produce AVIRIS-4 Geo-rectification data for a given line file"
    
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    
    # Read the configuration file
    print(f"Reading configuration file: {config_file}")
    config.read(config_file)

    
    # Initialize argparse without default values from config file
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('mission_path', type=str, nargs='?', help='Path to dir containing sub-dir line files')
    parser.add_argument('trajectory_path', type=str, nargs='?', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to dir containing sub-dir line files')
    parser.add_argument('--imu_path', help='Path to the raw IMU data')
    parser.add_argument('--intp_pose', type=bool, help='Generate interpolated poses for given line times (default: True)')
    parser.add_argument('--parse_sbet', help='parse input sbet.out file (default: True)')
    parser.add_argument('--sbet_deg',default=True, type=bool,help='convert sbet to deg or leave in radians (default:True)')
    #parser.add_argument('--out_dir_name', default='geo_rect_data', help='Output directory (default: georect_data)')
    parser.add_argument('--ext',default='.bin', help='Raw data file extension (default: .bin)')
    parser.add_argument('--buffer_size',default=1000, type=int, help='Time stamp buffer beyond min/max line time stamp (default: 1000)')

    # Parse the command-line arguments (without defaults from config file)
    args = parser.parse_args()
    
    #print Keys in config file

    print(config.keys())
    print(config.sections())
    
    # Update the arguments with defaults from the config file, if not provided on the command line
    if not args.mission_path and config.has_section('PATHS') and config.has_option('PATHS', 'path_to_mission'):
        args.mission_path = config['PATHS'].get('path_to_mission')
    if not args.trajectory_path and config.has_section('PATHS') and config.has_option('PATHS', 'trajectory_path'):
        args.trajectory_path = config['PATHS'].get('trajectory_path')
    #if not args.out_dir_name and config.has_section('PATHS') and config.has_option('PATHS', 'output_dir_name'):
    #    args.out_dir_name = config['PATHS'].get('output_dir_name')
    if not args.imu_path and config.has_section('PATHS') and config.has_option('PATHS', 'imu_path'):
        args.imu_path = config['PATHS'].get('imu_path')
    if not args.intp_pose and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'interpolate_poses'):
        args.intp_pose = config['OPTIONS'].getboolean('interpolate_poses')
    if not args.parse_sbet and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'parse_sbet'):
        args.parse_sbet = config['OPTIONS'].getboolean('parse_sbet')
    if not args.sbet_deg and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'sbet_deg'):
        args.sbet_deg = config['OPTIONS'].getboolean('sbet_deg')
    if not args.ext and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'extension'):
        args.ext = config['OPTIONS'].get('extension')
    if not args.buffer_size and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'buffer_size'):
        args.buffer_size = config['OPTIONS'].getint('buffer_size')

    #Clear args.config_file
    if args.config == config_file:
        args.config_file = None


    # Check if required arguments are provided
    if not args.mission_path or not args.trajectory_path:
        print("Mission path and trajectory path must be provided either in config file or command line.")
        parser.print_help()
        exit(1)

    # Call data extraction function
    av4_extract_time_pose(
        in_path=args.mission_path,
        traj_data=args.trajectory_path,
        imu_data=args.imu_path,
        interp_poses=args.intp_pose,
        parse_sbet=args.parse_sbet,
        sbet_deg=args.sbet_deg,
        extension=args.ext,
        buffer_size=args.buffer_size
    )


if __name__ == '__main__':
    # Set up the argument parser for the config file
    parser = argparse.ArgumentParser(description='Specify the config file.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    # Parse the config file argument
    args = parser.parse_args()

    # Call the main function with the specified config file
    main('config.ini') if not args.config else main(args.config)
   
    
     

        