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
from AV4EstimateTimes import load_frame_times
import struct
import re
import matplotlib.pyplot as plt
from plot_sbet import plot_2d_osm_map, plot_2d_osm_altitude

def write_csv(input_df,output_file,input_type=['traj','imu','gps','times']):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w', encoding='utf-8') as f:

            if input_type == 'imu' or input_type == 'gps':
                places = 6

            if input_type == 'times':
                places = 0
            else:
                places = 14
            
            if input_type == 'times':
                for i in range(len(input_df)):
                    f.write(f"{float(input_df.iloc[i].iloc[1]):.{places}f}" + '\n')
            else:
                # Write the column headers to the file
                f.write('#' + ','.join(input_df.columns.tolist()) + '\n')
                # Write all columns with string format 6 decimal places except the first column with one decimal place
                for i in range(len(input_df)):
                    f.write(f"{float(input_df.iloc[i].iloc[0]):.1f}" + ''.join([f",{float(value):.{places}f}" for value in input_df.iloc[i].iloc[1:3]]) + ''.join([f",{float(value):.6f}" for value in input_df.iloc[i].iloc[3:]]) + '\n')
            #input_df.to_csv(f, index=False,header=False,float_format='%.6f', sep=',')    

def write_poses_csv(input_df,output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w', encoding='utf-8') as f:
            # Write the column headers to the file
            f.write('#' + ','.join(input_df.columns.tolist()) + '\n')
            # Write all columns with string format 6 decimal places except the first column with one decimal place
            for i in range(len(input_df)):
                f.write(f"{input_df.iloc[i].iloc[0]:.0f}" +''.join(f",{input_df.iloc[i].iloc[1]:.1f}") + ''.join([f",{float(value):.6f}" \
                for value in input_df.iloc[i].iloc[2:5]]) + ''.join([f",{float(value):.14f}" for value in input_df.iloc[i].iloc[5:7]]) + ''.join(f",{input_df.iloc[i].iloc[7]:.3f}") +'\n')   

def read_file(file_path):
    with open(file_path, 'r') as file:
        values = [line.strip() for line in file.readlines()]
    return values


def get_line_files(directory,extension):
    # Function to read all .bin lines files in mission directory and its subdirectories
    # Create a list of paths to .bin files in the subdirectories of director    
    return glob.glob(os.path.join(directory, f"**/*{extension}"), recursive=True)

def ParseCreateTime(file_path):
    data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 9:  # Ensure we have all expected parts
                filename = parts[0]
                moddate = parts[2]
                modtime = parts[3]
                birthdate = parts[6]
                birthtime = parts[7]
               
                #modify_date = ' '.join(parts[2:4])
                #birth_date = ' '.join(parts[6:8])
                
                # Extract line number
                line_match = re.search(r'Line_(\d+)', filename)
                line_number = line_match.group(1) if line_match else None
                utc_offset = int(parts[8][2])
                
                # Check if it's an all_frames file
                all_frames = 'all_frames' in filename
                
                # Parse dates
                #modify_datetime = datetime.strptime(modify_date, '%Y-%m-%d %H:%M:%S')
                #birth_datetime = datetime.strptime(birth_date, '%Y-%m-%d %H:%M:%S')
                
                data.append([filename,  birthdate,birthtime, line_number,utc_offset])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['filename', 'birthdate','birthtime',  'line_number','utc_offset'])
    
    # Convert line_number to numeric, coercing invalid values to NaN
    df['line_number'] = pd.to_numeric(df['line_number'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['birthdate'] + ' ' + df['birthtime'])
    #correct for UTC offset
    df['datetime'] = df['datetime'] - pd.to_timedelta(df['utc_offset'], unit='h')
                                

    # Ensure datetime has nanosecond precision
    #df['zerotime'] = df['datetime'].replace(hour=0, minute=0, second=0, microsecond=0)

    df['tod'] = df['datetime'].apply(lambda x: f"{((x - x.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() + x.nanosecond / 1e9):.9f}")
    #df['tod'] = df['tod'].map(lambda x: f"{x:.9f}")

    # Drop unnecessary columns
    #df = df.drop(['permissions', 'links', 'owner', 'group', 'date', 'file_time', 'timezone', 'all_frames'], axis=1)

    # Reorder columns
    df = df[['filename', 'datetime','line_number', 'tod']]

    return df
    
def AV4EstimateLineTimes(frame_file_path,creation_times_path):

    # Constants
    data_t = 'H'  # Unsigned short (16-bit integer)
    aviris4img_channels = 327  # Does not include the band with time tags
    aviris4img_resolution = 1280
    aviris4img_headerlinelen = aviris4img_resolution * struct.calcsize(data_t)
    aviris4img_linelen = aviris4img_resolution * (aviris4img_channels + 1) * struct.calcsize(data_t)
    aviris4img_linedatalen = aviris4img_resolution * aviris4img_channels * struct.calcsize(data_t)
    sysTimeOffset = 0

    # Check file size and existence
    if not os.path.exists(frame_file_path):
        return []

    file_size = os.path.getsize(frame_file_path)

    if file_size % aviris4img_linelen != 0:
        return []  # Unexpected file size

    n_lines = file_size // aviris4img_linelen

    # Structure to hold line timing info
    class LineTimingInfos:
        def __init__(self, internal_time):
            self.internal_time = internal_time

    info = [None] * n_lines
    ret = [0] * n_lines

    #CubeCreationTime = ParseCreateTime(creation_times_path)
    #CubeCreationTime = pd.read_csv(creation_times_path,comment='#',header=None, delimiter = ',',names=['filename', 'datetime','line_number', 'tod'])

    # find CubeCreationTime['tod'] of current frame_file_path
    line_creation_tod = int(creation_times_path * 1e4) #float(CubeCreationTime['tod'].loc[CubeCreationTime['filename'] == frame_file_path.split('/')[-1]].values[0])
       
    with open(frame_file_path, 'rb') as f:
        header_data = bytearray(aviris4img_headerlinelen)
        # Read all header lines at once
        for i in range(n_lines):
            # Read the header line
            f.seek(i * aviris4img_linelen)
            f.readinto(header_data)

            # Extract various data from header
            line_internal_time = struct.unpack_from('<I', header_data, sysTimeOffset)[0]  # Little endian 4 bytes
            
            # Store the timing info
            info[i] = LineTimingInfos(line_internal_time)
    # Calculate the final times
    for i in range(n_lines):
        delta_t = info[i].internal_time - info[0].internal_time
        ret[i] = (line_creation_tod*10  + delta_t)
    frame_times = pd.DataFrame({"tod(10usec)": [f"{x:.0f}" for x in ret]})
    return frame_times 
   
def AV4_parse_line_times(cpp_file, input_file=None):
    # Compile C++ code to parse AV4 scan line GPS time-stamps (in 10 usec of day)
    # Extract the filename without extension
    filename = os.path.splitext(cpp_file)[0]
    
    # Compile the C++ file
    compile_command = f"clang++ -std=c++17 {cpp_file} -o {filename}"
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
        input_df = pd.read_csv(in_data, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt','r','p','hd', 'vel_x', 'vel_y','vel_z']) 
        if (input_df['time'] < 1e6).all():
            input_df['time'] = input_df['time']*1e5
    elif in_type == 'imu':
        
        input_df = pd.read_csv(in_data, encoding='utf-8', sep=',',comment='#',names=['time','gyro1','gyro2','gyro3','acc1','acc2','acc3'])
        #input_df = input_df.drop(columns=['sensorStatus'])
        freq = round(1/(np.mean(np.diff(input_df['time']))))
        input_df.iloc[:,1:7] = input_df.iloc[:,1:7].div(1/freq).round(6)
        if (input_df['time'] < 1e6).all():
            input_df['time'] = input_df['time']*1e5
        
            
    
    # convert line_times['tod(10usec)'] to a list and covert each element to an int

    line_times = np.array(extracted_times['tod(10usec)'], dtype=np.int64)
    #line_times = [int(time) for time in line_times['GPS_sod(10usec)']]

    

    idx_min = bisect_left(input_df['time'], line_times[0]) 
    idx_max = bisect_left(input_df['time'], line_times[-1]) 

    #Save line trajectory/imu data
    if in_type == 'traj' and idx_max < len(input_df['time']):
        dt = 1 
        idx_min -= dt
        idx_max += dt  
    elif in_type == 'imu' and idx_max < len(input_df['time']):
        dt = 2
        idx_min -= dt
        idx_max += dt 
    
    # Assert that idx_min and idx_max are within the bounds of the input data

    assert idx_min >= buffer_size, "Assertion failed: idx_min is less than 0"
    assert idx_max > buffer_size, "Assertion failed: idx_max is less than 1"
    
    data_min = max(idx_min - buffer_size, 0)
    data_max = min(idx_max + buffer_size, len(input_df['time'])-1)

    line_data = input_df.loc[data_min:data_max] #.to_csv(f, index=False,header=False,float_format='%.6f', sep=',')

    #assert that line_times[0] > input_df['time'].iloc[idx_min] and line_times[-1] < input_df['time'].iloc[idx_max]
    # Assuming line_times is a list or a Series, and input_df is the DataFrame with the 'time' column
    assert line_times[0] >= input_df['time'].iloc[data_min], "Assertion failed: line_times[0] is not greater than input_df['time'].iloc[data_min]"
    assert line_times[-1] <= input_df['time'].iloc[data_max], "Assertion failed: line_times[-1] is not less than input_df['time'].iloc[idx_max]"

    return line_data

def interpolate_line_poses_opt(traj, line_times):

    # Convert line_times to a NumPy array
    line_times = np.array(line_times['tod(10usec)'], dtype=np.float64)

    # Convert trajectory data to NumPy arrays
    traj_df = pd.read_csv(traj, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt','r','p','y', 'vel_x', 'vel_y','vel_z'])
    # drop velocity columns
    traj_df = traj_df.drop(columns=['vel_x', 'vel_y','vel_z'])

    #  select only unique rows of traj_df
    traj_df = traj_df.drop_duplicates(subset=['time'],keep='first')

    traj_times  = np.array(traj_df['time'])
    roll = np.array(traj_df['r'])
    pitch = np.array(traj_df['p'])
    head = np.array(traj_df['y'])
    lat = np.array(traj_df['lat'])
    lon = np.array(traj_df['lon'])
    alt = np.array(traj_df['alt'])
    
    
    def find_traj_index(traj_times, target_time):
        return bisect_left(traj_times, target_time)

    # Initialize result arrays
    roll_inter = np.zeros_like(line_times)
    pitch_inter = np.zeros_like(line_times)
    head_inter = np.zeros_like(line_times)
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
        head_inter[i] = splev(time, splrep(traj_times[m:n], head[m:n], k=1, s=0), der=0)
        lat_inter[i] = splev(time, splrep(traj_times[m:n], lat[m:n], k=1, s=0), der=0)
        lon_inter[i] = splev(time, splrep(traj_times[m:n], lon[m:n], k=1, s=0), der=0)
        alt_inter[i] = splev(time, splrep(traj_times[m:n], alt[m:n], k=1, s=0), der=0)

        # assert that the interpolated values are not NaN
        assert not np.isnan(roll_inter[i]), "Assertion failed: roll_inter[i] is NaN"
    
    # add line times and interpolated values to a DataFrame
    line_poses = pd.DataFrame({"line_id": list(range(1, len(line_times) + 1)), "tod(10usec)": line_times, "roll": roll_inter, "pitch": pitch_inter, "heading": head_inter, "lat": lat_inter, "lon": lon_inter, "alt": alt_inter})
    return line_poses

def interpolate_line_poses(traj, line_times): # roll, pitch, yaw, lat, lon, alt):
    # Interpolate line poses from traj using gps time stamps: assumes collumn order of traj_data is 'time', 'lat', 'lon', 'alt' (elps height), 'vx', 'vy','vz','r','p','y'
    roll_inter  = []
    pitch_inter  = []
    head_inter  = []
    lat_inter = []
    lon_inter = []
    alt_inter = []
    
    traj_df = pd.read_csv(traj, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt','r','p','y', 'vel_x', 'vel_y','vel_z'])
    traj_times  = traj_df['time']
    roll = traj_df['r']
    pitch = traj_df['p']
    head = traj_df['y']
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
        head_inter.append(splev(line_times[i], splrep(traj_times[m:n], head[m:n], k=1, s=0), der=0))
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
    head_inter = [float(f"{val:.10f}") for val in head_inter]

    # add line times and interpolated values to a DataFrame
    line_poses = pd.DataFrame({"line_id": list(range(1, len(line_times) + 1)), "tod(10usec)": line_times, "roll": roll_inter, "pitch": pitch_inter, "head": head_inter, "lat": lat_inter, "lon": lon_inter, "alt": alt_inter})
    
    return line_poses

def av4_extract_time_pose(in_path,traj_data,imu_data=None,interp_poses = True,parse_sbet=True,sbet_deg=True,LineCreationTimes=None,buffer_size=1000,extension=".bin",out_traj='Atlans_sbet_NED_tod_10usec.csv'):
    
    if parse_sbet:
        sbet = Sbet(traj_data,sbet_deg)
        sbet_csv_path = sbet_csv_path = traj_data.split('.')[0]+'.csv'
        print(f"Parsing SBET and saving to csv: {sbet_csv_path}")
        sbet.saveSbet2csv(sbet_csv_path)
    else:
        sbet_csv_path = traj_data

    # Plot 2D SBET trajectory for verification
    print("SBET parsing complete, plotting SBET trajectory for verification")
    plot_2d_osm_altitude(sbet_csv_path, zoom=10)

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
        times_path = os.path.join(os.path.dirname(line), os.path.splitext(os.path.basename(line))[0] + '.times')
        
        av4_time_reader = 'ExtractAV4LineTimes.cpp'
        if LineCreationTimes is None:
            line_times = AV4_parse_line_times(av4_time_reader, line)
        else:
            line_times = AV4EstimateLineTimes(line,LineCreationTimes)
            # Plot line times to ensure times increment linearly
            #import matplotlib.pyplot as plt
        
        # Compute delta time between line times
        computed_delta = np.diff(line_times['tod(10usec)'].astype(np.float64))
        print(f"Computed delta time for line {line.split('/')[-1]}: {np.median(computed_delta)} usec")
        #print first line_times
        print(line_times.head())
        
        #plot the computed delta time
        plt.plot(computed_delta)
        plt.xlabel('Line Number')
        plt.ylabel('Delta Time (10 usec)')
        #plt.title(f"Computed Delta Time for {line.split('/')[-1]}")
        #plt.show()


        write_csv(line_times,times_path,input_type='times')
        

        # Save traj for current line times
        #line_traj = extract_line_data(in_data = sbet_csv_path ,extracted_times = line_times,buffer_size=buffer_size,in_type='traj')
        #traj_path = os.path.join(os.path.dirname(line), os.path.splitext(os.path.basename(line))[0] + '_traj.csv')
        #write_csv(line_traj,traj_path,input_type='traj')

        # Save raw-imu for current line times
        # if imu_data is not None:
        #     line_imu = extract_line_data(in_data = imu_data,extracted_times = line_times,buffer_size=buffer_size,in_type='imu')
        #     imu_path = os.path.join(os.path.dirname(line), os.path.splitext(os.path.basename(line))[0] + '_imu.csv')
        #     write_csv(line_imu,imu_path,input_type='imu')
        #     #line_imu.to_csv(os.path.join(output_dir, 'line_imu.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')

        #     print(f"Processed times,traj and imu line: {line.split('/')[-1]}")

        # # Interpolate Line Poses and save
        # if interp_poses:
        #     line_poses = interpolate_line_poses_opt(traj=traj_path,line_times = line_times)
        #     poses_path = os.path.join(os.path.dirname(line), os.path.splitext(os.path.basename(line))[0] + '_poses.csv')
        #     write_poses_csv(line_poses,poses_path,input_type='pose')
        #     #line_poses.to_csv(os.path.join(output_dir, 'line_poses.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')

        #     print(f"Interpolated poses for line: {line.split('/')[-1]}")


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
    parser.add_argument('--ext',default=None, help='Raw data file extension (default: .bin)')
    parser.add_argument('--buffer_size',default=1000, type=int, help='Time stamp buffer beyond min/max line time stamp (default: 1000)')
    parser.add_argument('--LineCreationTimes',default=None, type=str, help='File creation times for non-time stamped .bin files')

    # Parse the command-line arguments (without defaults from config file)
    args = parser.parse_args()
    
    # Update the arguments with defaults from the config file, if not provided on the command line
    if not args.mission_path and config.has_section('PATHS') and config.has_option('PATHS', 'path_to_mission'):
        args.mission_path = config['PATHS'].get('path_to_mission')
    if not args.trajectory_path and config.has_section('PATHS') and config.has_option('PATHS', 'trajectory_path'):
        args.trajectory_path = config['PATHS'].get('trajectory_path')
    if not args.imu_path and config.has_section('PATHS') and config.has_option('PATHS', 'imu_path'):
        args.imu_path = config['PATHS'].get('imu_path')
    if not args.intp_pose and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'interpolate_poses'):
        args.intp_pose = config['OPTIONS'].getboolean('interpolate_poses')
    if not args.parse_sbet and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'parse_sbet'):
        args.parse_sbet = config['OPTIONS'].getboolean('parse_sbet')
    if not args.sbet_deg and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'sbet_deg'):
        args.sbet_deg = config['OPTIONS'].getboolean('sbet_deg')
    if not args.ext and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'raw_data_extension'):
        args.ext = config['OPTIONS'].get('raw_data_extension')
    if not args.buffer_size and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'buffer_size'):
        args.buffer_size = config['OPTIONS'].getint('buffer_size')
    #if not args.LineCreationTimes and config.has_section('PATHS') and config.has_option('PATHS', 'LineCreationTimes'):
    #    args.LineCreationTimes = config['PATHS'].get('LineCreationTimes')

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
        buffer_size=args.buffer_size, 
        #LineCreationTimes=float(args.LineCreationTimes)
    )


if __name__ == '__main__':
    # Set up the argument parser for the config file
    parser = argparse.ArgumentParser(description='Specify the config file.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    # Parse the config file argument
    args = parser.parse_args()

    # Call the main function with the specified config file
    main('config/av4-extract-time-pose.ini') if not args.config else main(args.config)
   
    
     

        