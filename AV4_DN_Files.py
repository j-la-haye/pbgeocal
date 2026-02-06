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
#from Sbet import Sbet
#from AV4EtimateTimes import load_frame_times
import struct
import re
import csv
from math import pi

def write_csv(input_df,output_file,in_type=['traj','imu','gps']):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w', encoding='utf-8') as f:
            # Write the column headers to the file
            #f.write('#' + ','.join(input_df.columns.tolist()) + '\n')
            # Write all columns with string format 6 decimal places except the first column with one decimal place
            if in_type == 'traj' or in_type == 'gps':
                 places = 14
            else:
                 places = 6
            for i in range(len(input_df)):
                f.write(f"{float(input_df.iloc[i].iloc[0]):.6f}" + ''.join([f",{float(value):.{places}f}" for value in input_df.iloc[i].iloc[1:]]) + '\n')
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


def steviapp_tps(input_csv, output_txt):
    with open(input_csv, 'r') as csv_file, open(output_txt, 'w') as txt_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            lat = row['Y']
            lon = row['X']
            description = row['desc.']
            txt_file.write(f"GEOXY EPSG:4326 {lat} {lon} ,PRIORID {description}\n")

def find_traj_index(traj_times, line_time):
	i = 0
	while traj_times[i] < line_time and i < len(traj_times):
		i += 1
	return i

def extract_data(in_data,drop_gps_SD=True,in_type=['traj','imu','gps']): 
    
    if in_type == 'gps':
        #time_10usec,latitude,longitude,altitude,roll,pitch,heading,'x_vel,y_vel,z_vel\n')
        input_df = pd.read_csv(in_data, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt','sdN','sdE','sdHt'])
        #drop the standard deviations
        if drop_gps_SD:
            input_df = input_df.drop(['sdN','sdE','sdHt'], axis=1) 
    elif in_type == 'imu':
        input_df = pd.read_csv(in_data, encoding='utf-8', sep='   ',names=['time','gyro1','gyro2','gyro3','acc1','acc2','acc3'],header=0)
        #freq = round(1/(np.mean(np.diff(input_df['time']))))
        #input_df.iloc[:,1:7] = input_df.iloc[:,1:7].div(1/freq).round(6)
        # First convert strings to float
        input_df.iloc[:, 1:4] = input_df.iloc[:, 1:4].astype(float)

#        Then convert to radians
        input_df.iloc[:, 1:4] = np.radians(input_df.iloc[:, 1:4])
        
        input_df.iloc[:,1] = float(input_df.iloc[:,1]).div(pi/180).round(6)
        input_df.iloc[:,2] = input_df.iloc[:,2].astype(float).div(pi/180).round(6)
        input_df.iloc[:,3] = input_df.iloc[:,3].astype(float).div(pi/180).round(6)
        #input_df['time'] = input_df['time']
    elif in_type == 'traj':
         input_df = pd.read_csv(in_data, encoding='utf-8', sep=',',comment='#',names=['time', 'lat', 'lon', 'alt','r','p','hd', 'vel_x', 'vel_y','vel_z'])
         #keep only the time and position columns to fake gps
         input_df = input_df[['time', 'lat', 'lon', 'alt']]
         input_df['time'] = input_df['time']/1e5
         freq = round(1/(np.mean(np.diff(input_df['time']))))
         #subsample the trajectory by the freq
         input_df = input_df.iloc[::freq, :]
         #
    
    return input_df

def av4_gps_imu_format(imu_data=None,gps_data=None, interp_poses = True,parse_sbet=True,sbet_deg=True,LineCreationTimes=None,buffer_size=1000,extension=".bin"):
     
        # Save raw-imu in expected format (time, gyro1(r/s), gyro2(r/s), gyro3(r/s), acc1(m/s2), acc2(m/s2), acc3(m/s2))
        #imu_DN = extract_data(in_data = imu_data,in_type='imu')
        #imu_path = os.path.join(os.path.dirname(imu_data), 'IMU.txt')
        #write_csv(imu_DN,imu_path)

        gps_DN = extract_data(in_data = imu_data,in_type='imu')
        gps_path = os.path.join(os.path.dirname(gps_data), 'IMU_Rad.txt')
        write_csv(gps_DN,gps_path,in_type='gps')
        #line_imu.to_csv(os.path.join(output_dir, 'line_imu.csv'), sep=',', index=False, header=True,float_format='%.6f',mode='w')


def main(config_file):
    description = "Produce AVIRIS-4 Geo-rectification data for a given line file"
    
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    
    # Read the configuration file
    print(f"Reading configuration file: {config_file}")
    config.read(config_file)

    
    # Initialize argparse without default values from config file
    parser = argparse.ArgumentParser(description=description)
    
    #parser.add_argument('mission_path', type=str, nargs='?', help='Path to dir containing sub-dir line files')
    parser.add_argument('trajectory_path', type=str, nargs='?', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to dir containing sub-dir line files')
    parser.add_argument('--imu_path', help='Path to the raw IMU data')
    parser.add_argument('--gps_path', help='Path to the raw GPS data')
    parser.add_argument('--sbet_deg',default=True, type=bool,help='convert sbet to deg or leave in radians (default:True)')

    # Parse the command-line arguments (without defaults from config file)
    args = parser.parse_args()
    
    # Update the arguments with defaults from the config file, if not provided on the command line
    #if not args.trajectory_path and config.has_section('PATHS') and config.has_option('PATHS', 'trajectory_path'):
    #    args.trajectory_path = config['PATHS'].get('trajectory_path')
    if not args.imu_path and config.has_section('PATHS') and config.has_option('PATHS', 'imu_path'):
        args.imu_path = config['PATHS'].get('imu_path')
    if not args.gps_path and config.has_section('PATHS') and config.has_option('PATHS', 'gps_path'):
        args.gps_path = config['PATHS'].get('gps_path')
    if not args.sbet_deg and config.has_section('OPTIONS') and config.has_option('OPTIONS', 'sbet_deg'):
        args.sbet_deg = config['OPTIONS'].getboolean('sbet_deg')

    #Clear args.config_file
    if args.config == config_file:
        args.config_file = None


    # Check if required arguments are provided
    #if not args.mission_path or not args.trajectory_path:
        #print("Mission path and trajectory path must be provided either in config file or command line.")
        #parser.print_help()
        #exit(1)

    # Call data extraction function
    av4_gps_imu_format(
        #in_path=args.mission_path,
        imu_data=args.imu_path,
        #gps_data=args.gps_path
    )


if __name__ == '__main__':
    # Set up the argument parser for the config file
    parser = argparse.ArgumentParser(description='Specify the config file.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    # # Parse the config file argument
    #args = parser.parse_args()

    # # Call the main function with the specified config file
    main('configs/av4-DN_inputs.ini') #if not args.config else main(args.config)
    
    # Format tie points for steviapp
    #input_csv = '/Users/jlahaye/Work/AVIRIS4/AV4_Thun/Thun_tie_points_DSM_Swiss_Image.csv'
    #steviapp_tps(input_csv,'/Volumes/fts-addlidar/AVIRIS_4_DATA/M002_240720_CHE-Thun/SteviAppProject/Thun_CHIM_DSM_TPS.txt')
   
    
     

        