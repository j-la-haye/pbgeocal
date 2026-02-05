import os
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from pyproj import Transformer, CRS

def read_timing_file(timing_file_path):
    """
    Read timing file and return list of timestamps.
    
    Args:
        timing_file_path: Path to timing file
        
    Returns:
        List of timestamps
    """
    timestamps = []
    try:
        with open(timing_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        timestamps.append(float(line.split(',')[0] if ',' in line else line))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading timing file {timing_file_path}: {e}")
    
    return timestamps

def read_checkpoints(checkpoint_file):
    """
    Read checkpoint file with ECEF coordinates.
    
    Args:
        checkpoint_file: Path to checkpoint CSV file
        
    Returns:
        Dictionary mapping landmark names to ECEF coordinates (x, y, z)
    """
    checkpoints = {}
    try:
        with open(checkpoint_file, 'r') as f:
            # Skip header lines (lines starting with #)
            lines = f.readlines()
            data_lines = [line for line in lines if not line.startswith('#')]
            
            reader = csv.reader(data_lines)
            next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 4:
                    landmark_name = row[0].strip()
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    checkpoints[landmark_name] = (x, y, z)
        
        print(f"Loaded {len(checkpoints)} checkpoints from {checkpoint_file}")
    except Exception as e:
        print(f"Error reading checkpoint file {checkpoint_file}: {e}")
    
    return checkpoints

def ecef_to_local(x_ecef, y_ecef, z_ecef, target_epsg):
    """
    Convert ECEF coordinates to local geocentric coordinates.
    
    Args:
        x_ecef, y_ecef, z_ecef: ECEF coordinates
        target_epsg: Target EPSG code (e.g., 2056 for Swiss LV95)
        
    Returns:
        Tuple of (x_local, y_local, z_local)
    """
    # Create transformer from ECEF (EPSG:4978) to target CRS
    transformer = Transformer.from_crs(
        CRS.from_epsg(4978),  # ECEF
        CRS.from_epsg(target_epsg),
        always_xy=True
    )
    
    x_local, y_local, z_local = transformer.transform(x_ecef, y_ecef, z_ecef)
    return x_local, y_local, z_local

def find_closest_time(uvt_time, timing_array, time_scale=1.0):
    """
    Find the closest timestamp in the timing array to the given UVT time.
    
    Args:
        uvt_time: Time coordinate from UVT file
        timing_array: Array of timestamps
        
    Returns:
        Closest timestamp from timing array
    """
    if len(timing_array) == 0:
        return uvt_time
    
    timing_array = np.array(timing_array) * time_scale
    idx = np.argmin(np.abs(timing_array - uvt_time))
    return timing_array[idx]

def process_tie_points(root_directory, checkpoint_file=None, target_epsg=2056,
                      output_bingo='bingo_output.txt', output_timing='timing_info.csv',
                      output_gcp='GCP.txt', timing_file_pattern='*timing*.csv',time_scale=1.0,ecef_coords=True):
    """
    Process CSV files containing tie point coordinates and convert to BINGO format.
    Matches each tie point to closest timing from associated timing file.
    Handles checkpoints with ECEF coordinates and converts to local system.
    
    Args:
        root_directory: Path to directory containing subdirectories with UVT and timing files
        checkpoint_file: Path to checkpoint file with ECEF coordinates (optional)
        target_epsg: Target EPSG code for local coordinate conversion (default: 2056 for Swiss LV95)
        output_bingo: Output filename for BINGO format file
        output_timing: Output filename for timing information CSV
        output_gcp: Output filename for GCP file with local coordinates
        timing_file_pattern: Glob pattern to identify timing files (default: '*timing*.csv')
    """
    
    # Load checkpoints if provided
    checkpoints = {}
    if checkpoint_file and Path(checkpoint_file).exists():
        checkpoints = read_checkpoints(checkpoint_file)
    else:
        print("No checkpoint file provided or file not found")
    
    # Dictionary to store landmarks: {landmark_name: [(filename, precise_time, u, v), ...]}
    landmarks = defaultdict(list)
    
    # Counter for generating unique observation IDs
    observation_id = 1
    
    # Dictionary to track observation_id -> (filename+landmark, precise_time)
    observation_timing = {}
    
    # Get all subdirectories in the root directory
    root_path = Path(root_directory)
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {root_directory}")
        return
    
    print(f"Found {len(subdirs)} subdirectories to process")
    
    # Process each subdirectory
    for subdir in subdirs:
        print(f"\nProcessing subdirectory: {subdir.name}")
        
        # Find UVT files (tie point files) and timing files
        uvt_files = [f for f in subdir.glob('*.csv') if not any(pattern in f.name.lower() for pattern in ['timing', 'time'])]
        # timing_files = list(subdir.glob(timing_file_pattern))
        
        # if not timing_files:
        #     # Try alternative patterns
        #     timing_files = [f for f in subdir.glob('*.csv') if 'timing' in f.name.lower() or 'time' in f.name.lower()]
        
        # if not timing_files:
        #     print(f"  Warning: No timing file found in {subdir.name}, using UVT times directly")
        #     timing_data = []
        # else:
        #     timing_file = timing_files[0]
        #     print(f"  Found timing file: {timing_file.name}")
        #     timing_data = read_timing_file(timing_file)
        #     print(f"  Loaded {len(timing_data)} timestamps")
        
        # Process each UVT file in this subdirectory
        for uvt_file in uvt_files:
            filename = uvt_file.stem  # Get filename without extension
            print(f"  Processing UVT file: {uvt_file.name}")
            
            try:
                with open(uvt_file, 'r') as f:
                    # Skip header lines (lines starting with #)
                    lines = f.readlines()
                    data_lines = [line for line in lines if not line.startswith('#')]
                    
                    # Parse the CSV data
                    reader = csv.reader(data_lines)
                    next(reader)  # Skip the column header line
                    
                    point_count = 0
                    for row in reader:
                        if len(row) >= 4:  # Ensure we have all required columns
                            landmark_name = row[0].strip()
                            uvt_time = float(row[1])
                            u_coord = float(row[2])
                            v_coord = float(row[3])
                            
                            # Find closest time in timing file
                            #if timing_data:
                            #    precise_time = find_closest_time(uvt_time, timing_data,time_scale)
                            #else:
                            #precise_time = uvt_time
                            
                            # Store the data organized by landmark name
                            landmarks[landmark_name].append({
                                'filename': filename,
                                'time': uvt_time,
                                'u': u_coord,
                                'v': 0 #v_coord
                            })
                            point_count += 1
                    
                    print(f"Processed {point_count} tie points")
                    
            except Exception as e:
                print(f"  Error processing {uvt_file}: {e}")
                continue
    
    print(f"Found {len(landmarks)} unique landmarks")
    
    # Separate landmarks into checkpoints (GCPs) and regular tie points
    gcp_landmarks = {name: obs for name, obs in landmarks.items() if name in checkpoints}
    tiepoint_landmarks = {name: obs for name, obs in landmarks.items() if name not in checkpoints}
    
    print(f"  - {len(gcp_landmarks)} landmarks have corresponding checkpoints (GCPs)")
    print(f"  - {len(tiepoint_landmarks)} landmarks are regular tie points")
    
    # Assign landmark IDs
    # GCPs get IDs 1-999
    # Regular tie points get IDs starting from 1000
    landmark_ids = {}
    
    # Assign IDs to GCP landmarks (1-999)
    current_gcp_id = 1
    for landmark_name in sorted(gcp_landmarks.keys()):
        landmark_ids[landmark_name] = current_gcp_id
        current_gcp_id += 1
    
    # Assign IDs to regular tie point landmarks (1000+)
    current_tiepoint_id = 1000
    for landmark_name in sorted(tiepoint_landmarks.keys()):
        landmark_ids[landmark_name] = current_tiepoint_id
        current_tiepoint_id += 1
    
    # Write GCP file with local coordinates
    if checkpoints:
        print(f"\nWriting GCP file with EPSG:{target_epsg} coordinates...")
        with open(output_gcp, 'w') as gcp_file:
            for landmark_name in sorted(gcp_landmarks.keys()):
                landmark_id = landmark_ids[landmark_name]
                x_ecef, y_ecef, z_ecef = checkpoints[landmark_name]
                
                # Convert ECEF to local coordinates
                if ecef_coords:
                    x_local, y_local, z_local = ecef_to_local(x_ecef, y_ecef, z_ecef, target_epsg)
                else:
                    x_local, y_local, z_local = x_ecef, y_ecef, z_ecef
                # Write to GCP file
                # check if epsg = 4326 and write lat lon alt as y_local, x_local, z_local
                if target_epsg == 4326:
                    gcp_file.write(f"{landmark_id}, {y_local:.14f}, {x_local:.14f}, {z_local:.3f}\n")
                else:
                    gcp_file.write(f"{landmark_id}, {x_local:.1f}, {y_local:.1f}, {z_local:.1f}\n")
        
        print(f"GCP file written to: {output_gcp}")
        print(f"  - {len(gcp_landmarks)} GCPs converted from ECEF to EPSG:{target_epsg}")
    
    # Create a flat list of all observations with their landmark info
    all_observations = []
    for landmark_name, observations in landmarks.items():
        landmark_id = landmark_ids[landmark_name]
        for obs in observations:
            all_observations.append({
                'landmark_name': landmark_name,
                'landmark_id': landmark_id,
                'filename': obs['filename'],
                'time': obs['time'],
                'u': obs['u'],
                'v': obs['v']
            })
    
    # Sort ALL observations by time in ascending order
    all_observations_sorted = sorted(all_observations, key=lambda x: x['time'])
    
    print(f"Total observations to write: {len(all_observations_sorted)}")
    
    # Write BINGO format output file with observations in time order
    with open(output_bingo, 'w') as bingo_file:
        for obs in all_observations_sorted:
            # Create the combined name for this observation
            obs_name = f"{obs['filename']} + {obs['landmark_name']}"
            
            # Write the observation line
            bingo_file.write(f"{observation_id} {obs_name}\n")
            
            # Write the landmark ID and coordinates
            bingo_file.write(f"{obs['landmark_id']}  {obs['u']}  {obs['v']}\n")
            
            # Write the separator
            bingo_file.write("-99\n")
            
            # Store timing info for this observation
            observation_timing[observation_id] = (obs_name, obs['time'])
            
            observation_id += 1
    
    print(f"BINGO format file written to: {output_bingo}")
    
    # Write timing information CSV
    with open(output_timing, 'w', newline='') as timing_file:
        writer = csv.writer(timing_file, delimiter=' ')
        
        # Write each observation's timing info
        for obs_id in sorted(observation_timing.keys()):
            obs_name, time = observation_timing[obs_id]
            writer.writerow([obs_id, time])
    
    print(f"Timing information written to: {output_timing}")
    print(f"Total observations processed: {observation_id - 1}")

# Example usage
if __name__ == "__main__":
    # Specify the root directory containing subdirectories with UVT and timing files
    # Structure expected:
    # root_directory/
    #   ├── subdir1/
    #   │   ├── uvt_file1.csv
    #   │   └── timing_file1.csv
    #   ├── subdir2/
    #   │   ├── uvt_file2.csv
    #   │   └── timing_file2.csv
    #   └── ...
    
    #root_dir = "."  # Current directory, change this to your data directory
    root_dir = "/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/steviapp_proj/LandMarks/raw/unrectified/UVT"
    checkpoint_file = "/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/DN_proc/25427_Thun_Colomb_Areuse/25_cal_ch/odyn_in_tp_unrect/DSM_check_points/DEM_GCP.csv"  # Path to checkpoint file with ECEF coordinates
    
    output_bingo = Path(root_dir) / 'bingo.txt'
    output_timing = Path(root_dir) / 'image_timestamps.txt'
    gcp_file = Path(root_dir) / 'GCPs.txt'

    # Process the files
    # You can customize the timing_file_pattern to match your timing file naming convention
    # Examples: '*timing*.csv', '*_time.csv', 'timestamps*.csv'
    process_tie_points(
        root_directory=root_dir,
        checkpoint_file=checkpoint_file,
        target_epsg=4326,  # Swiss LV95 -2056 default
        output_bingo=output_bingo,
        output_timing=output_timing,
        output_gcp=gcp_file,
        timing_file_pattern='*.timing',
        time_scale=1e-5,  # Adjust this pattern to match your timing files
        ecef_coords=False
    )
