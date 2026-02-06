import os
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

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

def process_tie_points(root_directory, output_bingo='bingo_output.txt', output_timing='timing_info.csv', 
                      timing_file_pattern='*timing*.csv',time_scale=1 ):
    """
    Process CSV files containing tie point coordinates and convert to BINGO format.
    Matches each tie point to closest timing from associated timing file.
    
    Args:
        root_directory: Path to directory containing subdirectories with UVT and timing files
        output_bingo: Output filename for BINGO format file
        output_timing: Output filename for timing information CSV
        timing_file_pattern: Glob pattern to identify timing files (default: '*timing*.csv')
    """
    
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
        timing_files = list(subdir.glob(timing_file_pattern))
        
        if not timing_files:
            # Try alternative patterns
            timing_files = [f for f in subdir.glob('*.csv') if 'timing' in f.name.lower() or 'time' in f.name.lower()]
        
        if not timing_files:
            print(f"  Warning: No timing file found in {subdir.name}, using UVT times directly")
            timing_data = []
        else:
            timing_file = timing_files[0]
            print(f"  Found timing file: {timing_file.name}")
            timing_data = read_timing_file(timing_file)
            print(f"  Loaded {len(timing_data)} timestamps")
        
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
                            if timing_data:
                                precise_time = find_closest_time(uvt_time, timing_data,time_scale)
                            else:
                                precise_time = uvt_time
                            
                            # Store the data organized by landmark name
                            landmarks[landmark_name].append({
                                'filename': filename,
                                'time': precise_time,
                                'u': u_coord,
                                'v': v_coord
                            })
                            point_count += 1
                    
                    print(f"    Processed {point_count} tie points")
                    
            except Exception as e:
                print(f"  Error processing {uvt_file}: {e}")
                continue
    
    print(f"Found {len(landmarks)} unique landmarks")
    
    # Assign landmark IDs starting from 1000
    landmark_ids = {}
    current_landmark_id = 1000
    for landmark_name in sorted(landmarks.keys()):
        landmark_ids[landmark_name] = current_landmark_id
        current_landmark_id += 1
    
    # Write BINGO format output file
    with open(output_bingo, 'w') as bingo_file:
        for landmark_name in sorted(landmarks.keys()):
            landmark_id = landmark_ids[landmark_name]
            observations = landmarks[landmark_name]
            
            # Sort observations by time in ascending order
            observations_sorted = sorted(observations, key=lambda x: x['time'])
            
            # Write each observation for this landmark in time order
            for obs in observations_sorted:
                # Create the combined name for this observation
                obs_name = f"{obs['filename']} + {landmark_name}"
                
                # Write the observation line
                bingo_file.write(f"{observation_id} {obs_name}\n")
                
                # Write the landmark ID and coordinates
                bingo_file.write(f"{landmark_id}  {obs['u']}  {obs['v']}\n")
                
                # Write the separator
                bingo_file.write("-99\n")
                
                # Store timing info for this observation
                observation_timing[observation_id] = (obs_name, obs['time'])
                
                observation_id += 1
    
    print(f"BINGO format file written to: {output_bingo}")
    
    # Write timing information CSV
    with open(output_timing, 'w', newline='') as timing_file:
        writer = csv.writer(timing_file)
        
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
    
    root_dir = "/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/steviapp_proj/LandMarks"  # Current directory, change this to your data directory
    
    # Process the files
    # You can customize the timing_file_pattern to match your timing file naming convention
    # Examples: '*timing*.csv', '*_time.csv', 'timestamps*.csv'
    # save bingo and timing info in the root directory
    output_bingo = Path(root_dir) / 'bingo.txt'
    output_timing = Path(root_dir) / 'image_timestamps.txt'
    process_tie_points(
        root_directory=root_dir,
        output_bingo=output_bingo,
        output_timing=output_timing,
        timing_file_pattern='*.timing*',
        time_scale=1e-5  # Adjust this pattern to match your timing files
    )
