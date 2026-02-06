import os
import csv
from pathlib import Path
from collections import defaultdict

def process_tie_points(root_directory, output_bingo='bingo_output.txt', output_timing='timing_info.csv'):
    """
    Process CSV files containing tie point coordinates and convert to BINGO format.
    
    Args:
        root_directory: Path to directory containing CSV files
        output_bingo: Output filename for BINGO format file
        output_timing: Output filename for timing information CSV
    """
    
    # Dictionary to store landmarks: {landmark_name: [(filename, time, u, v), ...]}
    landmarks = defaultdict(list)
    
    # Counter for generating unique observation IDs
    observation_id = 1
    
    # Dictionary to track observation_id -> (filename+landmark, time)
    observation_timing = {}
    
    # Get all CSV files in the root directory
    csv_files = list(Path(root_directory).rglob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {root_directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Read all CSV files and organize data by landmark name
    for csv_file in csv_files:
        filename = csv_file.stem  # Get filename without extension
        
        try:
            with open(csv_file, 'r') as f:
                # Skip header lines (lines starting with #)
                lines = f.readlines()
                data_lines = [line for line in lines if not line.startswith('#')]
                
                # Parse the CSV data
                reader = csv.reader(data_lines)
                next(reader)  # Skip the column header line
                
                for row in reader:
                    if len(row) >= 4:  # Ensure we have all required columns
                        landmark_name = row[0].strip()
                        time_coord = float(row[1])
                        u_coord = float(row[2])
                        v_coord = float(row[3])
                        
                        # Store the data organized by landmark name
                        landmarks[landmark_name].append({
                            'filename': filename,
                            'time': time_coord,
                            'u': u_coord,
                            'v': v_coord
                        })
                        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print(f"Found {len(landmarks)} unique landmarks")
    
    # Assign landmark IDs starting from 1000
    landmark_ids = {}
    current_landmark_id = 1000
    for landmark_name in sorted(landmarks.keys()):
        landmark_ids[landmark_name] = current_landmark_id
        current_landmark_id += 1
    
    # Write BINGO format output file
    # write bingo/timing to root_directory/output_bingo
    output_bingo = Path(root_directory) / output_bingo
    output_timing = Path(root_directory) / output_timing

    with open(output_bingo, 'w') as bingo_file:
        for landmark_name in sorted(landmarks.keys()):
            landmark_id = landmark_ids[landmark_name]
            observations = landmarks[landmark_name]
            
            # Write each observation for this landmark
            for obs in observations:
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
    # Specify the root directory containing your CSV files
    root_dir = "/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/steviapp_proj/LandMarks"  # Current directory, change this to your data directory
    
    # Process the files
    process_tie_points(
        root_directory=root_dir,
        output_bingo='AV4_Colombier_CAL_L3_5_bingo.txt',
        output_timing='AV4_Colombier_CAL_L3_5_timing_info.csv'
    )