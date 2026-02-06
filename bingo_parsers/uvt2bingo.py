import os
import re

def parse_correspondences(root_dir, output_file):
    # Regex to match: UVT line_4 331.0 0.0 40286.777...
    # Group 1: Line Name, Group 2: X-coordinate
    pattern = re.compile(r"UVT\s+(line_\d+)\s+([\d\.]+)\s+0\.0\s+[\d\.]+")
    
    bingo_entries = []
    point_id_counter = 1000
    global_entry_counter = 1

    # Walk through root directory and sub-directories
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Assuming correspondence files are .txt or similar
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    for line in f:
                        # Find all UVT matches in the current line
                        matches = pattern.findall(line)
                        
                        # We expect a pair for a correspondence
                        if len(matches) == 2:
                            img1_name, x1 = matches[0]
                            img2_name, x2 = matches[1]
                            
                            # Format image names (e.g., line_4 -> L4_UVT)
                            img1_formatted = img1_name.replace("line_", "L") + "_UVT"
                            img2_formatted = img2_name.replace("line_", "L") + "_UVT"
                            
                            # Add Entry for Image 1
                            bingo_entries.append(f"{global_entry_counter} {img1_formatted} + {point_id_counter}")
                            bingo_entries.append(f"{point_id_counter}  {x1}  0")
                            bingo_entries.append("-99")
                            global_entry_counter += 1
                            
                            # Add Entry for Image 2
                            bingo_entries.append(f"{global_entry_counter} {img2_formatted} + {point_id_counter}")
                            bingo_entries.append(f"{point_id_counter}  {x2}  0")
                            bingo_entries.append("-99")
                            global_entry_counter += 1
                            
                            # Increment Point ID for the next shared tie-point
                            point_id_counter += 1

    # Write to output file
    with open(output_file, 'w') as out:
        out.write("\n".join(bingo_entries))

    print(f"Successfully processed {point_id_counter - 1000} tie-points.")
    print(f"Output saved to: {output_file}")

# --- Configuration ---
ROOT_INPUT_DIR = '/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/rgb_matches/raw_matches_export/raw_matches/uvt'  # Change this to your root folder path
OUTPUT_BINGO_FILE = '/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/rgb_matches/raw_matches_export/raw_matches/bingo.txt'  # Desired output file path

if __name__ == "__main__":
    parse_correspondences(ROOT_INPUT_DIR, OUTPUT_BINGO_FILE)
