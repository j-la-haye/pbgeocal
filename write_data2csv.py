from pathlib import Path
import numpy as np
import csv
import pandas as pd
import numpy as np

def convert_gyro_to_radians(input_file,stem="output", extension=".csv"):
    """
    Read space-separated data file, convert gyro columns (1-3) to radians,
    and save as CSV.
    
    Parameters:
    -----------
    input_file : str
        Path to input text file with space-separated columns
    output_file : str
        Path to output CSV file
    """
    # Read the space-separated file

    output_file = Path(input_file).with_stem(stem).with_suffix(extension)
    df = pd.read_csv(input_file, sep=r'\s+',header=0)
    
    # Convert gyro columns (columns 1-3) from degrees to radians
    gyro_cols = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
    df[gyro_cols] = np.deg2rad(df[gyro_cols]).round(6)
    
    # Write to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Converted {input_file} to {output_file}")
    print(f"Gyro columns converted to radians: {gyro_cols}")

# Example usage:
if __name__ == "__main__":

    input_data = "/media/addLidar/Projects/Lidar_Processing/0008_ITA-MonteIato_UZH_2025_AVIRIS4-1560II-SPO/20250609-1_HB-TEN/03_Processing/02_DN_Proc/250609_Montelato/IMU_gyroDeg.txt"
    convert_gyro_to_radians(input_data, stem="IMU_r_s_m_s2", extension=".csv")
    #write_sbet_to_csv(input_data, input_data, step=1, stem="output_imu", extension=".csv", type='imu')
