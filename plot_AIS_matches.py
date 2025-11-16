import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pandas as pd
from spectral import envi

def read_envi_rgb(hdr_file):
    """
    Read RGB BIL ENVI format image.
    
    Parameters:
    -----------
    hdr_file : str
        Path to the .hdr header file
        
    Returns:
    --------
    rgb_image : numpy.ndarray
        RGB image array with shape (height, width, 3)
    """
    img = envi.open(hdr_file)
    data = img.load()
    
    # BIL format: (bands, lines, samples)
    # Transpose to (height, width, bands)
    if len(data.shape) == 3:
        rgb = np.transpose(data, (1, 2, 0))
    else:
        rgb = data
    
    # Normalize to 0-1 range for display
    rgb = rgb.astype(float)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    return rgb

def load_matches(csv_file):
    """
    Load matching points from CSV file.
    
    Expected CSV format:
    u1, v1, u2, v2
    (x1, y1, x2, y2) pixel coordinates
    """
    matches = pd.read_csv(csv_file)
    return matches

def visualize_matches(img1_hdr, img2_hdr, matches_csv, max_matches=None):
    """
    Visualize two images side by side with matching points connected.
    
    Parameters:
    -----------
    img1_hdr : str
        Path to first image header file
    img2_hdr : str
        Path to second image header file
    matches_csv : str
        Path to CSV file with matching coordinates
    max_matches : int, optional
        Maximum number of matches to display (for clarity)
    """
    # Load images
    print("Loading images...")
    img1 = read_envi_rgb(img1_hdr)
    img2 = read_envi_rgb(img2_hdr)
    
    # Load matches
    print("Loading matches...")
    matches = load_matches(matches_csv)
    
    # Limit matches if specified
    if max_matches and len(matches) > max_matches:
        matches = matches.sample(n=max_matches, random_state=42)
    
    print(f"Visualizing {len(matches)} matches...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display images
    ax1.imshow(img1)
    ax1.set_title('Image 1', fontsize=14, fontweight='bold')
    ax1.axis('on')
    
    ax2.imshow(img2)
    ax2.set_title('Image 2', fontsize=14, fontweight='bold')
    ax2.axis('on')
    
    # Plot matching points and connections
    colors = plt.cm.rainbow(np.linspace(0, 1, len(matches)))
    
    for idx, (_, match) in enumerate(matches.iterrows()):
        u1, v1 = match.iloc[0], match.iloc[1]
        u2, v2 = match.iloc[2], match.iloc[3]
        
        color = colors[idx]
        
        # Plot points
        ax1.plot(u1, v1, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
        ax2.plot(u2, v2, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
        
        # Draw connection line
        con = ConnectionPatch(
            xyA=(u2, v2), xyB=(u1, v1),
            coordsA='data', coordsB='data',
            axesA=ax2, axesB=ax1,
            color=color, alpha=0.5, linewidth=1
        )
        ax2.add_artist(con)
    
    plt.tight_layout()
    
    # Enable interactive zoom
    plt.gcf().canvas.manager.set_window_title('Image Match Viewer - Use mouse to zoom and pan')
    
    return fig, ax1, ax2, matches

def main():
    """
    Main function - modify these paths for your data
    """
    # Example usage - modify these paths
    img1_hdr = '/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/' \
    'HS_Data_Lines/RGB/l3_rgb/bil_data/M024_250427_CHE_Colombier_Line_30001_131452_163_refl_rgb_650_550_450.hdr'
    img2_hdr = '/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/' \
    'HS_Data_Lines/RGB/l5_rgb/bil_data/M024_250427_CHE_Colombier_Line_50001_131938_226_refl_rgb_650_550_450.hdr'
    matches_csv = '/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/rgb_matches/raw_matches_export/' \
    'raw_matches/l3_5/stevi_match_raw_M024_250427_CHE_Colombier_Line_30001_131452_163_refl_rgb_650_550_450_M024_250427_CHE_Colombier_Line_50001_131938_226_refl_rgb_650_550_450.txt'
    
    # Limit number of matches for clearer visualization (optional)
    max_matches = 50  # Set to None to show all matches
    
    try:
        fig, ax1, ax2, matches = visualize_matches(
            img1_hdr, img2_hdr, matches_csv, max_matches
        )
        
        print("\nVisualization complete!")
        print(f"Total matches displayed: {len(matches)}")
        print("\nTips:")
        print("- Use the zoom tool to inspect match quality")
        print("- Use the pan tool to navigate the images")
        print("- Close the window to exit")
        
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("\nPlease update the file paths in the main() function:")
        print("- img1_hdr: path to first image .hdr file")
        print("- img2_hdr: path to second image .hdr file")
        print("- matches_csv: path to CSV file with matches")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
