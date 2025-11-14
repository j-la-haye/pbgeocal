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
    
    print(f"Original shape: {data.shape}")
    
    # Check if transpose is needed
    # Only transpose if bands dimension (should be 3) is not in the last position
    if len(data.shape) == 3:
        # If last dimension is already 3 or close to 3, assume correct format
        if data.shape[2] <= 3:
            rgb = data
            print(f"Data already in correct format (lines, samples, bands)")
        # Otherwise, assume BIL format: (lines, bands, samples) -> (lines, samples, bands)
        elif data.shape[1] == 3 or data.shape[1] < data.shape[2]:
            rgb = np.transpose(data, (0, 2, 1))
            print(f"Transposed from BIL format: {rgb.shape}")
        else:
            rgb = data
            print(f"Using data as-is")
        
        # If we have more than 3 bands, take first 3
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
            print(f"Using first 3 bands: {rgb.shape}")
    else:
        rgb = data
    
    # Normalize to 0-1 range for display
    rgb = rgb.astype(float)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    return rgb

def load_matches(csv_file, delimiter=','):
    """
    Load matching points from CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    delimiter : str
        Delimiter used in the file (default: ',')
        Common options: ',' (comma), ' ' (space), '\t' (tab)
    
    Expected file format (no header):
    u1 v1 u2 v2
    (x1, y1, x2, y2) pixel coordinates
    """
    matches = pd.read_csv(csv_file, delimiter=delimiter, header=None, index_col=None)
    return matches

def visualize_matches(img1_hdr, img2_hdr, matches_csv, delimiter=',', max_matches=None):
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
    delimiter : str
        Delimiter used in the CSV file (default: ',')
    max_matches : int, optional
        Maximum number of matches to display (for clarity)
    """
    # Load images
    print("Loading images...")
    img1 = read_envi_rgb(img1_hdr)
    img2 = read_envi_rgb(img2_hdr)
    
    # Load matches
    print("Loading matches...")
    matches = load_matches(matches_csv, delimiter)
    
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
    
    # Plot matching points with numbered labels and connecting lines
    colors = plt.cm.rainbow(np.linspace(0, 1, len(matches)))
    
    for idx, (_, match) in enumerate(matches.iterrows()):
        u1, v1 = match.iloc[0], match.iloc[1]
        u2, v2 = match.iloc[2], match.iloc[3]
        
        match_num = idx + 1
        color = colors[idx]
        
        # Plot points with markers
        ax1.plot(u1, v1, 'o', color='red', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax2.plot(u2, v2, 'o', color='red', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        
        # Add numbered labels
        ax1.text(u1, v1, str(match_num), color='yellow', fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', edgecolor='white', alpha=0.7))
        ax2.text(u2, v2, str(match_num), color='yellow', fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', edgecolor='white', alpha=0.7))
        
        # Draw connection line
        con = ConnectionPatch(
            xyA=(u2, v2), xyB=(u1, v1),
            coordsA='data', coordsB='data',
            axesA=ax2, axesB=ax1,
            color=color, alpha=0.4, linewidth=1.5
        )
        ax2.add_artist(con)
    
    # Enable mouse scroll zoom
    def zoom_factory(ax, base_scale=1.2):
        def zoom_fun(event):
            if event.inaxes != ax:
                return
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                return
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
            ax.figure.canvas.draw()
        
        fig = ax.get_figure()
        fig.canvas.mpl_connect('scroll_event', zoom_fun)
        return zoom_fun
    
    zoom_factory(ax1)
    zoom_factory(ax2)
    
    plt.tight_layout()
    
    # Enable interactive zoom
    plt.gcf().canvas.manager.set_window_title('Image Match Viewer - Use mouse to zoom and pan')
    
    return fig, ax1, ax2, matches

def main():
    """
    Main function - modify these paths for your data
    """
    #Data Paths 
    img1_hdr = '/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/HS_Data_Lines/RGB/l3_rgb/bil_data/M024_250427_CHE_Colombier_Line_30001_131452_163_refl_rgb_650_550_450.hdr'
    img2_hdr = '/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/HS_Data_Lines/RGB/l5_rgb/bil_data/M024_250427_CHE_Colombier_Line_50001_131938_226_refl_rgb_650_550_450.hdr'
    matches_csv = '/media/addLidar/AVIRIS_4_Testing/SteviApp_TiePoint_Testing/rgb_matches/rgb_matches_export/M024_250427_CHE_Colombier_Line_30001_131452_163_refl_rgb_650_550_450_M024_250427_CHE_Colombier_Line_50001_131938_226_refl_rgb_650_550_450.txt'
    
    # Delimiter for the matches file
    # Common options: ',' (comma), ' ' (space), '\t' (tab)
    delimiter = ' '
    
    # Limit number of matches for clearer visualization (optional)
    max_matches = 50  # Set to None to show all matches
    
    try:
        fig, ax1, ax2, matches = visualize_matches(
            img1_hdr, img2_hdr, matches_csv, delimiter, max_matches
        )
        
        print("\nVisualization complete!")
        print(f"Total matches displayed: {len(matches)}")
        
        # Save to file
        #print(f"\nSaving visualization to {output_file}...")
        #plt.savefig(output_file, dpi=150, bbox_inches='tight')
        #print(f"Saved successfully!")
        
        # Try to show interactively (will work in interactive environments)
        try:
            plt.show()
        except:
            pass
            #print(f"\nInteractive display not available. View the saved file: {output_file}")
        
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
