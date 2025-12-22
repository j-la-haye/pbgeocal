import rasterio
import numpy as np

def crop_dsm_edges(input_dsm, output_dsm, buffer_pixels=10, nodata_value=-9999):
    """
    Crop edges of DSM to remove interpolation artifacts using rasterio
    """
    with rasterio.open(input_dsm) as src:
        # Read data
        data = src.read(1)
        
        # Get metadata
        profile = src.profile.copy()
        
        # Set nodata value if not already set
        if src.nodata is None:
            profile['nodata'] = nodata_value
        else:
            nodata_value = src.nodata
        
        # Crop edges - set to nodata
        data[:buffer_pixels, :] = nodata_value  # Top
        data[-buffer_pixels:, :] = nodata_value  # Bottom
        data[:, :buffer_pixels] = nodata_value  # Left
        data[:, -buffer_pixels:] = nodata_value  # Right
        
        # Update compression settings
        profile.update(compress='lzw', tiled=True)
        
        # Write output
        with rasterio.open(output_dsm, 'w', **profile) as dst:
            dst.write(data, 1)
    
    print(f"Cropped DSM saved to {output_dsm}")

# Usage
#crop_dsm_edges('input_dsm.tif', 'cropped_dsm.tif', buffer_pixels=15)

# Usage
crop_dsm_edges('/home/lasigadmin/Data/CHE_Blatten_DSM_LV95_LN02_OM_10cm.tif', '/home/lasigadmin/Data/cropped_dsm.tif', buffer_pixels=20)
