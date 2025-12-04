from __future__ import unicode_literals
#from hyspecgeocal.rectify import rectify_flat
from hyspecgeocal.JL_branch.load_data import find_hdr, read_bil,load_bils,split_bil_file
#from hyspecgeocal.resampling import resample
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
from hyspecgeocal.preprocess import linear_percentile_stretch
from hyspecgeocal.JL_branch.hyperspectral import select_band, compute_grayscale, compute_rgb
import sys
import glob
import os

#sys.path.append('/Users/jlahaye/Work/Students/JulienB/hygeocal/lib/python3.9/site-packages/osgeo/gdal.py')
import numpy as np
from osgeo import gdal,osr 
print(gdal.__version__)
import numpy as np
from skimage import exposure
import numpy as np

def linear_stretch(band, min_percent=2, max_percent=98):
    min_val = np.percentile(band, min_percent)
    max_val = np.percentile(band, max_percent)
    stretched = (band - min_val) / (max_val - min_val)
    stretched = np.clip(stretched, 0, 1)
    return (stretched * 255).astype(np.uint8)

def apply_gamma(image, gamma=0.8):
    return np.power(image / 255.0, gamma) * 255

def process_band(band, gamma=0.85):
    stretched = linear_percentile_stretch(band) #linear_stretch(band)
    equalized = exposure.equalize_adapthist(stretched)
    gamma_corrected = np.power(equalized, gamma) * 255
    return gamma_corrected.astype(np.uint8)

def numpy_to_envi_bil(image_array, output_file,wavelengths):
    """
    Convert a 3-channel NumPy array to ENVI BIL format.
    
    Parameters:
    image_array (numpy.ndarray): Input image as a 3D NumPy array (height, width, 3)
    output_file (str): Path to the output .bil file
    """
    
    # Get image dimensions
    height, width,bands = image_array.shape

    #image_array[:,:,0] = np.uint32(255 * linear_percentile_stretch(image_array[:,:,0]))
    #image_array[:,:,1] = np.uint32(255 * linear_percentile_stretch(image_array[:,:,1]))
    #image_array[:,:,2] = np.uint32(255 * linear_percentile_stretch(image_array[:,:,2]))

    #image_array[:,:,0] =  linear_percentile_stretch(image_array[:,:,0])
    #image_array[:,:,1] =  linear_percentile_stretch(image_array[:,:,1])
    #image_array[:,:,2] =  linear_percentile_stretch(image_array[:,:,2])
    #plt.imshow(image_array)

    #image_array = np.uint32(255 * image_array)

    #plt.imshow(image_array)

    
    # Reshape array to BIL format
    #image_array_bil = image_array.transpose(1, 0, 2).reshape(width, height * bands)
    
    # Create ENVI file
    driver = gdal.GetDriverByName('ENVI')
    dataset = driver.Create(output_file, width, height, bands, gdal.GDT_Float32, options=["INTERLEAVE=BIL"])
    #dataset = driver.Create(output_file, width, height, bands, gdal.GDT_Float32)
    # Set the interleave to BIL in the metadata
    #dataset.SetMetadataItem('INTERLEAVE', 'BIL', 'ENVI')
    
    # Add wavelength information to the metadata
    dataset.SetMetadataItem('wavelength', '{' + ', '.join(map(str, wavelengths)) + '}', 'ENVI')
    dataset.SetMetadataItem('wavelength units', 'nm', 'ENVI')
    dataset.SetMetadataItem('field of view','40.2', 'ENVI')
    dataset.SetMetadataItem('imager serial number','AVIRIS-4', 'ENVI')
    dataset.SetMetadataItem('fps','214', 'ENVI')
    dataset.SetMetadataItem( 'ceiling','1', 'ENVI')
    dataset.SetMetadataItem('byte order','0')
    dataset.SetMetadataItem('data type','2')
    dataset.SetMetadataItem('header offset','0')
    dataset.SetMetadataItem('file type','ENVI Standard')
    dataset.SetMetadataItem('interleave','bil')
    dataset.SetMetadataItem('samples',str(width))
    dataset.SetMetadataItem('lines',str(height))
    dataset.SetMetadataItem('bands',str(bands))
    
   

    # Write data to the dataset
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        # tranform image array to 2D array
        band_data = image_array[:, :, i].reshape(height, width)
        #band_data = image_array[:, :, i]
        band.WriteArray(process_band(band_data))
        #band.WriteArray(linear_percentile_stretch(band_data)) #image_array[:, :, i])

    # Write data to file
    #for i in range(bands):
    #    band = dataset.GetRasterBand(i + 1)
    #    band_data = image_array_bil[:, i*height:(i+1)*height]
        #band.WriteArray(image_array_bil[:, i::bands].transpose())
        #band.WriteArray(band_data.transpose())
    
    # Set projection (WGS84 as an example)
    #srs = osr.SpatialReference()
    #srs.ImportFromEPSG(4326)
    #dataset.SetProjection(srs.ExportToWkt())
    
    # Set geotransform (example values, adjust as needed)
    #dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    
    # Close the dataset
    dataset = None
    
    print(f"ENVI BIL file created: {output_file}")

if __name__ == '__main__':

    #input_files =  '/Volumes/workspace/common/PROJECTS/AIS/AVIRIS_4/AV4_Missions/24_07_Campaigns/20-7-24-AV4Flights/M002_240720_CHE-Thun/radiance_lines/L101/radiance_bils/L101_locked/1014_0001'
    #input_files = '/Volumes/workspace/common/PROJECTS/AIS/AVIRIS_4/AV4_Missions/24_Sept_Campaign/AV4_Data/FRA_Pau/Raw_AV4_Data/Image_Data/L0001/rdn_bils_tiles'
    input_files = '/Volumes/fts-addlidar/AVIRIS_4_Mission_Processing/M007_240725_CHE-Swiss-National-Park/bils_radiance/L007/0001_0001'
    #input_files = '1014_0001/'
    #input_files = 'color-1014.hdr'

    
    
    for root, dirs, files in os.walk(input_files):

        #find files with .hdr extension
        header_files = glob.glob(os.path.join(root, "*.hdr"))
        for i,header_file_path in enumerate(header_files):
            # Process the header file here

            hdr,img = read_bil(header_file_path)
            input_file_path = hdr.filename
            #create new 'clr' directory in the same directory as the input file
            output_file = os.path.join(os.path.join(os.path.dirname(input_files), 'clr_bil_tiles'), f"{i:02d}_tile/{i:02d}_rdn_clr.bil")
            
            #check if the output directory exists, if not create it
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            
            rgb_image,channels = compute_rgb([hdr], [img])

            #rgb_image[0] = np.uint32(255 * rgb_image[0])

            # Process each band
            #red_processed = process_band(rgb_image[0][:,:,2])
            #green_processed = process_band(rgb_image[0][:,:,1])
            #blue_processed = process_band(rgb_image[0][:,:,0])
            processed_bands = []
            for i in [2,1,0]: #range(3):
                processed_band = process_band(rgb_image[0][:, :, i])
                processed_bands.append(processed_band)
            final_rgb = np.dstack(processed_bands)
            # Create final RGB image
            #final_rgb = np.dstack((red_processed, green_processed, blue_processed))
    
    
            #rgb_image[0][:,:,0] = linear_percentile_stretch(rgb_image[0][:,:,0])
            #rgb_image[0][:,:,1] = linear_percentile_stretch(rgb_image[0][:,:,1])
            #rgb_image[0][:,:,2] = linear_percentile_stretch(rgb_image[0][:,:,2])
            #plt.imshow(rgb_image[0])

            #rgb_img = np.uint32(255 * rgb_image[0])

            # Plot the RGB image
            plt.imshow(final_rgb)

            # Convert to ENVI BIL format
            numpy_to_envi_bil(rgb_image[0], output_file,wavelengths=[449,553,650])    

    #hdr,img = read_bil(in_file)
    # Extract number of samples

    # Plot data as an rgb image selecting best channels corresponding to red, green and x = ('apple', 'banana', 'cherry')
    # Select the best channels for red, green, and blue
    #red_channel = img[:, :, 36]  # Replace 10 with the index of the best red channel
    #green_channel = img[:, :, 28]  # Replace 20 with the index of the best green channel
    #blue_channel = img[:, :, 12]  # Replace 30 with the index of the best blue channel

   

    # Create the RGB image
    #rgb_image = np.dstack((red_channel, green_channel, blue_channel))
    #plt.imshow(rgb_image[0])
    #rgb_image = rgb_image.astype(np.uint8)*255
    #Modify image to be in the range 0-255
    #rgb_image = ((rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image)) * 255).astype(np.uint8)
    #rgb_image = rgb_image.astype(np.uint8)
    
    # Plot the RGB image
    #rgb_image = linear_percentile_stretch(rgb_image[0])
    #plt.imshow(rgb_image)
    
    # rgb_image[0] = np.uint32(255 * rgb_image[0])
    
    
    # rgb_image[0][:,:,0] = linear_percentile_stretch(rgb_image[0][:,:,0])
    # rgb_image[0][:,:,1] = linear_percentile_stretch(rgb_image[0][:,:,1])
    # rgb_image[0][:,:,2] = linear_percentile_stretch(rgb_image[0][:,:,2])
    # plt.imshow(rgb_image[0])

    # rgb_img = rgb_image[0] #np.uint32(255 * rgb_image[0])

    # # Plot the RGB image
    # plt.imshow(rgb_img)

    

    # Set the plot title
    #plt.title('RGB Image')

    # Set the axis labels
    #plt.xlabel('X')
    #plt.ylabel('Y')

    # Display the plot
    #plt.show()

    # Create sample array
    #arr = np.random.rand(100, 200).astype(np.float32)

# Save as .bil file
#rgb_image.tofile('/Volumes/workspace/common/PROJECTS/AIS/AVIRIS_4/July_24_Campaigns/20-7-24-AV4Flights/M002_240720_CHE-Thun/Line_101/101_AV4_Locked/processed_bils/1014_0001/color-1014.bil')

# Write .hdr file
    # with open('/Volumes/workspace/common/PROJECTS/AIS/AVIRIS_4/July_24_Campaigns/20-7-24-AV4Flights/M002_240720_CHE-Thun/Line_101/101_AV4_Locked/processed_bils/1014_0001/color-1014.hdr', 'w') as f:
    #     f.write(f"samples {rgb_image.shape[1]}\n")
    #     f.write(f"lines {rgb_image.shape[0]}\n") 
    #     f.write(f"bands {rgb_image.shape[2]}\n")
    #     f.write("data type 4\n")
    #     f.write("BYTEORDER 0\n")
    #     f.write("interleave bil\n")
    #     f.write("PIXELTYPE FLOAT\n")

    # Example usage
#if __name__ == "__main__":
    # Create a sample 3-channel image (100x100 pixels)
#    sample_image = np.random.rand(100, 100, 3)
    
    
   