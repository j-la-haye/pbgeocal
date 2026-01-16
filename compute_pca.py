
import numpy as np
from spectral import open_image, principal_components,view_cube
from spectral.io import envi
import os  
import matplotlib.pyplot as plt
from preprocess import linear_percentile_stretch, process_band
from hyperspectral import select_band
from plot_cube import plot_hyperspectral_cube as plot_cube
import wx #Enable for view_cube to work, comment out for plot_cube to work without GUI
app = wx.App(False) #Enable for view_cube to work, comment out for plot_cube to work without GUI
import spectral
spectral.settings.show_progress = True  # enable progress bar
spectral.settings.show_gui = True
#spectral.settings.WX_APP = False  # disable wx


# === Configuration ===
hdr_file = '/Volumes/fts-addlidar/fts-addlidar/Pika_L/2025/Blatten_Glacier/250610_CH_Blatten_PikaL_Data/00_image_data/00_Raw_Flight_Lines/line_10/data/manual-92/manual_Pika_L_92.bil.hdr'  #
  # Output ENVI file
num_top_bands = 4  # Number of top bands to extract per component

# === Load Image ===
img = open_image(hdr_file)
data = img.load()

# # === Perform PCA ===
pc = principal_components(data)
pc3 = pc.reduce(fraction=0.98).transform(data)  # Reduce to 98 percent of variance components
pc3_file = hdr_file.replace('.hdr', '_pca_98_components.hdr')
envi.save_image(pc3_file, pc3, dtype=np.float32, interleave='bil', ext='bil', metadata=img.metadata, force=True)

# === Get PCA Loadings (Eigenvectors) ===
loadings = pc.eigenvectors

# === Determine Unique Top Band Indices from First 3 Components ===
selected_band_indices_pc1 = set()
component = loadings[:, 0]
abs_component = np.abs(component)
top_indices = np.argsort(abs_component)[-num_top_bands:][::-1]
selected_band_indices_pc1.update(top_indices)

selected_band_indices_pc2 = set()
component = loadings[:, 1]
abs_component = np.abs(component)
top_indices = np.argsort(abs_component)[-num_top_bands:][::-1]
selected_band_indices_pc2.update(top_indices)


selected_band_indices_pc3 = set()
component = loadings[:, 2]
abs_component = np.abs(component)
top_indices = np.argsort(abs_component)[-num_top_bands:][::-1]
selected_band_indices_pc3.update(top_indices)
# Combine all selected indices
selected_band_indices = selected_band_indices_pc1.union(selected_band_indices_pc2).union(selected_band_indices_pc3)

for i in range(num_top_bands):
    component = loadings[:, i]
    abs_component = np.abs(component)
    top_indices = np.argsort(abs_component)[-num_top_bands:][::-1]
    selected_band_indices.update(top_indices)

# # Sort the band indices
selected_band_indices = sorted(selected_band_indices)
# #selected_band_indices = [132,82,32] # AV4 - RGB 38 (650),25 (550),11 (450), PikaL - RGB 132 (650),82 (550),32 (450)

# #Plot to verify
# #view_cube(data, bands=selected_band_indices, stretch=linear_percentile_stretch, title='PCA Top Bands')



# # === Create Output Metadata ===
# output_metadata = {
#     'bands': selected_band_indices,
#     'num_bands': len(selected_band_indices),
#     #'pca_components': 3,
# }
# output_metadata['wavelengths'] = [img.metadata['wavelength'][i] for i in selected_band_indices]
# #output_metadata['band_names'] = [img.metadata['band names'][i] for i in selected_band_indices]
# output_metadata['interleave'] = 'bil'
# output_metadata['data type'] = img.metadata['data type']
# output_metadata['samples'] = img.metadata['samples']
# output_metadata['lines'] = img.metadata['lines']
# output_metadata['header offset'] = img.metadata['header offset']
# output_metadata['byte order'] = img.metadata['byte order']
# output_metadata['file type'] = 'ENVI Standard'

# output_metadata['description'] = 'RGB of Pika_L data'
# output_metadata['sensor'] = img.metadata.get('sensor', 'Pika_L')
# output_metadata['acquisition date'] = img.metadata.get('acquisition date', 'Unknown')
# #output_metadata['processing history'] = 'PCA applied to the first 3 components, extracting top bands.'

# #output_file = os.path.basename(hdr_file).replace('.hdr', '_top_{}_pca.hdr'.format('_'.join(map(str, output_metadata['wavelengths']))))
# output_file = os.path.basename(hdr_file).replace('.bil.hdr', '_pca_wl.bil.hdr') #.format( len(selected_band_indices)))

# # check the output directory exists, if not create it
# output_dir = os.path.join(os.path.dirname(hdr_file),'pca_output')
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir) 
# out_path = os.path.join(output_dir,output_file)

# # === Extract Selected Bands ===
subset_data = data[:, :, selected_band_indices] 
rgb_indices = [120,71,39] # PikaL - RGB 123 (650),82 (550),32 (450) # AV4 - RGB 38 (650),25 (550),11 (450)
processed_bands = []
for i in rgb_indices: #range(3):
  processed_band = process_band(data[:, :, i])
  processed_bands.append(processed_band)
  final_rgb = np.dstack(processed_bands)
#plt.imshow(final_rgb)

#Plot to verify
#plot_cube(data, wavelengths=np.linspace(387,1027,300),top=final_rgb,alpha=1, title='Pika_L Blatten')


view_cube(subset_data, title='Pika_L Blatten RGB', stretch=linear_percentile_stretch, top=final_rgb, alpha=0.5)
app.MainLoop()

# === Save the Subset Image ===
#envi.save_image(out_path, final_rgb, dtype=np.float32, interleave='bil',ext='bil',metadata=output_metadata, force=True)

print()# f"\nSaved image with bands: {selected_band_indices}")
# print(f"Output written to: {output_file}.hdr and {output_file}")
