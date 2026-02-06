#!/usr/bin/env python3
"""
Convert binary hyperspectral data cube to ENVI BIL format.
Configuration is read from a YAML file.
Optionally processes RGB bands for visualization.
"""

import numpy as np
from spectral.io import envi
from skimage import exposure
from PIL import Image
import yaml
import os
import sys


ENVI_DTYPE_MAP = {
    'uint8': 1,
    'int16': 2,
    'int32': 3,
    'float32': 4,
    'float64': 5,
    'uint16': 12,
    'uint32': 13,
    'int64': 14,
    'uint64': 15,
}

NUMPY_DTYPE_MAP = {
    'uint8': np.uint8,
    'int16': np.int16,
    'int32': np.int32,
    'float32': np.float32,
    'float64': np.float64,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'int64': np.int64,
    'uint64': np.uint64,
}


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_wavelengths(filepath):
    """Load wavelengths from text file, ignoring comments and blank lines."""
    wavelengths = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                wavelengths.append(float(line))
    return wavelengths


def find_closest_band(wavelengths, target):
    """Find index of band closest to target wavelength."""
    wavelengths = np.array(wavelengths)
    idx = np.argmin(np.abs(wavelengths - target))
    return idx


def linear_percentile_stretch(band, min_percent=2, max_percent=98):
    """Apply linear percentile stretch to a band."""
    min_val = np.percentile(band, min_percent)
    max_val = np.percentile(band, max_percent)
    if max_val == min_val:
        return np.zeros_like(band, dtype=np.float64)
    stretched = (band - min_val) / (max_val - min_val)
    stretched = np.clip(stretched, 0, 1)
    return stretched


def process_band_for_display(band, min_percent=2, max_percent=98, gamma=0.85, 
                              apply_clahe=True, output_max=65535):
    """
    Process a single band for visualization.
    
    Steps:
    1. Linear percentile stretch
    2. Optional CLAHE (adaptive histogram equalization)
    3. Gamma correction
    4. Scale to output range
    """
    stretched = linear_percentile_stretch(band, min_percent, max_percent)
    
    if apply_clahe:
        equalized = exposure.equalize_adapthist(stretched, clip_limit=0.03)
    else:
        equalized = stretched
    
    gamma_corrected = np.power(equalized, gamma)
    scaled = (gamma_corrected * output_max).astype(np.uint16)
    
    return scaled


def process_rgb_bands(bil_path, hdr_path, wavelengths, config):
    """Process RGB bands in the BIL file for better visualization."""
    rgb_config = config.get('rgb_wavelengths', {'red': 650, 'green': 550, 'blue': 450})
    stretch_config = config.get('rgb_stretch', {'min_percent': 2, 'max_percent': 98})
    gamma = config.get('rgb_gamma', 0.85)
    apply_clahe = config.get('rgb_clahe', True)
    
    # Find closest bands
    red_idx = find_closest_band(wavelengths, rgb_config['red'])
    green_idx = find_closest_band(wavelengths, rgb_config['green'])
    blue_idx = find_closest_band(wavelengths, rgb_config['blue'])
    
    print(f"Processing RGB bands: R={red_idx} ({wavelengths[red_idx]:.1f}nm), "
          f"G={green_idx} ({wavelengths[green_idx]:.1f}nm), "
          f"B={blue_idx} ({wavelengths[blue_idx]:.1f}nm)")
    
    # Open the ENVI file for read/write
    img = envi.open(hdr_path, bil_path)
    data = img.open_memmap(interleave='bip', writable=True)  # (lines, samples, bands)
    
    nlines, samples, bands = data.shape
    
    # Extract RGB bands as float for processing
    red_band = data[:, :, red_idx].astype(np.float64)
    green_band = data[:, :, green_idx].astype(np.float64)
    blue_band = data[:, :, blue_idx].astype(np.float64)
    
    # Process each band
    red_processed = process_band_for_display(
        red_band,
        stretch_config['min_percent'],
        stretch_config['max_percent'],
        gamma,
        apply_clahe
    )
    green_processed = process_band_for_display(
        green_band,
        stretch_config['min_percent'],
        stretch_config['max_percent'],
        gamma,
        apply_clahe
    )
    blue_processed = process_band_for_display(
        blue_band,
        stretch_config['min_percent'],
        stretch_config['max_percent'],
        gamma,
        apply_clahe
    )
    
    # Write processed bands back to BIL
    data[:, :, red_idx] = red_processed
    data[:, :, green_idx] = green_processed
    data[:, :, blue_idx] = blue_processed
    
    # Flush changes
    del data
    
    # Update header with default bands for RGB display
    update_header_default_bands(hdr_path, red_idx, green_idx, blue_idx)
    
    print(f"RGB bands processed and written to BIL")
    
    # Optionally save PNG preview
    if config.get('rgb_preview_png', False):
        save_rgb_preview_png(bil_path, hdr_path, red_idx, green_idx, blue_idx, config)


def update_header_default_bands(hdr_path, red_idx, green_idx, blue_idx):
    """Update ENVI header with default bands for RGB display."""
    with open(hdr_path, 'r') as f:
        header_content = f.read()
    
    # Add or update default bands (1-indexed in ENVI)
    default_bands_line = f"default bands = {{{red_idx + 1}, {green_idx + 1}, {blue_idx + 1}}}"
    
    if 'default bands' in header_content:
        # Replace existing
        lines = header_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('default bands'):
                lines[i] = default_bands_line
                break
        header_content = '\n'.join(lines)
    else:
        # Add before final closing or at end
        header_content = header_content.rstrip()
        if not header_content.endswith('\n'):
            header_content += '\n'
        header_content += default_bands_line + '\n'
    
    with open(hdr_path, 'w') as f:
        f.write(header_content)
    
    print(f"Header updated with default bands: R={red_idx + 1}, G={green_idx + 1}, B={blue_idx + 1}")


def save_rgb_preview_png(bil_path, hdr_path, red_idx, green_idx, blue_idx, config):
    """Save RGB preview as PNG from processed BIL."""
    img = envi.open(hdr_path, bil_path)
    data = img.open_memmap(interleave='bip')
    
    # Extract and scale to 8-bit
    red = (data[:, :, red_idx] / 256).astype(np.uint8)
    green = (data[:, :, green_idx] / 256).astype(np.uint8)
    blue = (data[:, :, blue_idx] / 256).astype(np.uint8)
    
    rgb = np.stack([red, green, blue], axis=-1)
    
    output_rgb = config['output_base'] + '_rgb.png'
    Image.fromarray(rgb).save(output_rgb)
    print(f"RGB preview saved: {output_rgb}")


def convert_bin_to_envi_bil(config):
    """
    Convert binary hyperspectral data cube to ENVI BIL format.
    
    Parameters:
        config: Dictionary with configuration parameters
    """
    input_path = config['input_file']
    output_path = config['output_base']
    samples = config['samples']
    bands = config['bands']
    data_type = config.get('data_type', 'uint16')
    interleave = config.get('interleave', 'bil').lower()
    byte_order = config.get('byte_order', 'little')
    
    np_dtype = NUMPY_DTYPE_MAP[data_type]
    envi_dtype = ENVI_DTYPE_MAP[data_type]
    bytes_per_pixel = np.dtype(np_dtype).itemsize
    bytes_per_line = samples * bands * bytes_per_pixel
    
    # Determine number of lines
    if 'lines' in config and config['lines'] is not None:
        nlines = config['lines']
    else:
        file_size = os.path.getsize(input_path)
        nlines = file_size // bytes_per_line
        print(f"Detected {nlines} lines from file size")
    
    # Load wavelengths from file
    wavelengths = None
    if 'wavelength_file' in config:
        wavelengths = load_wavelengths(config['wavelength_file'])
        if len(wavelengths) != bands:
            print(f"Warning: wavelength count ({len(wavelengths)}) doesn't match bands ({bands})")
    
    # Build ENVI header metadata
    metadata = {
        'lines': nlines,
        'samples': samples,
        'bands': bands,
        'interleave': 'bil',
        'data type': envi_dtype,
        'byte order': 0 if byte_order == 'little' else 1,
        'header offset': 0,
    }
    
    # Add optional metadata
    if 'description' in config:
        metadata['description'] = config['description']
    if 'sensor_type' in config:
        metadata['sensor type'] = config['sensor_type']
    if wavelengths:
        metadata['wavelength'] = wavelengths
    if 'wavelength_units' in config:
        metadata['wavelength units'] = config['wavelength_units']
    
    # Create output file
    output_bil = output_path + '.bil'
    output_hdr = output_path + '.hdr'
    
    out_image = envi.create_image(output_hdr, metadata, force=True, ext='.bil')
    out_memmap = out_image.open_memmap(writable=True)
    
    # Process line by line
    with open(input_path, 'rb') as f:
        for line_idx in range(nlines):
            raw = np.fromfile(f, dtype=np_dtype, count=samples * bands)
            
            if raw.size != samples * bands:
                print(f"Warning: Incomplete line at {line_idx}, got {raw.size} values")
                break
            
            # Apply XOR transformation (for uint16 data)
            if data_type == 'uint16':
                transformed = np.bitwise_xor(raw, np.uint16(32768))
            else:
                transformed = raw
            
            # Reshape based on input interleave
            if interleave == 'bil':
                transformed = transformed.reshape(bands, samples)
            elif interleave == 'bip':
                transformed = transformed.reshape(samples, bands).T
            elif interleave == 'bsq':
                raise ValueError("BSQ input requires different processing logic")
            
            out_memmap[line_idx, :, :] = transformed
            
            if (line_idx + 1) % 100 == 0:
                print(f"Processed {line_idx + 1}/{nlines} lines")
    
    del out_memmap
    print(f"Conversion complete: {output_bil}")
    
    # Process RGB bands if requested
    if config.get('rgb_processing', False):
        if wavelengths:
            process_rgb_bands(output_bil, output_hdr, wavelengths, config)
        else:
            print("Warning: RGB processing requires wavelength_file to be specified")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_bil.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = load_config(config_path)
    convert_bin_to_envi_bil(config)


if __name__ == '__main__':
    main()
