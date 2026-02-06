# Swiss DSM Tile Converter

Convert Swiss GeoTiff DSM tiles from **LV95/LN02** (Swiss planimetric and altimetric reference) to **WGS84/ETRS89** with ellipsoidal heights.

## Features

- **Dual transformation methods:**
  - **SwissTopo Reframe API**: Official high-accuracy transformation service
  - **PyProj**: Local transformation using PROJ library
  
- **Accuracy comparison**: Evaluate differences between methods across all tiles

- **Batch processing**: Process entire directories of DSM tiles

- **Configurable**: YAML-based configuration for all parameters

## Coordinate Systems

### Source (Swiss National Grid)
- **Planimetric**: LV95 (EPSG:2056) - Swiss CH1903+ / LV95
- **Altimetric**: LN02 - Swiss Levelling Network 1902 (geoid-based orthometric heights)

### Target (Global)
- **Default**: EPSG:4979 - WGS84 3D (longitude, latitude, ellipsoidal height)
- **Alternatives**: 
  - EPSG:4326 - WGS84 2D
  - EPSG:4937 - ETRS89 3D

## Transformation Details

### SwissTopo Reframe API
The [REFRAME service](https://www.swisstopo.admin.ch/en/rest-api-reframe) uses official Swiss geodetic models:
- **CHENyx06**: High-accuracy planimetric transformation grid
- **CHGeo2004**: Swiss geoid model for height transformation
- **Accuracy**: ~1-2 cm planimetric, ~1-2 cm vertical

### PyProj
Uses PROJ library with available transformation grids:
- Requires grid files for best accuracy (CHENyx06, CHGeo2004)
- Without grids: Uses Helmert transformation (~1-2 m accuracy)
- With grids: Matches Reframe accuracy

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Download PROJ grid files for better pyproj accuracy
# Note: projsync may be broken, use manual download instead:
try: projsync --source-id ch_swisstopo

mkdir -p ~/.local/share/proj
cd ~/.local/share/proj
wget https://cdn.proj.org/ch_swisstopo_CHENyx06_ETRS.tif
wget https://cdn.proj.org/ch_swisstopo_CHENyx06a.tif
wget https://cdn.proj.org/ch_swisstopo_chgeo2004_ETRS89_LN02.tif
```

## Usage

### 1. Configure

Edit `config.yaml`:

```yaml
input_dir: "/path/to/swiss/dsm/tiles"
output_dir: "/path/to/output"
target_crs: "EPSG:4979"  # WGS84 3D
```

### 2. Run

```bash
# Standard processing
python dsm_converter.py config.yaml

# Verbose output
python dsm_converter.py config.yaml --verbose
```

### 3. Review Results

- **Converted tiles**: `output_dir/*.tif`
- **Accuracy report**: `output_dir/conversion_report.json`
- **Statistics CSV**: `output_dir/accuracy_stats.csv`

## Output Files

### Converted GeoTiffs
- Named: `{original_name}_pyproj.tif`
- CRS: Target CRS (e.g., EPSG:4979)
- Heights: Ellipsoidal (WGS84/GRS80)

### Accuracy Report (JSON)
```json
{
  "summary": {
    "mean_horizontal_diff_m": 0.023,
    "mean_vertical_diff_m": 0.015,
    "notes": ["Reframe uses official SwissTopo parameters"]
  },
  "comparisons": [...]
}
```

### Statistics CSV
| file | n_points | horiz_mean_m | vert_mean_m | ... |
|------|----------|--------------|-------------|-----|
| tile1.tif | 10000 | 0.021 | 0.012 | ... |

## Understanding the Accuracy Comparison

### What the differences mean:
- **Small differences (< 5 cm)**: Both methods using proper grid files
- **Large horizontal differences (> 1 m)**: PyProj missing CHENyx06 grid
- **Large vertical differences (> 0.5 m)**: PyProj missing CHGeo2004 geoid

### Typical values (with proper grids):
- Horizontal: 1-3 cm RMS
- Vertical: 1-2 cm RMS

### Why differences exist:
1. **Algorithm differences**: Reframe may use FINELTRA for sub-cm accuracy
2. **Grid interpolation**: Different bilinear interpolation implementations
3. **Reference epoch**: ETRF93 vs WGS84 realizations differ by ~1 m globally

## API Reference

### ReframeClient

```python
from reframe_client import SwissTopoReframeClient, ReframeFormat

client = SwissTopoReframeClient()

# Single point transformation
result = client.transform_point(
    easting=2_600_000,   # LV95
    northing=1_200_000,  # LV95
    altitude=500.0,       # LN02
    transformation=ReframeFormat.LV95_TO_WGS84
)

print(f"Lon: {result.easting}, Lat: {result.northing}, H: {result.altitude}")
```

### Batch Transformation

```python
import numpy as np

# Arrays of coordinates
e = np.array([2_600_000, 2_601_000, 2_602_000])
n = np.array([1_200_000, 1_201_000, 1_202_000])
h = np.array([500.0, 510.0, 520.0])

# Transform all points
lon, lat, h_ellip = client.transform_points_lv95_to_wgs84(e, n, h)
```

## Troubleshooting

### "Coordinates out of valid range"
- Input coordinates must be within Switzerland bounds
- LV95 Easting: 2,485,000 - 2,834,000
- LV95 Northing: 1,074,000 - 1,296,000

### Large pyproj differences
projsync --source-id ch_swisstopo

Install PROJ grids manually (projsync may be broken):
```bash
mkdir -p ~/.local/share/proj
cd ~/.local/share/proj
wget https://cdn.proj.org/ch_swisstopo_CHENyx06_ETRS.tif
wget https://cdn.proj.org/ch_swisstopo_CHENyx06a.tif
wget https://cdn.proj.org/ch_swisstopo_chgeo2004_ETRS89_LN02.tif
```

### API rate limiting
- Reframe API may limit requests
- Increase `retry_delay` in config
- Use larger `comparison_sample_spacing` to reduce API calls

## References

- [SwissTopo REFRAME API](https://www.swisstopo.admin.ch/en/rest-api-reframe)
- [Swiss Coordinate Systems](https://www.swisstopo.admin.ch/en/knowledge-facts/surveying-geodesy/reference-systems.html)
- [CHGeo2004 Geoid Model](https://www.swisstopo.admin.ch/en/knowledge-facts/surveying-geodesy/geoid.html)
- [PROJ Coordinate Transformation](https://proj.org/)
- [EPSG:2056 - CH1903+/LV95](https://epsg.io/2056)
- [EPSG:4979 - WGS84 3D](https://epsg.io/4979)

## License

MIT License
