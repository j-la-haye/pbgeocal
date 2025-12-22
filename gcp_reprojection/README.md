# GCP Reprojection Validation Package

A Python package to validate the reprojection of 3D Ground Control Points (GCPs) from geocentric (ECEF) coordinates to image coordinates using camera intrinsics and trajectory information.

## Features

- **BINGO Format Support**: Parse BINGO-style correspondence files with GCP observations
- **Trajectory Interpolation**: Interpolate camera poses from SBET or CSV trajectory files based on image capture times
- **Coordinate Transformations**: Full transformation chain from ECEF → NED → Body → Camera → Image
- **Flexible Configuration**: YAML-based configuration with support for:
  - Camera intrinsics (focal length, principal point, distortion)
  - Boresight misalignment correction
  - Lever arm offset
  - Coordinate conventions (V-axis direction)
- **Comprehensive Reporting**: JSON and CSV output with per-GCP and per-image statistics

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Input Files

**BINGO Correspondence File** (`correspondences.bingo`):
```
1 L4_UVT + 69_3_bldgs
1061  431.97  4.65812
-99
2 L4_UVT + 67_dmnd_bldg
1059  518.88  14.0513
-99
```

**GCP ECEF Coordinates** (`gcp_ecef.csv`):
```csv
gcp_id,x,y,z
1,4433500.123,-412345.678,4556789.012
2,4433512.456,-412334.567,4556778.901
```

**Trajectory File** (`trajectory.csv`):
```csv
time,latitude,longitude,height,roll,pitch,yaw
432001.000000,45.123456,-122.654321,1500.0,0.5,-2.3,45.2
432001.100000,45.123467,-122.654310,1500.5,0.6,-2.2,45.3
```

**Image Timing File** (`image_timing.csv`):
```csv
image_id,time
1059,432005.523456
1061,432005.723456
```

### 2. Create Configuration

```yaml
# config.yaml
epsg_code: 4978

camera:
  fx: 8000.0
  fy: 8000.0
  cx: 4000.0
  cy: 3000.0
  k1: -0.08
  k2: 0.005
  image_width: 8000
  image_height: 6000

conventions:
  v_axis_up: true
  correspondence_format: bingo
  trajectory_format: csv

files:
  gcp_image_coords: correspondences.bingo
  gcp_geocentric_coords: gcp_ecef.csv
  trajectory: trajectory.csv
  timing: image_timing.csv

validation_threshold: 3.0
```

### 3. Run Validation

**Command Line:**
```bash
gcp-validate config.yaml --output-dir ./results
```

**Python API:**
```python
from gcp_reprojection import Config, GCPValidator

config = Config.from_yaml("config.yaml")
validator = GCPValidator(config)
validator.load_data()
report = validator.validate()

print(f"RMSE: {report.rmse:.3f} pixels")
print(f"Pass rate: {report.pass_rate:.1%}")

validator.save_report(report, "results.json")
validator.save_residuals_csv(report, "residuals.csv")
```

## Coordinate Systems

### Transformation Chain

```
GCP (ECEF) → Local NED → Body Frame → Camera Frame → Image (u,v)
```

1. **ECEF (Earth-Centered, Earth-Fixed)**
   - X towards 0° longitude
   - Y towards 90°E
   - Z towards North Pole

2. **NED (North-East-Down)**
   - Local tangent plane at camera position
   - N: North, E: East, D: Down (nadir)

3. **Body Frame (SBET Convention)**
   - Forward-Right-Down
   - Roll (φ): rotation about forward axis
   - Pitch (θ): rotation about right axis
   - Yaw (ψ): rotation about down axis (heading from North)

4. **Camera Frame**
   - X: Right
   - Y: Back (opposite to forward)
   - Z: Down

5. **Image Coordinates**
   - u: Horizontal (right positive)
   - v: Vertical (down positive)
   - Origin: Top-left corner

### BINGO Photo-Coordinates

BINGO uses photo-coordinates with origin at image center:
- U: Positive right
- V: Positive up (photogrammetric convention)

The package automatically converts to pixel coordinates using camera dimensions.

## Configuration Reference

### Camera Intrinsics

| Parameter | Description |
|-----------|-------------|
| `fx`, `fy` | Focal lengths in pixels |
| `cx`, `cy` | Principal point (usually image center) |
| `k1`, `k2`, `k3` | Radial distortion coefficients |
| `p1`, `p2` | Tangential distortion coefficients |
| `image_width`, `image_height` | Image dimensions for BINGO conversion |

### Boresight Angles

Small angular offsets (degrees) between IMU body frame and camera frame:
- `roll`: Rotation about forward axis
- `pitch`: Rotation about right axis  
- `yaw`: Rotation about down axis

### Lever Arm

Offset (meters) from IMU center to camera projection center in body frame:
- `x`: Forward offset
- `y`: Right offset
- `z`: Down offset (negative = up)

### Conventions

| Parameter | Options | Description |
|-----------|---------|-------------|
| `v_axis_up` | `true`/`false` | V-axis direction in BINGO file |
| `correspondence_format` | `bingo`/`csv` | Correspondence file format |
| `trajectory_format` | `csv`/`sbet` | Trajectory file format |

## Output

### Validation Report (JSON)

```json
{
  "summary": {
    "total_measurements": 100,
    "valid_projections": 98,
    "pass_rate": 0.95
  },
  "error_statistics": {
    "mean_error": 1.23,
    "rmse": 1.45,
    "max_error": 3.21
  },
  "per_gcp": { ... },
  "per_image": { ... }
}
```

### Residuals (CSV)

```csv
gcp_id,gcp_name,image_id,measured_u,measured_v,projected_u,projected_v,residual_u,residual_v,error
1,L4_UVT + 69_3_bldgs,1061,4432.0,2995.3,4431.5,2996.1,-0.5,0.8,0.94
```

## API Reference

### Main Classes

- `Config`: Configuration management
- `GCPValidator`: Main validation workflow
- `DataLoader`: Load and manage input data
- `CoordinateTransformer`: Coordinate transformations
- `CameraModel`: Camera projection model
- `BINGOParser`: Parse BINGO format files
- `TrajectoryInterpolator`: Time-based pose interpolation

### Key Functions

```python
# Parse BINGO file directly
from gcp_reprojection import parse_bingo_file
observations = parse_bingo_file("correspondences.bingo")

# Parse timing file
from gcp_reprojection import parse_timing_file
timings = parse_timing_file("image_timing.csv")

# Load trajectory interpolator
from gcp_reprojection import load_trajectory_interpolator
interpolator = load_trajectory_interpolator("trajectory.csv")
pose = interpolator.interpolate(432001.5)
```

## Testing

```bash
pytest gcp_reprojection/tests/ -v
```

## License

MIT License
