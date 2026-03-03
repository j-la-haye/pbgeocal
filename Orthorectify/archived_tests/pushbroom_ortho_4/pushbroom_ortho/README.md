# Pushbroom Orthorectification Pipeline

**Bottom-up (output-to-input) orthorectification of hyperspectral linear-array
(pushbroom) imagery with true-ortho occlusion handling.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        main.py                                      │
│  Load config → Launch tile_processor → Write GeoTIFF                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │        tile_processor.py             │
            │  Divide grid → ProcessPoolExecutor   │
            │  tqdm progress → Mosaic results      │
            └──────────────────┬──────────────────┘
                               │  per tile
            ┌──────────────────▼──────────────────┐
            │         ortho_engine.py              │
            │  Grid→DSM→ECEF→Newton→Vis→Resample   │
            └───┬──────┬──────┬──────┬──────┬─────┘
                │      │      │      │      │
    ┌───────────▼┐ ┌───▼───┐ ┌▼─────┐ ┌────▼────┐ ┌──▼──────────┐
    │ dsm_handler│ │traject│ │camera │ │time_solv│ │ visibility  │
    │ .py        │ │ory.py │ │_model │ │er.py    │ │ .py         │
    │            │ │       │ │.py    │ │         │ │             │
    │ GeoTIFF    │ │ Slerp │ │Brown- │ │ Newton  │ │ Ray-DSM     │
    │ bilinear Z │ │ interp│ │Conrady│ │ iterate │ │ occlusion   │
    └────────────┘ └───┬───┘ └───────┘ └─────────┘ └─────────────┘
                       │
              ┌────────▼────────┐      ┌────────────────┐
              │ sbet_reader.py  │      │ coord_utils.py  │
              │ Binary SBET I/O │      │ ECEF/NED/quat   │
              └─────────────────┘      └────────────────┘
```

---

## Module Descriptions

### 1. `config.yaml` — Pipeline Configuration

All parameters in a single YAML file:

| Section | Contents |
|---|---|
| `paths` | Input SBET, BIL, exposure times, DSM; output GeoTIFF |
| `crs` | EPSG codes for DSM and output; SBET datum |
| `camera` | Smile polynomial (a, b, c); optics correction (f, f_lab, cx, cy); angle LUT path in `paths` |
| `mounting` | Lever arm (body frame, metres), boresight angles (°), 3×3 mounting matrix |
| `processing` | GSD, tile size, workers, Newton params, resampling, visibility settings |

### 2. `config_loader.py` — Configuration Parser

Reads YAML and produces typed dataclass objects. Converts boresight angles from
degrees to radians and mounting matrix/lever arm from lists to numpy arrays.
Validates all required fields and array shapes.

### 3. `sbet_reader.py` — SBET Binary Reader

Reads the Applanix POSPac SBET binary format:

- **17 fields × float64 = 136 bytes per record** at 200 Hz
- Fields: time, lat, lon, alt, velocities, roll, pitch, heading, wander angle, accelerations, angular rates
- Computes **true heading** = `platform_heading + wander_angle`
- Provides `trim_sbet()` to extract only the time window covering the flight line (with margin for interpolation edge effects)

### 4. `coord_utils.py` — Coordinate & Rotation Utilities

The mathematical foundation of the pipeline:

**Coordinate conversions:**
- `geodetic_to_ecef()` — WGS-84 geodetic (φ, λ, h) → ECEF (X, Y, Z)
- `ecef_to_geodetic()` — Bowring's iterative method

**Rotation matrices:**
- `rotation_ecef_to_ned(lat, lon)` — R_{NED←ECEF} at a geodetic position
- `euler_to_rotation(roll, pitch, heading)` — R_{Body←NED} using ZYX Euler order: `Rz(ψ) · Ry(θ) · Rx(φ)`
- Batch versions for vectorised processing of N records

**Quaternion Slerp:**
- `rotation_to_quaternion()` / `quaternion_to_rotation()` — Shepperd's method
- `slerp()` / `slerp_batch()` — Spherical linear interpolation with shortest-path enforcement and near-parallel fallback
- Quaternion continuity correction (sign flipping) to prevent path reversals

**Full orientation chain:**
```
R_{Cam←ECEF}(t) = R_{Cam←Body} · R_{Body←NED}(t) · R_{NED←ECEF}(t)

where:
  R_{Cam←Body} = R_boresight · R_mounting     (fixed calibration)
  R_{Body←NED} = Rz(ψ) · Ry(θ) · Rx(φ)       (interpolated at time t)
  R_{NED←ECEF} = f(lat, lon)                   (interpolated at time t)
```

### 5. `trajectory.py` — Trajectory Interpolator

Pre-computes quaternions from all SBET Euler angles and provides interpolation at arbitrary times:

- **Position**: Linear interpolation in ECEF (sufficient at 200 Hz / ~5 ms intervals)
- **Attitude**: Quaternion Slerp for both R_{Body←NED} and R_{NED←ECEF}, avoiding gimbal lock and ensuring smooth rotation interpolation
- **Lever arm**: Transforms the IMU→Camera offset from body frame to ECEF at each time step:
  ```
  L_ecef(t) = R_{NED←ECEF}(t)^T · R_{Body←NED}(t)^T · L_body
  pos_camera(t) = pos_IMU(t) + L_ecef(t)
  ```
- **Initial time guess**: Coarse nearest-neighbour search on decimated trajectory for Newton initialisation

### 6. `camera_model.py` — Decoupled Pushbroom Camera Model

**Why NOT Brown-Conrady for linear arrays:**

The standard Brown-Conrady model couples X and Y distortion via r² = x² + y².
For a pushbroom sensor where y ≈ 0 always, this causes fundamental problems:
the radial terms collapse to functions of x² alone, the tangential terms model
non-existent cross-coupling, and the optimizer fights between fitting pixel
positions (X) and forcing along-track angles (Y) to zero. The along-track model
error dominates and **corrupts the across-track fit** via the shared r² terms.
The optical "smile" follows a quadratic-in-tan(θ) law, not a radially symmetric
polynomial.

**Decoupled calibration with in-flight correction:**

The camera model has three layers:

1. **Across-Track (X) — Lab-measured angle LUT:**
   - Each pixel has a precisely measured viewing angle θ_xt from PSF analysis
   - The LUT captures per-pixel irregularities no polynomial can reproduce
     (polynomial fits plateau at ~0.07 px residual; LUT interpolation is exact)
   - 1240 pixels, ±20° FOV, ~0.0325°/pixel IFOV

2. **Along-Track (Y) — Smile polynomial:**
   - `θ_at(xt) = a·xt² + b·xt + c`  where `xt = tan(θ_across_track)`
   - `c` = boresight offset (~0.60°, sensor line not exactly at θ_y = 0)
   - `a` = parabolic curvature (smile amplitude ~2.3 arcsec edge-to-edge)

3. **In-Flight Correction — Tangent-space affine transform:**

   The lab LUT is measured at reference temperature/pressure (T₀, P₀). In flight,
   thermal expansion and ambient pressure shift the focal length and principal point.

   On the focal plane, pixel i sits at: `u_det(i) = f · tan(θ) + cx`

   Equating lab and flight conditions:
   ```
   f₀ · tan(θ_lab) + cx₀ = f · tan(θ_true) + cx
   ```
   Solving for the lab-equivalent tangent:
   ```
   tan(θ_lab) = s · tan(θ_true) + Δcx
   ```
   where `s = f/f₀` (focal ratio) and `Δcx = (cx−cx₀)/f₀` (normalised PP shift).

   The full projection chain becomes:
   ```
   x_true = Xc/Zc                      (observed tangent from 3D point)
   x_lab  = s · x_true + Δcx           (affine correction to lab space)
   pixel  = interp(arctan(x_lab), LUT) (across-track pixel from LUT)

   y_true = Yc/Zc                      (observed along-track tangent)
   y_lab  = s · y_true + Δcy           (affine correction)
   residual = y_lab − tan(smile(x_lab)) → 0   (Newton target)
   ```

   Properties of this formulation:
   - s=1, Δcx=Δcy=0 recovers the pure lab calibration exactly
   - The correction is **linear in tangent space** (physically exact, not approximate)
   - All per-pixel LUT irregularities are preserved through the correction
   - Only 3 free parameters (f, cx, cy) estimated externally and entered in config
   - Typical magnitudes: Δf/f ~ 50–200 ppm, ΔPP ~ 0.1–1 px

   Config parameters (`camera.optics`):
   ```yaml
   optics:
     f_lab: 1762.2    # lab focal length [pixels] (from LUT mean IFOV)
     f:     1762.2    # in-flight focal length [pixels]
     cx:    0.0       # across-track PP shift [pixels]
     cy:    0.0       # along-track  PP shift [pixels]
   ```

### 7. `time_solver.py` — Newton Iteration (Core Algorithm)

**The central mathematical problem:** Given a ground point P in ECEF, find the
time t* when the pushbroom sensor's scanline swept across P.

**Formulation:**
```
Define:
  x_true(t) = Xc(t)/Zc(t),   y_true(t) = Yc(t)/Zc(t)

  where P_cam(t) = R_{Cam←ECEF}(t) · (P_ecef − S_ecef(t))

  x_lab(t)  = s · x_true(t) + Δcx
  y_lab(t)  = s · y_true(t) + Δcy
  pixel(t)  = interp(arctan(x_lab(t)), angle_LUT)

  f(t) = y_lab(t) − tan(smile(x_lab(t)))

Solve:  f(t*) = 0   (corrected along-track tangent matches the smile
                      prediction at the corrected across-track position)
```

**Newton's method:**
```
t_{n+1} = t_n − f(t_n) / f'(t_n)

f'(t) ≈ [f(t+δ) − f(t−δ)] / (2δ)    (central difference, δ = 1 μs)
```

**Convergence:** Typically 3–5 iterations for airborne platforms (smooth trajectory,
small angular rates). Convergence threshold ~1 ns.

**Vectorised implementation:** All M pixels in a tile are solved simultaneously
using numpy array operations. Three trajectory interpolations per iteration
(at t, t+δ, t−δ) with batch quaternion Slerp.

**Output:** For each ground point: solved time t*, across-track pixel u, fractional
scanline index (for BIL lookup), and a validity mask.

### 8. `visibility.py` — Occlusion Detection (True Ortho)

For true orthophoto generation, occluded areas must be identified and masked.

**Algorithm:**
1. Parameterise the ray from sensor S to ground point P: `R(α) = S + α·(P−S)`
2. Sample at N points along α ∈ (0.05, 0.95)
3. At each sample, compare the DSM height to the ray height
4. If `z_DSM > z_ray + tolerance` anywhere → point is **occluded**

The tolerance parameter (default 0.5 m) prevents false positives from DSM noise.

### 9. `ortho_engine.py` — Per-Tile Processing

Orchestrates the full bottom-up pipeline for one tile:

```
For each output pixel (x, y) in the tile:
  1. z = DSM(x, y)                           → elevation lookup
  2. (X,Y,Z)_ecef = CRS_to_ECEF(x, y, z)    → coordinate transform
  3. t* = Newton(P_ecef)                      → time solving
  4. (u, v) = project(P_cam)                  → distorted pixel coords
  5. visible = ray_check(S, P)                → occlusion test
  6. pixel = BIL[scanline, :, sample]         → bilinear resampling
```

**Resampling:** Bilinear interpolation in both the scanline (along-track) and
sample (across-track) dimensions, performed on the BIL-interleaved data
(lines, bands, samples) to extract all spectral bands simultaneously.

### 10. `tile_processor.py` — Parallel Orchestration

- Generates the output grid from DSM extent and target GSD
- Divides into tiles (default 512×512)
- Dispatches to `ProcessPoolExecutor` with `tqdm` progress bar
- Each worker initialises its own copies of trajectory, DSM, camera, and solver
- BIL data is memory-mapped (shared read-only access across forks)
- Results written to GeoTIFF as tiles complete (with DEFLATE compression)

---

## Coordinate Frame Summary

```
ECEF (WGS-84)                Earth-Centered Earth-Fixed
    │
    │  R_{NED←ECEF}(lat, lon)
    ▼
NED (Navigation)             X-North, Y-East, Z-Down
    │
    │  R_{Body←NED}(roll, pitch, heading)    ← Euler ZYX
    ▼
Body (Applanix)              X-Forward, Y-Right, Z-Down
    │
    │  R_{Cam←Body} = R_boresight · R_mounting
    ▼
Camera                       X-Right, Y-Back, Z-Down
```

---

## Usage

```bash
# Install dependencies
pip install numpy scipy pyproj spectral pyyaml tqdm gdal

# Edit config.yaml with your paths and parameters

# Run
python main.py config.yaml
```

---

## Inputs

| File | Format | Description |
|---|---|---|
| SBET | Binary (17×float64/record) | Applanix trajectory at 200 Hz |
| BIL image | ENVI BIL | Hyperspectral pushbroom data |
| BIL header | ENVI .hdr | Metadata (lines, samples, bands, dtype) |
| Exposure times | Text (one per line) | GPS time in 10 μs (1e-5 s) units |
| Angle LUT | CSV (one per line) | Lab-measured across-track angles [degrees] |
| DSM | GeoTIFF | Ellipsoidal heights covering flight line |

## Output

- **GeoTIFF** with all spectral bands, DEFLATE compressed, tiled,
  georeferenced in the output CRS, with NODATA masking for
  out-of-swath and occluded pixels.

---

## Performance Notes

- Newton convergence: 3–5 iterations typical (1 μs numerical δ, 1 ns threshold)
- Tile parallelism scales linearly with worker count up to I/O saturation
- BIL memory mapping avoids duplication across worker processes
- Coarse initial time guess via decimated trajectory search (~10 Hz)
- Vectorised numpy operations throughout (no per-pixel Python loops)
