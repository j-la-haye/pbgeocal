# -*- coding: utf-8 -*-
"""

### 1. Package Structure

Create a folder named `photogrammetry_verify` with the following structure:

```text
photogrammetry_verify/
├── __init__.py           # Makes it a package
├── camera.py             # Intrinsics and Pinhole logic
├── transforms.py         # Rotation/Translation logic
├── geometry.py           # Projection and Ray-tracing algorithms
├── io_utils.py           # YAML and data loading
└── main.py               # The CLI entry point

```

---

### 2. Component Code




### Benefits of this Modular Setup:

1. **Independent Distortion Models:** If you want to use a more complex Equidistant or Fisheye model, you only edit `geometry.py`.
2. **Trajectory Sources:** If you need to add interpolation for trajectory timestamps, you can add a `trajectory.py` module without touching the camera math.
3. **Coordinate Handedness:** Your specific **X-Right, Y-Back** requirement is localized to the `config.yaml` and the `Transform` class, making it easy to swap if the hardware mount changes.

Would you like me to add a unit test file to this package to verify the **X-Right, Y-Back** rotation specifically?
"""