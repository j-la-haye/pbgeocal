#!/usr/bin/env python3
"""
main.py — Pushbroom Orthorectification Pipeline Entry Point.

Bottom-up (output-to-input) orthorectification of hyperspectral pushbroom
imagery using:
  • Precise SBET trajectory (200 Hz, Applanix NED)
  • Per-scanline GPS-synchronised exposure times
  • Digital Surface Model (ellipsoidal heights)
  • Decoupled camera model (lab LUT + smile + in-flight optics correction)
  • Quaternion Slerp for attitude interpolation
  • Newton iteration for time solving
  • Ray-based occlusion detection for true ortho
  • Tile-based parallel processing with ProcessPoolExecutor

Usage
-----
    python main.py                        # uses ./config.yaml
    python main.py /path/to/config.yaml   # custom config path
"""

import sys
import time
from pathlib import Path

from pipeline.config_loader import load_config
from pipeline.tile_processor import run_parallel_ortho


def main():
    config_path = "Orthorectify/pushbroom_ortho/pushbroom_ortho_pipeline.yaml"
    # Resolve config path
    # if len(sys.argv) > 1:
    #     config_path = sys.argv[1]
    # else:
    #     config_path = "Orthorectify/pushbroom_ortho/pushbroom_ortho_pipeline.yaml" #str(Path(__file__).parent / "pushbroom_ortho_pipeline.yaml")

    print(f"Loading configuration: {config_path}")
    config = load_config(config_path)

    t0 = time.time()
    run_parallel_ortho(config, config_path)
    elapsed = time.time() - t0

    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\nTotal processing time: {minutes}m {seconds:.1f}s")


if __name__ == "__main__":
    main()
