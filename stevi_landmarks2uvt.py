#!/usr/bin/env python3
import yaml
import sys
import os

#!/usr/bin/env python3
import yaml
import sys
import os
import csv

#!/usr/bin/env python3
import yaml
import sys
import os
import csv

def load_yaml_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def resolve_path(root_dir, path):
    """Resolve path relative to root_dir if not absolute."""
    if os.path.isabs(path):
        return path
    if root_dir:
        return os.path.join(root_dir, path)
    return path

def load_timestamps(file_path):
    """Load timestamps from a file (one per line)."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def process_image(image_cfg, root_dir=None):
    name = image_cfg["name"]
    tie_file = resolve_path(root_dir, image_cfg["tiepoint_file"])
    time_file = resolve_path(root_dir, image_cfg["timing_file"])

    if not os.path.isfile(tie_file):
        print(f"❌ Tie-point file not found for {name}: {tie_file}")
        return
    if not os.path.isfile(time_file):
        print(f"❌ Timing file not found for {name}: {time_file}")
        return

    timestamps = load_timestamps(time_file)

    base_name = os.path.basename(tie_file)
    out_path = os.path.join(os.path.dirname(tie_file), f"stevi_uvt_{base_name}")

    print(f"➡️  Processing image '{name}'")
    print(f"    tie-points: {tie_file}")
    print(f"    timings:    {time_file}")
    print(f"    output:     {out_path}")

    with open(tie_file, "r", newline="") as fin, open(out_path, "w") as fout:
        reader = csv.reader(fin)

        # Skip header: e.g. Landmark name,X coord,Y coord
        try:
            header = next(reader)
        except StopIteration:
            print(f"[WARN] Empty tie-point file for {name}: {tie_file}")
            return

        for row_no, row in enumerate(reader, start=2):  # line numbers for messages
            if not row or all(not cell.strip() for cell in row):
                continue

            if len(row) < 3:
                print(f"[WARN] Skipping malformed CSV row {row_no} in {tie_file}: {row}")
                continue

            marker_name = row[0].strip()
            u_str = row[1].strip()
            v_str = row[2].strip()

            # Parse V as float, then use as index into timestamps
            try:
                v_val = float(v_str)
                v_idx = int(round(v_val))
            except ValueError:
                print(
                    f"[WARN] Invalid V value '{v_str}' at row {row_no} in {tie_file}, "
                    f"cannot convert to index."
                )
                continue

            if v_idx < 0 or v_idx >= len(timestamps):
                print(
                    f"[WARN] V index {v_idx} out of range for timing file at row {row_no} "
                    f"in {tie_file} (len={len(timestamps)})"
                )
                continue

            t = timestamps[v_idx]

            # Output format: UVT <image_name> U V T
            fout.write(f"{marker_name}, {u_str}, {v_str}, {t}\n")

    print(f"✅ Created: {out_path}\n")



def main(config_path=None):
    if config_path is None:
        if len(sys.argv) != 2:
            print("Usage: python generate_uvt_from_tiepoints.py <config.yaml>")
            sys.exit(1)

        yaml_file = sys.argv[1]
    else:
        yaml_file = config_path
    cfg = load_yaml_config(yaml_file)

    root_dir = cfg.get("root_dir", None)
    images = cfg.get("images", [])

    if not images:
        print("❌ No images defined in YAML under 'images'.")
        sys.exit(1)

    for img_cfg in images:
        process_image(img_cfg, root_dir=root_dir)

    print("🎯 All images processed.")

if __name__ == "__main__":
    main(config_path="configs/stevmatch2uvt.yaml")

