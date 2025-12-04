#!/usr/bin/env python3
import yaml
import sys
import os

# Mapping of types to expected number of parameters
PARAM_COUNT = {
    "GEOXY": 2,
    "UV": 2,
    "UVT": 3,
    "GEOXYZ": 3,
    "XYZ": 3,
    "XYZT": 4,
    "PRIORID": 0
}

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_timestamps(file_path):
    """Load a list of timestamps from file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_params_for_type(type_name, prefix):
    """Return parameter names for a given TYPE."""
    n = PARAM_COUNT.get(type_name.upper(), 0)
    if type_name.upper() in ["UV", "UVT"]:
        return [f"{prefix}u", f"{prefix}v", f"{prefix}t"][:n]
    elif n == 2:
        return [f"{prefix}x", f"{prefix}y"]
    elif n == 3:
        return [f"{prefix}x", f"{prefix}y", f"{prefix}z"]
    elif n == 4:
        return [f"{prefix}x", f"{prefix}y", f"{prefix}z", f"{prefix}t"]
    else:
        return []

def build_side(type_name, object_name, mapping, prefix):
    """Construct one side of the output string."""
    params = get_params_for_type(type_name, prefix)
    values = [str(mapping[p]) for p in params if p in mapping]
    return f"{type_name} {object_name}" + (f" {' '.join(values)}" if values else "")

def process_file(input_path, cfg, timestamps_1, timestamps_2):
    """Process one input text file and generate its stevi_match_ output."""
    base_name = os.path.basename(input_path)
    output_path = os.path.join(os.path.dirname(input_path), f"stevi_match_{base_name}")

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) != 4:
                print(f"[WARN] Skipping malformed line in {base_name}: {line.strip()}")
                continue

            # For UVT input: 1u idx1 2u idx2
            mapping = {
                "1u": parts[0],
                "2u": parts[2],
                "1v": "0.0",
                "2v": "0.0"
            }

            try:
                time_scale = float(cfg.get("image_pair")['time_scale'])
                idx1 = int(float(parts[1]))
                idx2 = int(float(parts[3]))
                mapping["1t"] = int(timestamps_1[idx1]) * time_scale
                mapping["2t"] = int(timestamps_2[idx2]) * time_scale
            except (IndexError, ValueError):
                print(f"[WARN] Timestamp index out of range in {base_name}: {line.strip()}")
                continue

            # Get image names from image_pair config
            img_cfg = cfg.get("image_pair", {}) 
            #mapping["1img"] = img_cfg["image1_name"]
            #mapping["2img"] = img_cfg["image2_name"]
            # Apply all templates
            for template in cfg.get("templates", []):
                lhs = build_side(template['type1'], img_cfg["image1_name"], mapping, "1")
                rhs = build_side(template['type2'], img_cfg["image2_name"], mapping, "2")
                fout.write(f"{lhs}, {rhs}\n")

    print(f"✅ Created: {output_path}")

def main():
    #if len(sys.argv) != 2:
    #    print("Usage: python generate_template_uvt_dir.py <config.yaml>")
    #    sys.exit(1)

    yaml_file = "configs/stevi_match.yaml" #sys.argv[1]
    cfg = load_yaml_config(yaml_file)

    root_dir = cfg.get("root_dir")
    if not root_dir or not os.path.isdir(root_dir):
        print(f"❌ Error: root_dir not defined or invalid in YAML.")
        sys.exit(1)

    img_cfg = cfg.get("image_pair", {})
    # define tfile1 and tfile2 as file ending with .timing in tdir1 and tdir2 respectively
    tfile1 = os.path.join(img_cfg.get("image1_timing"), next((f for f in os.listdir(img_cfg.get("image1_timing")) if f.endswith(".timing")), None))
    tfile2 = os.path.join(img_cfg.get("image2_timing"), next((f for f in os.listdir(img_cfg.get("image2_timing")) if f.endswith(".timing")), None))

    if not (tfile1 and tfile2):
        print("❌ Timing files not specified in YAML under image_pair.")
        sys.exit(1)

    timestamps_1 = load_timestamps(tfile1)
    timestamps_2 = load_timestamps(tfile2)

    txt_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                 if f.endswith(".txt") and not f.startswith("stevi_match_")]

    if not txt_files:
        print(f"⚠️ No .txt files found in {root_dir}.")
        sys.exit(0)

    for file_path in txt_files:
        process_file(file_path, cfg, timestamps_1, timestamps_2)

    print("🎯 All matching files processed successfully.")

if __name__ == "__main__":
    main()
