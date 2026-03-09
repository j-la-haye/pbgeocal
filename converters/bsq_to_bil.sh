#!/usr/bin/env bash
# ============================================================================
# bsq_to_bil.sh — Convert ENVI BSQ data to BIL interleave using gdal_translate
#                  and regenerate a correct ENVI .hdr for the output.
#
# Usage:
#   ./bsq_to_bil.sh <input.bsq> [output.bil]
#
# If output path is omitted the script writes <input_base>_bil.bil next to
# the source file.
#
# Requirements:
#   - gdal_translate  (GDAL >= 2.x)
#   - python3 with the `spectral` package  (pip install spectral)
# ============================================================================
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 <input_bsq_file> [output_bil_file]"
    echo ""
    echo "  input_bsq_file   Path to the ENVI BSQ data file (.bsq / .dat / raw)"
    echo "  output_bil_file   (optional) Output BIL path. Default: <input>_bil.bil"
    exit 1
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ── Argument handling ────────────────────────────────────────────────────────
[[ $# -lt 1 ]] && usage

INPUT="$1"
[[ -f "$INPUT" ]] || die "Input file not found: $INPUT"

# Resolve the companion .hdr for the input (try common naming conventions)
INPUT_DIR="$(dirname "$INPUT")"
INPUT_BASE="$(basename "$INPUT")"
INPUT_STEM="${INPUT_BASE%.*}"

INPUT_HDR=""
for candidate in \
    "${INPUT_DIR}/${INPUT_STEM}.hdr" \
    "${INPUT_DIR}/${INPUT_BASE}.hdr" \
    "${INPUT_DIR}/${INPUT_STEM}.HDR"; do
    if [[ -f "$candidate" ]]; then
        INPUT_HDR="$candidate"
        break
    fi
done
[[ -n "$INPUT_HDR" ]] || die "Cannot find ENVI .hdr for $INPUT (tried ${INPUT_STEM}.hdr / ${INPUT_BASE}.hdr)"

# Verify the source is actually BSQ
if ! grep -qi 'interleave\s*=\s*bsq' "$INPUT_HDR"; then
    echo "WARNING: Input header does not declare interleave = bsq. Proceeding anyway."
fi

# Output path
if [[ $# -ge 2 ]]; then
    OUTPUT="$2"
else
    OUTPUT="${INPUT_DIR}/${INPUT_STEM}_bil.bil"
fi
OUTPUT_DIR="$(dirname "$OUTPUT")"
OUTPUT_BASE="$(basename "$OUTPUT")"
OUTPUT_STEM="${OUTPUT_BASE%.*}"
OUTPUT_HDR="${OUTPUT_DIR}/${OUTPUT_STEM}.hdr"

echo "────────────────────────────────────────"
echo "  Input BSQ  : $INPUT"
echo "  Input HDR  : $INPUT_HDR"
echo "  Output BIL : $OUTPUT"
echo "  Output HDR : $OUTPUT_HDR"
echo "────────────────────────────────────────"

# ── Step 1: gdal_translate BSQ → BIL ────────────────────────────────────────
echo "[1/2] Running gdal_translate (BSQ → BIL) ..."

gdal_translate \
    -of ENVI \
    -co "INTERLEAVE=BIL" \
    "$INPUT" \
    "$OUTPUT"

echo "      gdal_translate finished."

# GDAL writes its own .hdr but it can be missing fields that downstream ENVI
# readers expect (wavelengths, fwhm, sensor type, custom metadata …).
# We use the Python spectral package to read the ORIGINAL header, patch the
# interleave, and write a complete .hdr alongside the new BIL file.

# ── Step 2: Regenerate a complete ENVI .hdr via spectral ────────────────────
echo "[2/2] Generating complete ENVI .hdr from original metadata ..."

python3 - "$INPUT_HDR" "$OUTPUT_HDR" "$OUTPUT" <<'PYEOF'
"""
Read the original ENVI header, update interleave and file paths, then write
a new header for the BIL file.  Uses spectral.io.envi for robust parsing.
"""
import sys
import os

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit("ERROR: python spectral package not found. Install with:  pip install spectral")

src_hdr  = sys.argv[1]   # original .hdr (BSQ)
dst_hdr  = sys.argv[2]   # target .hdr  (BIL)
dst_data = sys.argv[3]   # target data file path

# ── Parse the original header ───────────────────────────────────────────
# spectral.io.envi.read_envi_header returns an OrderedDict of all fields
metadata = envi.read_envi_header(src_hdr)

# ── Patch fields for BIL output ─────────────────────────────────────────
metadata['interleave'] = 'bil'

# If header_offset was non-zero in the source, the new GDAL file has none
metadata['header offset'] = '0'

# Some ENVI headers carry an absolute data file path — update it
if 'data file' in metadata:
    metadata['data file'] = os.path.abspath(dst_data)

# Remove GDAL-injected keys that may conflict (GDAL sometimes adds these)
for key in ['description', 'file_type']:
    pass  # keep them if present; they are harmless

# ── Write the new header ────────────────────────────────────────────────
# spectral doesn't have a public write_envi_header for arbitrary dicts,
# so we write it manually following ENVI spec.

def format_value(v):
    """Format a header value: lists get {braces}, strings stay as-is."""
    if isinstance(v, list):
        inner = ", ".join(str(i).strip() for i in v)
        return "{\n  " + inner + "\n}"
    return str(v).strip()

with open(dst_hdr, 'w') as f:
    f.write("ENVI\n")
    for key, val in metadata.items():
        f.write(f"{key} = {format_value(val)}\n")

print(f"      Wrote {dst_hdr}  ({sum(1 for _ in metadata)} fields)")
PYEOF

echo ""
echo "Done.  Output files:"
echo "  Data : $OUTPUT"
echo "  HDR  : $OUTPUT_HDR"
echo "────────────────────────────────────────"
