#!/usr/bin/env python3
"""
pca_envi.py — PCA on an ENVI hyperspectral image (BSQ or BIL).

Retains only the principal components that cumulatively explain a target
fraction of the total variance (default 95 %) and writes the result as an
ENVI BIL file with a complete .hdr.

All parameters are read from a YAML configuration file.

Usage
-----
    python pca_envi.py config.yaml
    python pca_envi.py                   # looks for pca_config.yaml in cwd

Requirements
------------
    pip install spectral numpy pyyaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("ERROR: 'pyyaml' package not found.  Install with:  pip install pyyaml")

try:
    import spectral
    import spectral.io.envi as envi
except ImportError:
    sys.exit("ERROR: 'spectral' package not found.  Install with:  pip install spectral")


# ── Default configuration ────────────────────────────────────────────────────
DEFAULTS = {
    "input_hdr":        None,       # required
    "output":           None,       # stem without extension; default: <input>_pca
    "variance":         0.95,       # cumulative explained-variance fraction
    "nodata_value":     0.0,        # pixels whose bands all equal this are masked
    "output_dtype":     "float32",  # float32 | float64
    "output_interleave": "bil",     # bil | bsq | bip
    "max_print_components": 20,     # how many PCs to show in the summary table
    "validate":         True,       # cross-check with spectral.principal_components
    "top_bands":        20,         # number of top contributing bands to report
    "write_top_bands":  True,       # write top bands as a separate ENVI image
}

EXAMPLE_YAML = """\
# ── pca_config.yaml ─────────────────────────────────────────────────────
# Configuration for pca_envi.py
# All paths can be absolute or relative to this YAML file's directory.

# (required) Path to the ENVI .hdr file
input_hdr: /data/flight01/scene.hdr

# (optional) Output stem without extension.  Default: <input_stem>_pca
# output: /data/flight01/scene_pca

# Cumulative explained-variance fraction to retain  (0.0 – 1.0)
variance: 0.95

# Pixel value treated as no-data (all bands equal this → masked)
nodata_value: 0.0

# Output numpy dtype:  float32 | float64
output_dtype: float32

# Output interleave:  bil | bsq | bip
output_interleave: bil

# How many components to print in the variance summary table
max_print_components: 20

# Cross-validate against spectral.principal_components()
validate: true

# Number of top contributing input bands to report
top_bands: 20

# Write the top contributing bands as a separate ENVI image
write_top_bands: true
"""


# ── Config loading ───────────────────────────────────────────────────────────
def load_config(yaml_path: str) -> dict:
    """Load and validate the YAML configuration file."""
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.is_file():
        sys.exit(f"ERROR: Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    # Merge with defaults
    cfg = {**DEFAULTS, **raw}

    # ── Validation ───────────────────────────────────────────────────────
    if cfg["input_hdr"] is None:
        sys.exit("ERROR: 'input_hdr' is required in the YAML config.")

    # Resolve paths relative to the YAML file's directory
    yaml_dir = yaml_path.parent
    cfg["input_hdr"] = str((yaml_dir / cfg["input_hdr"]).resolve())

    if cfg["output"] is not None:
        cfg["output"] = str((yaml_dir / cfg["output"]).resolve())

    # Type coercions
    cfg["variance"]  = float(cfg["variance"])
    cfg["nodata_value"] = float(cfg["nodata_value"])
    cfg["max_print_components"] = int(cfg["max_print_components"])
    cfg["validate"]  = bool(cfg["validate"])
    cfg["top_bands"] = int(cfg["top_bands"])
    cfg["write_top_bands"] = bool(cfg["write_top_bands"])

    if not 0.0 < cfg["variance"] <= 1.0:
        sys.exit(f"ERROR: 'variance' must be in (0, 1], got {cfg['variance']}")

    if cfg["output_dtype"] not in ("float32", "float64"):
        sys.exit(f"ERROR: 'output_dtype' must be float32 or float64, got {cfg['output_dtype']}")

    if cfg["output_interleave"] not in ("bil", "bsq", "bip"):
        sys.exit(f"ERROR: 'output_interleave' must be bil/bsq/bip, got {cfg['output_interleave']}")

    return cfg


def parse_args():
    """Minimal CLI: just the path to the YAML config (or --example)."""
    p = argparse.ArgumentParser(
        description="PCA on ENVI hyperspectral image.  Parameters via YAML config.",
    )
    p.add_argument(
        "config",
        nargs="?",
        default="pca_config.yaml",
        help="Path to YAML configuration file (default: pca_config.yaml)",
    )
    p.add_argument(
        "--example",
        action="store_true",
        help="Print an example YAML config to stdout and exit.",
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────
ENVI_DTYPE_MAP = {
    np.dtype("uint8"):    1,
    np.dtype("int16"):    2,
    np.dtype("int32"):    3,
    np.dtype("float32"):  4,
    np.dtype("float64"):  5,
    np.dtype("uint16"):  12,
    np.dtype("uint32"):  13,
    np.dtype("int64"):   14,
    np.dtype("uint64"):  15,
}


def envi_typecode(dt: np.dtype) -> int:
    """Map a numpy dtype to its ENVI data type code."""
    dt = np.dtype(dt)
    if dt in ENVI_DTYPE_MAP:
        return ENVI_DTYPE_MAP[dt]
    raise ValueError(f"No ENVI type code for dtype {dt}")


# ── Validation via spectral.principal_components ─────────────────────────────
def validate_with_spectral_pca(data, eigenvalues, eigenvectors, mean,
                               n_components, target_variance):
    """
    Run spectral.principal_components() on the same image data and compare
    eigenvalues, eigenvectors, and the reduced component count against the
    custom implementation.

    Parameters
    ----------
    data : np.ndarray
        The loaded image array, shape (rows, cols, bands).
    eigenvalues : np.ndarray
        Eigenvalues from the custom implementation (descending).
    eigenvectors : np.ndarray
        Eigenvectors from the custom implementation, shape (bands, bands),
        columns are eigenvectors.
    mean : np.ndarray
        Mean spectrum from the custom implementation, length bands.
    n_components : int
        Number of components retained by the custom implementation.
    target_variance : float
        Target cumulative variance fraction.

    Returns
    -------
    dict with comparison metrics, or None on failure.
    """
    print()
    print("─" * 52)
    print("  VALIDATION: spectral.principal_components()")
    print("─" * 52)

    try:
        pc = spectral.principal_components(data)
    except Exception as e:
        print(f"  WARNING: spectral.principal_components() failed: {e}")
        return None

    spy_eigenvalues  = np.real(pc.eigenvalues)
    spy_eigenvectors = np.real(pc.eigenvectors)
    spy_mean         = np.real(pc.stats.mean)

    bands = len(spy_eigenvalues)

    # ── Mean spectrum comparison ─────────────────────────────────────────
    # spectral uses ALL pixels (no nodata mask), so means may differ;
    # report the magnitude of the difference.
    mean_diff = np.linalg.norm(mean - spy_mean)
    mean_reldiff = mean_diff / (np.linalg.norm(spy_mean) + 1e-30)
    print(f"  Mean spectrum L2 diff       : {mean_diff:.6g}  "
          f"(relative: {mean_reldiff:.4e})")
    if mean_reldiff > 0.01:
        print("  NOTE: Mean differs because spectral uses ALL pixels "
              "(including nodata).")
        print("        Custom implementation masks nodata pixels. This is expected.")

    # ── Eigenvalue comparison ────────────────────────────────────────────
    n_compare = min(len(eigenvalues), len(spy_eigenvalues))
    ev_custom = eigenvalues[:n_compare]
    ev_spy    = spy_eigenvalues[:n_compare]

    # Relative differences per eigenvalue
    denom = np.maximum(np.abs(ev_spy), 1e-30)
    ev_reldiff = np.abs(ev_custom - ev_spy) / denom

    print(f"\n  Eigenvalue comparison (top {min(n_compare, 10)}):")
    print(f"  {'PC':>4s}   {'Custom':>14s}   {'Spectral':>14s}   {'RelDiff':>10s}")
    print(f"  {'───':>4s}   {'──────':>14s}   {'────────':>14s}   {'───────':>10s}")
    for i in range(min(n_compare, 10)):
        print(f"  {i+1:4d}   {ev_custom[i]:14.6g}   {ev_spy[i]:14.6g}   "
              f"{ev_reldiff[i]:10.4e}")

    max_ev_reldiff = np.max(ev_reldiff[:min(n_compare, n_components)])
    print(f"\n  Max eigenvalue rel. diff (top {n_components} PCs): "
          f"{max_ev_reldiff:.4e}")

    # ── Eigenvector comparison ───────────────────────────────────────────
    # Eigenvectors may have opposite sign; compare via absolute dot product
    dot_products = np.zeros(min(n_compare, n_components))
    for i in range(len(dot_products)):
        v_custom = eigenvectors[:, i]
        v_spy    = spy_eigenvectors[:, i]
        dot_products[i] = np.abs(np.dot(v_custom, v_spy))

    print(f"\n  Eigenvector alignment |dot product| (top {len(dot_products)}):")
    print(f"  {'PC':>4s}   {'|v_custom · v_spy|':>20s}   {'Angle (deg)':>12s}")
    print(f"  {'───':>4s}   {'──────────────────':>20s}   {'───────────':>12s}")
    for i in range(min(len(dot_products), 10)):
        angle = np.degrees(np.arccos(np.clip(dot_products[i], 0.0, 1.0)))
        print(f"  {i+1:4d}   {dot_products[i]:20.10f}   {angle:12.6f}")

    min_dot = np.min(dot_products)
    print(f"\n  Min eigenvector |dot| across retained PCs: {min_dot:.10f}")

    # ── Reduced component count comparison ───────────────────────────────
    pc_reduced = pc.reduce(fraction=target_variance)
    spy_n_components = len(pc_reduced.eigenvalues)
    print(f"\n  Components for {target_variance:.2%} variance:")
    print(f"    Custom   : {n_components}")
    print(f"    Spectral : {spy_n_components}")
    match = "MATCH" if n_components == spy_n_components else "DIFFER"
    print(f"    Result   : {match}")

    # ── Overall verdict ──────────────────────────────────────────────────
    ev_ok  = max_ev_reldiff < 0.01   # eigenvalues within 1%
    vec_ok = min_dot > 0.999         # eigenvectors within ~2.6°
    if ev_ok and vec_ok:
        print("\n  ✓ VALIDATION PASSED — custom PCA matches spectral package")
    else:
        print("\n  ⚠ VALIDATION WARNING — differences detected (may be due to "
              "nodata masking)")
        if not ev_ok:
            print(f"    Eigenvalue max rel. diff {max_ev_reldiff:.4e} > 1%")
        if not vec_ok:
            print(f"    Eigenvector min |dot| {min_dot:.10f} < 0.999")

    print("─" * 52)

    return {
        "mean_reldiff":     mean_reldiff,
        "max_ev_reldiff":   max_ev_reldiff,
        "min_dot_product":  min_dot,
        "spy_n_components": spy_n_components,
    }


# ── Band contribution analysis ───────────────────────────────────────────────
def compute_band_contributions(eigenvalues, eigenvectors, n_components,
                               bands, metadata, top_n=20, out_stem=None):
    """
    Compute and report which input bands contribute most to the retained PCs.

    The importance score for band j is the variance-weighted sum of squared
    loadings across the retained components:

        importance_j = Σ_k  λ_k · |v_{jk}|²      for k = 1..n_components

    This measures how much of the total *retained* variance is attributable
    to each original band.

    Parameters
    ----------
    eigenvalues : array, shape (B,)
    eigenvectors : array, shape (B, B), columns are eigenvectors
    n_components : int
    bands : int, total number of bands
    metadata : dict, original ENVI header metadata
    top_n : int
    out_stem : str or None, if given writes a CSV report

    Returns
    -------
    top_indices : array of band indices (0-based), length top_n
    importance  : array of importance scores, length bands
    """
    W = eigenvectors[:, :n_components]                    # (bands, n_comp)
    lam = eigenvalues[:n_components]                      # (n_comp,)

    # Variance-weighted squared loadings:  importance[j] = Σ_k λ_k * W[j,k]²
    importance = (W ** 2) @ lam                           # (bands,)
    importance_norm = importance / importance.sum()        # fraction of retained variance

    # Sort descending
    rank_idx = np.argsort(importance)[::-1]
    top_n = min(top_n, bands)
    top_indices = rank_idx[:top_n]

    # Try to get wavelength info from header
    wavelengths = None
    wl_units = metadata.get("wavelength units", "")
    if "wavelength" in metadata:
        try:
            wavelengths = np.array([float(w) for w in metadata["wavelength"]])
        except (ValueError, TypeError):
            pass

    # Try to get band names from header
    band_names_hdr = None
    if "band names" in metadata:
        band_names_hdr = metadata["band names"]

    # ── Print report ─────────────────────────────────────────────────────
    print()
    print("─" * 68)
    print(f"  TOP {top_n} CONTRIBUTING BANDS  "
          f"(variance-weighted squared loadings)")
    print("─" * 68)

    has_wl = wavelengths is not None and len(wavelengths) == bands

    if has_wl:
        hdr = (f"  {'Rank':>4s}   {'Band':>5s}   {'Wavelength':>12s} "
               f"{'':>3s}   {'Importance':>11s}   {'Cumul%':>7s}   "
               f"{'Top PC loadings'}")
    else:
        hdr = (f"  {'Rank':>4s}   {'Band':>5s}   {'Importance':>11s}   "
               f"{'Cumul%':>7s}   {'Top PC loadings'}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    cum = 0.0
    for rank, bi in enumerate(top_indices):
        cum += importance_norm[bi]
        # Top 3 PC loadings for this band (absolute value, with sign)
        loadings_str = "  ".join(
            f"PC{k+1}:{W[bi, k]:+.3f}"
            for k in range(min(3, n_components))
        )
        if has_wl:
            print(f"  {rank+1:4d}   {bi:5d}   {wavelengths[bi]:10.2f} "
                  f"{wl_units:>3s}   {importance_norm[bi]:11.6f}   "
                  f"{cum*100:6.2f}%   {loadings_str}")
        else:
            print(f"  {rank+1:4d}   {bi:5d}   {importance_norm[bi]:11.6f}   "
                  f"{cum*100:6.2f}%   {loadings_str}")

    print("─" * 68)

    # ── Write CSV report ─────────────────────────────────────────────────
    if out_stem is not None:
        csv_path = out_stem + "_top_bands.csv"
        with open(csv_path, "w") as f:
            # Header
            cols = ["rank", "band_index", "band_number"]
            if has_wl:
                cols += ["wavelength", "wavelength_units"]
            if band_names_hdr is not None and len(band_names_hdr) == bands:
                cols += ["band_name"]
            cols += ["importance", "importance_fraction", "cumulative_fraction"]
            for k in range(min(n_components, 10)):
                cols.append(f"loading_PC{k+1}")
            f.write(",".join(cols) + "\n")

            # Data rows (all bands, sorted by importance)
            cum = 0.0
            for rank, bi in enumerate(rank_idx):
                cum += importance_norm[bi]
                row = [str(rank + 1), str(bi), str(bi + 1)]
                if has_wl:
                    row += [f"{wavelengths[bi]:.4f}", wl_units]
                if band_names_hdr is not None and len(band_names_hdr) == bands:
                    # Escape commas in band names
                    bname = str(band_names_hdr[bi]).replace('"', '""')
                    row.append(f'"{bname}"')
                row += [
                    f"{importance[bi]:.8g}",
                    f"{importance_norm[bi]:.8g}",
                    f"{cum:.8g}",
                ]
                for k in range(min(n_components, 10)):
                    row.append(f"{W[bi, k]:.8g}")
                f.write(",".join(row) + "\n")

        print(f"  Band contribution report: {csv_path}")
        print(f"  (all {bands} bands ranked, top {min(n_components, 10)} PC loadings)")

    return top_indices, importance


# ── Write top contributing bands as a new ENVI image ─────────────────────────
def write_top_bands_image(img, top_indices, metadata, out_stem,
                          out_interleave="bil"):
    """
    Extract the top contributing bands from the original image and write
    them to a new ENVI BIL (or BSQ/BIP) file using spectral.io.envi.

    The output preserves the original data type, spatial metadata, and
    carries over per-band metadata (wavelengths, fwhm, band names, bbl)
    for the selected bands only, in wavelength-sorted order.

    Parameters
    ----------
    img : spectral.SpyFile
        The opened original image (used for read_bands).
    top_indices : array-like of int
        0-based band indices of the top contributing bands.
    metadata : dict
        Original ENVI header metadata.
    out_stem : str
        Output file stem (without extension).
    out_interleave : str
        Output interleave format ('bil', 'bsq', 'bip').
    """
    top_indices = list(top_indices)
    n_sel = len(top_indices)

    # Sort selected bands by original band index (= wavelength order) so the
    # output image has monotonically increasing wavelength
    sorted_indices = sorted(int(i) for i in top_indices)

    print(f"\n  Extracting {n_sel} top bands from original image ...")
    print(f"  Band indices (0-based, wavelength order): {sorted_indices}")

    # ── Read selected bands ──────────────────────────────────────────────
    selected_data = img.read_bands(sorted_indices)    # shape (rows, cols, n_sel)

    # ── Build metadata for the band-selected output ──────────────────────
    sel_meta = {}

    # Description
    sel_meta["description"] = (
        f"Top {n_sel} PCA-contributing bands from "
        f"{os.path.basename(metadata.get('description', 'source'))}"
    )

    # Core dimensions (envi.save_image infers these from the array, but we
    # set them explicitly to be safe)
    rows, cols = selected_data.shape[0], selected_data.shape[1]
    sel_meta["samples"] = str(cols)
    sel_meta["lines"]   = str(rows)
    sel_meta["bands"]   = str(n_sel)
    sel_meta["interleave"] = out_interleave

    # Carry over scalar spatial / projection metadata verbatim
    for key in [
        "map info", "coordinate system string", "projection info",
        "x start", "y start", "default stretch",
        "data type", "byte order", "header offset",
        "data ignore value", "reflectance scale factor",
        "sensor type",
    ]:
        if key in metadata:
            sel_meta[key] = metadata[key]

    # Carry over per-band list metadata, sub-selected to sorted_indices
    per_band_keys = ["wavelength", "fwhm", "band names", "bbl",
                     "data gain values", "data offset values"]
    for key in per_band_keys:
        if key in metadata:
            src = metadata[key]
            if isinstance(src, (list, tuple)) and len(src) == img.nbands:
                sel_meta[key] = [src[i] for i in sorted_indices]

    # Wavelength units (scalar, not per-band)
    if "wavelength units" in metadata:
        sel_meta["wavelength units"] = metadata["wavelength units"]

    # Custom field: record which original band indices were selected
    sel_meta["selected band indices"] = [str(i) for i in sorted_indices]

    # ── Write via envi.save_image ────────────────────────────────────────
    out_hdr  = out_stem + "_top_bands.hdr"

    envi.save_image(
        out_hdr,
        selected_data,
        interleave=out_interleave,
        metadata=sel_meta,
        force=True,
    )

    # envi.save_image names the data file based on interleave
    # (e.g. .bil, .bsq, .bip, or .img depending on version).
    # Find the actual data file that was created.
    out_ext  = f".{out_interleave}"
    out_data = out_stem + "_top_bands" + out_ext
    # Fallback: save_image might use .img
    if not os.path.isfile(out_data):
        out_data = out_stem + "_top_bands.img"

    print(f"  Top-bands data : {out_data}")
    print(f"  Top-bands HDR  : {out_hdr}")
    print(f"  Shape          : {rows} × {cols} × {n_sel}  ({out_interleave.upper()})")

    # Print wavelength range if available
    if "wavelength" in sel_meta:
        wls = [float(w) for w in sel_meta["wavelength"]]
        wl_units = sel_meta.get("wavelength units", "")
        print(f"  Wavelength range: {min(wls):.2f} – {max(wls):.2f} {wl_units}")

    return out_hdr, out_data


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    cli = parse_args()

    # --example: dump a sample YAML and exit
    if cli.example:
        print(EXAMPLE_YAML)
        sys.exit(0)

    # Load YAML config
    cfg = load_config(cli.config)

    # Unpack for readability
    input_hdr          = cfg["input_hdr"]
    target_variance    = cfg["variance"]
    nodata_val         = cfg["nodata_value"]
    out_dtype_str      = cfg["output_dtype"]
    out_interleave     = cfg["output_interleave"]
    max_print          = cfg["max_print_components"]
    do_validate        = cfg["validate"]
    top_bands_n        = cfg["top_bands"]
    write_top          = cfg["write_top_bands"]
    out_dtype          = np.dtype(out_dtype_str)

    print(f"Config loaded from: {cli.config}")
    print(f"  variance={target_variance}  nodata={nodata_val}  "
          f"dtype={out_dtype_str}  interleave={out_interleave}")
    print()

    # ── Load image ───────────────────────────────────────────────────────
    print(f"[1/6] Opening {input_hdr} ...")
    img = envi.open(input_hdr)
    meta = img.metadata.copy()
    rows, cols, bands = img.nrows, img.ncols, img.nbands
    print(f"       Image shape: {rows} lines × {cols} samples × {bands} bands")
    print(f"       Interleave : {meta.get('interleave', '?')}")

    # Memory-map the data (spectral returns a SpyFile; .load() brings all to RAM)
    print("[2/6] Loading image data into memory ...")
    data = img.load()                         # shape (rows, cols, bands), numpy array

    # Reshape to 2-D  (n_pixels × bands)
    X = data.reshape(-1, bands).astype(np.float64)

    # Mask out no-data pixels (all bands equal nodata_value)
    valid_mask = np.any(X != nodata_val, axis=1)
    n_valid = int(valid_mask.sum())
    n_total = X.shape[0]
    print(f"       Valid pixels: {n_valid} / {n_total}  "
          f"({100 * n_valid / n_total:.1f} %)")

    X_valid = X[valid_mask]

    # ── Compute PCA (covariance-based) ───────────────────────────────────
    print("[3/6] Computing covariance matrix and eigen-decomposition ...")
    mean = X_valid.mean(axis=0)
    X_centered = X_valid - mean

    # Covariance  (bands × bands)  — efficient for typical band counts
    cov = np.cov(X_centered, rowvar=False)

    # Eigen-decomposition  (sorted descending)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp tiny negative eigenvalues from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total_var = eigenvalues.sum()
    cumvar    = np.cumsum(eigenvalues) / total_var

    # Determine number of components for target variance
    n_components = int(np.searchsorted(cumvar, target_variance) + 1)
    n_components = min(n_components, bands)
    explained   = cumvar[n_components - 1]

    print(f"       Total variance        : {total_var:.6g}")
    print(f"       Target fraction       : {target_variance:.2%}")
    print(f"       Components retained   : {n_components}  "
          f"(explains {explained:.4%})")

    # Print per-component summary
    print()
    print("       PC#   Eigenvalue     Var%     CumVar%")
    print("       ───   ──────────     ────     ───────")
    for i in range(min(n_components, max_print)):
        pct = eigenvalues[i] / total_var * 100
        cpct = cumvar[i] * 100
        print(f"       {i+1:3d}   {eigenvalues[i]:12.4g}   {pct:6.2f}   {cpct:7.3f}")
    if n_components > max_print:
        print(f"       ... ({n_components - max_print} more components)")
    print()

    # ── Validate against spectral.principal_components() ─────────────────
    if do_validate:
        validate_with_spectral_pca(
            data, eigenvalues, eigenvectors, mean,
            n_components, target_variance,
        )

    # ── Band contribution analysis ───────────────────────────────────────
    input_stem = os.path.splitext(input_hdr)[0]
    if cfg["output"] is None:
        out_stem = f"{input_stem}_pca"
    else:
        out_stem = cfg["output"]

    top_indices, importance = compute_band_contributions(
        eigenvalues, eigenvectors, n_components,
        bands, meta, top_n=top_bands_n, out_stem=out_stem,
    )

    # ── Project all pixels ───────────────────────────────────────────────
    print("[4/6] Projecting onto principal components ...")
    W = eigenvectors[:, :n_components]        # (bands × n_components)

    # Centre ALL pixels (including no-data) and project
    X_centered_all = X - mean                 # (n_pixels × bands)
    scores_all = X_centered_all @ W           # (n_pixels × n_components)

    # Zero-out no-data pixels so they stay as background
    scores_all[~valid_mask] = 0.0

    # Reshape back to image
    pca_cube = scores_all.reshape(rows, cols, n_components).astype(out_dtype)

    # ── Write output ENVI file ───────────────────────────────────────────
    print(f"[5/6] Writing output ENVI {out_interleave.upper()} ...")

    out_ext  = f".{out_interleave}"
    out_hdr  = out_stem + ".hdr"
    out_data = out_stem + out_ext

    # Build metadata for the output
    out_meta = {}
    out_meta["description"]    = (
        f"PCA of {os.path.basename(input_hdr)}: "
        f"{n_components} components, {explained:.4%} variance"
    )
    out_meta["samples"]        = str(cols)
    out_meta["lines"]          = str(rows)
    out_meta["bands"]          = str(n_components)
    out_meta["header offset"]  = "0"
    out_meta["data type"]      = str(envi_typecode(out_dtype))
    out_meta["interleave"]     = out_interleave
    out_meta["byte order"]     = "0"   # little-endian on x86

    # Preserve spatial metadata from original if present
    for key in [
        "map info", "coordinate system string", "projection info",
        "x start", "y start", "default stretch",
    ]:
        if key in meta:
            out_meta[key] = meta[key]

    # Band names = PC1 … PCn  with variance %
    band_names = []
    for i in range(n_components):
        pct = eigenvalues[i] / total_var * 100
        band_names.append(f"PC{i+1} ({pct:.2f}%)")
    out_meta["band names"] = band_names

    # Store eigenvalues and mean spectrum as custom fields for reproducibility
    out_meta["pca eigenvalues"] = [f"{v:.8g}" for v in eigenvalues[:n_components]]
    out_meta["pca mean spectrum"] = [f"{v:.8g}" for v in mean]
    out_meta["pca target variance"] = str(target_variance)
    out_meta["pca config file"] = str(Path(cli.config).resolve())

    # Write via spectral
    out_img = envi.create_image(
        out_hdr,
        metadata=out_meta,
        dtype=out_dtype,
        force=True,
        ext=out_ext,
    )

    # Write data via memmap
    mm = out_img.open_memmap(writable=True)
    mm[:] = pca_cube
    del mm   # flush

    print()
    print("═" * 52)
    print(f"  Output data : {out_data}")
    print(f"  Output HDR  : {out_hdr}")
    print(f"  Shape       : {rows} × {cols} × {n_components}  ({out_dtype_str} {out_interleave.upper()})")
    print(f"  Variance    : {explained:.4%}  in {n_components} components")
    print("═" * 52)

    # ── Write top contributing bands as a separate image ─────────────────
    if write_top:
        print(f"\n[6/6] Writing top {len(top_indices)} contributing bands ...")
        write_top_bands_image(
            img, top_indices, meta, out_stem,
            out_interleave=out_interleave,
        )
        print()
        print("═" * 52)
        print("  All outputs written successfully.")
        print("═" * 52)


if __name__ == "__main__":
    main()
