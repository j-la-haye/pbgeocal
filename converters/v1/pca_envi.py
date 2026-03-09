#!/usr/bin/env python3
"""
pca_envi.py — PCA on an ENVI hyperspectral image (BSQ or BIL).

Designed for large pushbroom images (18–25 GB, 20k–30k lines).  All data-
touching operations stream through the file in configurable line-chunks so
peak RAM is bounded by  chunk_lines × samples × bands × 8 bytes  regardless
of total image size.

All parameters are read from a YAML configuration file.

Usage
-----
    python pca_envi.py config.yaml
    python pca_envi.py                   # looks for pca_config.yaml in cwd
    python pca_envi.py --example         # print a template YAML to stdout

Requirements
------------
    pip install spectral numpy pyyaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("ERROR: 'pyyaml' package not found.  Install with:  pip install pyyaml")

try:
    import spectral
    import spectral.io.envi as envi
    from spectral.algorithms.algorithms import GaussianStats
except ImportError:
    sys.exit("ERROR: 'spectral' package not found.  Install with:  pip install spectral")


# ── Default configuration ────────────────────────────────────────────────────
DEFAULTS = {
    "input_hdr":         None,        # required
    "output":            None,        # stem without extension; default: <input>_pca
    "variance":          0.95,        # cumulative explained-variance fraction
    "nodata_value":      0.0,         # pixels whose bands all equal this are masked
    "output_dtype":      "float32",   # float32 | float64
    "output_interleave": "bil",       # bil | bsq | bip
    "max_print_components": 20,       # how many PCs to show in the summary table
    "validate":          True,        # cross-check with spectral.principal_components
    "top_bands":         20,          # number of top contributing bands to report
    "write_top_bands":   True,        # write top bands as a separate ENVI image
    "chunk_lines":       256,         # lines per I/O chunk  (memory control)
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

# Cross-validate eigen-decomposition against spectral.principal_components()
validate: true

# Number of top contributing input bands to report
top_bands: 20

# Write the top contributing bands as a separate ENVI image
write_top_bands: true

# Number of image lines to process per I/O chunk.
# Controls peak memory:  ~ chunk_lines × samples × bands × 8 bytes.
# For a 320-band, 1000-sample image: 256 lines ≈ 625 MB working memory.
# Reduce if running low on RAM; increase for faster throughput.
chunk_lines: 256
"""


# ── Config loading ───────────────────────────────────────────────────────────
def load_config(yaml_path: str) -> dict:
    """Load and validate the YAML configuration file."""
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.is_file():
        sys.exit(f"ERROR: Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULTS, **raw}

    if cfg["input_hdr"] is None:
        sys.exit("ERROR: 'input_hdr' is required in the YAML config.")

    yaml_dir = yaml_path.parent
    cfg["input_hdr"] = str((yaml_dir / cfg["input_hdr"]).resolve())

    if cfg["output"] is not None:
        cfg["output"] = str((yaml_dir / cfg["output"]).resolve())

    cfg["variance"]             = float(cfg["variance"])
    cfg["nodata_value"]         = float(cfg["nodata_value"])
    cfg["max_print_components"] = int(cfg["max_print_components"])
    cfg["validate"]             = bool(cfg["validate"])
    cfg["top_bands"]            = int(cfg["top_bands"])
    cfg["write_top_bands"]      = bool(cfg["write_top_bands"])
    cfg["chunk_lines"]          = int(cfg["chunk_lines"])

    if not 0.0 < cfg["variance"] <= 1.0:
        sys.exit(f"ERROR: 'variance' must be in (0, 1], got {cfg['variance']}")
    if cfg["output_dtype"] not in ("float32", "float64"):
        sys.exit(f"ERROR: 'output_dtype' must be float32 or float64, "
                 f"got {cfg['output_dtype']}")
    if cfg["output_interleave"] not in ("bil", "bsq", "bip"):
        sys.exit(f"ERROR: 'output_interleave' must be bil/bsq/bip, "
                 f"got {cfg['output_interleave']}")
    if cfg["chunk_lines"] < 1:
        sys.exit(f"ERROR: 'chunk_lines' must be >= 1, got {cfg['chunk_lines']}")

    return cfg


def parse_args():
    p = argparse.ArgumentParser(
        description="PCA on ENVI hyperspectral image.  Parameters via YAML config.",
    )
    p.add_argument(
        "config", nargs="?", default="pca_config.yaml",
        help="Path to YAML configuration file (default: pca_config.yaml)",
    )
    p.add_argument(
        "--example", action="store_true",
        help="Print an example YAML config to stdout and exit.",
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────
ENVI_DTYPE_MAP = {
    np.dtype("uint8"):    1,   np.dtype("int16"):    2,
    np.dtype("int32"):    3,   np.dtype("float32"):  4,
    np.dtype("float64"):  5,   np.dtype("uint16"):  12,
    np.dtype("uint32"):  13,   np.dtype("int64"):   14,
    np.dtype("uint64"):  15,
}


def envi_typecode(dt: np.dtype) -> int:
    dt = np.dtype(dt)
    if dt in ENVI_DTYPE_MAP:
        return ENVI_DTYPE_MAP[dt]
    raise ValueError(f"No ENVI type code for dtype {dt}")


def _line_chunks(rows, chunk_lines):
    """Yield (r0, r1) tuples partitioning [0, rows) into chunks."""
    for r0 in range(0, rows, chunk_lines):
        yield r0, min(r0 + chunk_lines, rows)


def _progress(label, r0, rows):
    pct = min(100.0, 100.0 * r0 / rows)
    print(f"\r       {label}  {pct:5.1f}%  (line {r0}/{rows})",
          end="", flush=True)


def _progress_done(label, rows, elapsed):
    print(f"\r       {label}  100.0%  (line {rows}/{rows})  "
          f"[{elapsed:.1f}s]          ")


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 + 2 :  Chunked mean and covariance
# ══════════════════════════════════════════════════════════════════════════════
def compute_stats_chunked(img, nodata_val, chunk_lines):
    """
    Two-pass streaming computation of per-band mean and covariance.

    Pass 1:  accumulate Σ x_i  and  n_valid  →  mean = Σ / n
    Pass 2:  accumulate Σ (x - mean)^T (x - mean)  →  cov / (n - 1)

    Peak memory  ≈  chunk_lines × cols × bands × 8  bytes.
    """
    rows, cols, bands = img.nrows, img.ncols, img.nbands

    # ── Pass 1: mean ─────────────────────────────────────────────────────
    pixel_sum = np.zeros(bands, dtype=np.float64)
    n_valid   = 0

    t0 = time.time()
    for r0, r1 in _line_chunks(rows, chunk_lines):
        _progress("Pass 1 (mean)", r0, rows)
        chunk = np.asarray(
            img.read_subregion((r0, r1), (0, cols)),
            dtype=np.float64,
        )                                             # (nlines, cols, bands)
        X = chunk.reshape(-1, bands)                  # (nlines*cols, bands)
        valid = np.any(X != nodata_val, axis=1)
        if valid.any():
            pixel_sum += X[valid].sum(axis=0)
            n_valid   += int(valid.sum())
        del chunk, X, valid                           # free immediately

    _progress_done("Pass 1 (mean)", rows, time.time() - t0)

    if n_valid == 0:
        sys.exit("ERROR: No valid (non-nodata) pixels found in the image.")

    mean = pixel_sum / n_valid

    # ── Pass 2: covariance ───────────────────────────────────────────────
    cov_accum = np.zeros((bands, bands), dtype=np.float64)

    t0 = time.time()
    for r0, r1 in _line_chunks(rows, chunk_lines):
        _progress("Pass 2 (cov) ", r0, rows)
        chunk = np.asarray(
            img.read_subregion((r0, r1), (0, cols)),
            dtype=np.float64,
        )
        X = chunk.reshape(-1, bands)
        valid = np.any(X != nodata_val, axis=1)
        if valid.any():
            Xc = X[valid] - mean                      # (n_chunk, bands)
            cov_accum += Xc.T @ Xc                    # BLAS dgemm  → (B, B)
            del Xc
        del chunk, X, valid

    _progress_done("Pass 2 (cov) ", rows, time.time() - t0)

    cov = cov_accum / (n_valid - 1)
    return mean, cov, n_valid


# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 :  Chunked projection  →  write PCA image
# ══════════════════════════════════════════════════════════════════════════════
def project_and_write_chunked(img, mean, W, nodata_val,
                              out_img, out_dtype, chunk_lines):
    """
    Project image onto PCA eigenvectors and write the result chunk-by-chunk
    via the output file's writable memmap.

    Peak memory  ≈  chunk_lines × cols × bands × 8   (input chunk)
                  + chunk_lines × cols × n_comp × 8   (output chunk)
    """
    rows, cols = img.nrows, img.ncols
    bands = W.shape[0]
    n_comp = W.shape[1]

    mm = out_img.open_memmap(writable=True)

    t0 = time.time()
    for r0, r1 in _line_chunks(rows, chunk_lines):
        _progress("Projecting   ", r0, rows)
        nlines = r1 - r0
        chunk = np.asarray(
            img.read_subregion((r0, r1), (0, cols)),
            dtype=np.float64,
        )
        X = chunk.reshape(-1, bands)
        del chunk

        valid  = np.any(X != nodata_val, axis=1)
        scores = (X - mean) @ W                       # (N, n_comp)
        del X
        scores[~valid] = 0.0
        del valid

        mm[r0:r1] = scores.reshape(nlines, cols, n_comp).astype(out_dtype)
        del scores

    del mm   # flush
    _progress_done("Projecting   ", rows, time.time() - t0)


# ══════════════════════════════════════════════════════════════════════════════
# PASS 4 :  Chunked top-bands extraction
# ══════════════════════════════════════════════════════════════════════════════
def write_top_bands_chunked(img, sorted_indices, metadata, out_stem,
                            out_interleave, chunk_lines):
    """
    Extract selected bands from the original image and write them to a new
    ENVI file, streaming chunk-by-chunk.
    """
    n_sel = len(sorted_indices)
    rows, cols = img.nrows, img.ncols

    print(f"\n  Extracting {n_sel} top bands  "
          f"(chunked, {chunk_lines} lines/chunk)")
    print(f"  Band indices (0-based, wavelength order): {sorted_indices}")

    # ── Build metadata ───────────────────────────────────────────────────
    sel_meta = {}
    sel_meta["description"] = (
        f"Top {n_sel} PCA-contributing bands from "
        f"{os.path.basename(metadata.get('description', 'source'))}"
    )
    sel_meta["samples"]    = str(cols)
    sel_meta["lines"]      = str(rows)
    sel_meta["bands"]      = str(n_sel)
    sel_meta["interleave"] = out_interleave

    for key in [
        "map info", "coordinate system string", "projection info",
        "x start", "y start", "default stretch",
        "data type", "byte order", "header offset",
        "data ignore value", "reflectance scale factor", "sensor type",
    ]:
        if key in metadata:
            sel_meta[key] = metadata[key]

    for key in ["wavelength", "fwhm", "band names", "bbl",
                "data gain values", "data offset values"]:
        if key in metadata:
            src = metadata[key]
            if isinstance(src, (list, tuple)) and len(src) == img.nbands:
                sel_meta[key] = [src[i] for i in sorted_indices]

    if "wavelength units" in metadata:
        sel_meta["wavelength units"] = metadata["wavelength units"]
    sel_meta["selected band indices"] = [str(i) for i in sorted_indices]

    # ── Create output & stream ───────────────────────────────────────────
    out_hdr = out_stem + "_top_bands.hdr"
    out_ext = f".{out_interleave}"

    src_dtype = np.float32
    try:
        if hasattr(img, 'dtype'):
            src_dtype = np.dtype(img.dtype)
    except TypeError:
        pass

    out_file = envi.create_image(
        out_hdr, metadata=sel_meta, dtype=src_dtype,
        force=True, ext=out_ext,
    )
    mm = out_file.open_memmap(writable=True)

    t0 = time.time()
    for r0, r1 in _line_chunks(rows, chunk_lines):
        _progress("Top bands    ", r0, rows)
        chunk = np.asarray(
            img.read_subregion((r0, r1), (0, cols), sorted_indices),
        )
        mm[r0:r1] = chunk.astype(src_dtype)
        del chunk

    del mm
    _progress_done("Top bands    ", rows, time.time() - t0)

    out_data = out_stem + "_top_bands" + out_ext
    if not os.path.isfile(out_data):
        out_data = out_stem + "_top_bands.img"

    print(f"  Top-bands data : {out_data}")
    print(f"  Top-bands HDR  : {out_hdr}")
    print(f"  Shape          : {rows} × {cols} × {n_sel}  "
          f"({out_interleave.upper()})")

    if "wavelength" in sel_meta:
        wls = [float(w) for w in sel_meta["wavelength"]]
        wl_units = sel_meta.get("wavelength units", "")
        print(f"  Wavelength range: {min(wls):.2f} – {max(wls):.2f} {wl_units}")

    return out_hdr, out_data


# ══════════════════════════════════════════════════════════════════════════════
# Validation  (uses pre-computed stats — no image I/O)
# ══════════════════════════════════════════════════════════════════════════════
def validate_with_spectral_pca(mean, cov, n_valid, eigenvalues, eigenvectors,
                               n_components, target_variance):
    """
    Construct a GaussianStats from the chunked mean/cov and pass it to
    spectral.principal_components(stats) to cross-check the eigen-
    decomposition without loading the full image a second time.
    """
    print()
    print("─" * 52)
    print("  VALIDATION: spectral.principal_components()")
    print("─" * 52)

    try:
        stats = GaussianStats(mean=mean, cov=cov, nsamples=n_valid)
        pc    = spectral.principal_components(stats)
    except Exception as e:
        print(f"  WARNING: spectral.principal_components() failed: {e}")
        return None

    spy_ev   = np.real(pc.eigenvalues)
    spy_evec = np.real(pc.eigenvectors)
    spy_mean = np.real(pc.stats.mean)

    # ── Mean ─────────────────────────────────────────────────────────────
    mean_diff    = np.linalg.norm(mean - spy_mean)
    mean_reldiff = mean_diff / (np.linalg.norm(spy_mean) + 1e-30)
    print(f"  Mean spectrum L2 diff       : {mean_diff:.6g}  "
          f"(relative: {mean_reldiff:.4e})")

    # ── Eigenvalues ──────────────────────────────────────────────────────
    n_cmp = min(len(eigenvalues), len(spy_ev))
    ev_c  = eigenvalues[:n_cmp]
    ev_s  = spy_ev[:n_cmp]
    denom = np.maximum(np.abs(ev_s), 1e-30)
    ev_rd = np.abs(ev_c - ev_s) / denom

    print(f"\n  Eigenvalue comparison (top {min(n_cmp, 10)}):")
    print(f"  {'PC':>4s}   {'Custom':>14s}   {'Spectral':>14s}   {'RelDiff':>10s}")
    print(f"  {'───':>4s}   {'──────':>14s}   {'────────':>14s}   {'───────':>10s}")
    for i in range(min(n_cmp, 10)):
        print(f"  {i+1:4d}   {ev_c[i]:14.6g}   {ev_s[i]:14.6g}   "
              f"{ev_rd[i]:10.4e}")

    max_ev_rd = np.max(ev_rd[:min(n_cmp, n_components)])
    print(f"\n  Max eigenvalue rel. diff (top {n_components} PCs): "
          f"{max_ev_rd:.4e}")

    # ── Eigenvectors ─────────────────────────────────────────────────────
    n_dot = min(n_cmp, n_components)
    dots  = np.array([
        np.abs(np.dot(eigenvectors[:, i], spy_evec[:, i]))
        for i in range(n_dot)
    ])
    print(f"\n  Eigenvector alignment |dot product| (top {n_dot}):")
    print(f"  {'PC':>4s}   {'|v_custom · v_spy|':>20s}   {'Angle (deg)':>12s}")
    print(f"  {'───':>4s}   {'──────────────────':>20s}   {'───────────':>12s}")
    for i in range(min(n_dot, 10)):
        ang = np.degrees(np.arccos(np.clip(dots[i], 0.0, 1.0)))
        print(f"  {i+1:4d}   {dots[i]:20.10f}   {ang:12.6f}")

    min_dot = np.min(dots)
    print(f"\n  Min eigenvector |dot| across retained PCs: {min_dot:.10f}")

    # ── Component count ──────────────────────────────────────────────────
    pc_red   = pc.reduce(fraction=target_variance)
    spy_n    = len(pc_red.eigenvalues)
    print(f"\n  Components for {target_variance:.2%} variance:")
    print(f"    Custom   : {n_components}")
    print(f"    Spectral : {spy_n}")
    print(f"    Result   : {'MATCH' if n_components == spy_n else 'DIFFER'}")

    # ── Verdict ──────────────────────────────────────────────────────────
    ev_ok  = max_ev_rd < 0.01
    vec_ok = min_dot > 0.999
    if ev_ok and vec_ok:
        print("\n  ✓ VALIDATION PASSED — custom PCA matches spectral package")
    else:
        print("\n  ⚠ VALIDATION WARNING — differences detected")
        if not ev_ok:
            print(f"    Eigenvalue max rel. diff {max_ev_rd:.4e} > 1%")
        if not vec_ok:
            print(f"    Eigenvector min |dot| {min_dot:.10f} < 0.999")

    print("─" * 52)
    return {
        "mean_reldiff":    mean_reldiff,
        "max_ev_reldiff":  max_ev_rd,
        "min_dot_product": min_dot,
        "spy_n_components": spy_n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Band contribution analysis  (operates on small eigenvector matrix only)
# ══════════════════════════════════════════════════════════════════════════════
def compute_band_contributions(eigenvalues, eigenvectors, n_components,
                               bands, metadata, top_n=20, out_stem=None):
    """
    Compute importance_j = Σ_k  λ_k · |v_{jk}|²   for k = 1..n_components
    and report the top contributing input bands.
    """
    W   = eigenvectors[:, :n_components]
    lam = eigenvalues[:n_components]

    importance      = (W ** 2) @ lam
    importance_norm = importance / importance.sum()

    rank_idx    = np.argsort(importance)[::-1]
    top_n       = min(top_n, bands)
    top_indices = rank_idx[:top_n]

    wavelengths = None
    wl_units    = metadata.get("wavelength units", "")
    if "wavelength" in metadata:
        try:
            wavelengths = np.array([float(w) for w in metadata["wavelength"]])
        except (ValueError, TypeError):
            pass

    band_names_hdr = metadata.get("band names", None)
    has_wl = wavelengths is not None and len(wavelengths) == bands

    # ── Console report ───────────────────────────────────────────────────
    print()
    print("─" * 68)
    print(f"  TOP {top_n} CONTRIBUTING BANDS  "
          f"(variance-weighted squared loadings)")
    print("─" * 68)

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

    # ── CSV report ───────────────────────────────────────────────────────
    if out_stem is not None:
        csv_path = out_stem + "_top_bands.csv"
        with open(csv_path, "w") as f:
            cols_csv = ["rank", "band_index", "band_number"]
            if has_wl:
                cols_csv += ["wavelength", "wavelength_units"]
            if band_names_hdr is not None and len(band_names_hdr) == bands:
                cols_csv += ["band_name"]
            cols_csv += ["importance", "importance_fraction",
                         "cumulative_fraction"]
            for k in range(min(n_components, 10)):
                cols_csv.append(f"loading_PC{k+1}")
            f.write(",".join(cols_csv) + "\n")

            cum = 0.0
            for rank, bi in enumerate(rank_idx):
                cum += importance_norm[bi]
                row = [str(rank + 1), str(bi), str(bi + 1)]
                if has_wl:
                    row += [f"{wavelengths[bi]:.4f}", wl_units]
                if band_names_hdr is not None and len(band_names_hdr) == bands:
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
        print(f"  (all {bands} bands ranked, "
              f"top {min(n_components, 10)} PC loadings)")
    # Find visible band indices closest to (Blue -450, Green -550, Red -650)  and add to the top_indices if not already present
    if has_wl:
        target_wls = [450, 550, 650]
        for target in target_wls:
            idx_closest = np.argmin(np.abs(wavelengths - target))
            if idx_closest not in top_indices:
                top_indices = np.append(top_indices, idx_closest)
                print(f"  Added band {idx_closest} (wavelength {wavelengths[idx_closest]:.2f} {wl_units}) to top indices for being closest to {target} nm.")

    return top_indices, importance


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cli = parse_args()

    if cli.example:
        print(EXAMPLE_YAML)
        sys.exit(0)

    config_path = 'converters/v1/pca_config.yaml'
    #cfg = load_config(cli.config)
    cfg = load_config(config_path)
    
    input_hdr       = cfg["input_hdr"]
    target_variance = cfg["variance"]
    nodata_val      = cfg["nodata_value"]
    out_dtype_str   = cfg["output_dtype"]
    out_interleave  = cfg["output_interleave"]
    max_print       = cfg["max_print_components"]
    do_validate     = cfg["validate"]
    top_bands_n     = cfg["top_bands"]
    write_top       = cfg["write_top_bands"]
    chunk_lines     = cfg["chunk_lines"]
    out_dtype       = np.dtype(out_dtype_str)

    print(f"Config loaded from: {config_path}")
    print(f"  variance={target_variance}  nodata={nodata_val}  "
          f"dtype={out_dtype_str}  interleave={out_interleave}  "
          f"chunk_lines={chunk_lines}")
    print()

    # ── 1. Open image (no data loaded) ───────────────────────────────────
    print(f"[1/6] Opening {input_hdr} ...")
    img  = envi.open(input_hdr)
    meta = img.metadata.copy()
    rows, cols, bands = img.nrows, img.ncols, img.nbands
    n_pixels = rows * cols

    src_dtype = np.float32
    try:
        if hasattr(img, 'dtype'):
            src_dtype = np.dtype(img.dtype)
    except TypeError:
        pass

    file_gb  = n_pixels * bands * src_dtype.itemsize / 2**30
    chunk_gb = chunk_lines * cols * bands * 8 / 2**30    # float64 working

    print(f"       Image shape  : {rows} lines × {cols} samples × {bands} bands")
    print(f"       Interleave   : {meta.get('interleave', '?')}")
    print(f"       Src dtype    : {src_dtype}")
    print(f"       File size    : {file_gb:.2f} GB")
    print(f"       Chunk budget : {chunk_lines} lines  ≈ {chunk_gb:.2f} GB "
          f"(float64 working copy)")
    n_chunks = (rows + chunk_lines - 1) // chunk_lines
    print(f"       Passes       : 2 × {n_chunks} chunks  (mean + cov), "
          f"then 1 × {n_chunks} (projection)")

    # ── 2. Chunked statistics ────────────────────────────────────────────
    print(f"\n[2/6] Computing mean and covariance "
          f"(two-pass, {chunk_lines}-line chunks) ...")
    t_all = time.time()
    mean, cov, n_valid = compute_stats_chunked(img, nodata_val, chunk_lines)
    print(f"       Valid pixels : {n_valid} / {n_pixels}  "
          f"({100 * n_valid / n_pixels:.1f} %)")
    print(f"       Stats time   : {time.time() - t_all:.1f}s")

    # ── 3. Eigen-decomposition (tiny — bands × bands matrix) ────────────
    print(f"\n[3/6] Eigen-decomposition of {bands}×{bands} covariance ...")

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues  = np.maximum(eigenvalues, 0.0)

    total_var = eigenvalues.sum()
    cumvar    = np.cumsum(eigenvalues) / total_var

    n_components = int(np.searchsorted(cumvar, target_variance) + 1)
    n_components = min(n_components, bands)
    explained    = cumvar[n_components - 1]

    print(f"       Total variance      : {total_var:.6g}")
    print(f"       Target fraction     : {target_variance:.2%}")
    print(f"       Components retained : {n_components}  "
          f"(explains {explained:.4%})")

    print()
    print("       PC#   Eigenvalue     Var%     CumVar%")
    print("       ───   ──────────     ────     ───────")
    for i in range(min(n_components, max_print)):
        pct  = eigenvalues[i] / total_var * 100
        cpct = cumvar[i] * 100
        print(f"       {i+1:3d}   {eigenvalues[i]:12.4g}   {pct:6.2f}   "
              f"{cpct:7.3f}")
    if n_components > max_print:
        print(f"       ... ({n_components - max_print} more components)")
    print()

    # ── Validate (uses pre-computed stats — zero extra I/O) ──────────────
    if do_validate:
        validate_with_spectral_pca(
            mean, cov, n_valid, eigenvalues, eigenvectors,
            n_components, target_variance,
        )

    # ── Band contributions (no I/O — just eigenvector algebra) ───────────
    input_stem = os.path.splitext(input_hdr)[0]
    out_stem   = (cfg["output"] if cfg["output"] is not None
                  else f"{input_stem}_pca")

    top_indices, importance = compute_band_contributions(
        eigenvalues, eigenvectors, n_components,
        bands, meta, top_n=top_bands_n, out_stem=out_stem,
    )

    # ── 4. Chunked projection → write PCA image ─────────────────────────
    W = eigenvectors[:, :n_components]

    out_ext  = f".{out_interleave}"
    out_hdr  = out_stem + ".hdr"
    out_data = out_stem + out_ext

    out_meta = {}
    out_meta["description"]  = (
        f"PCA of {os.path.basename(input_hdr)}: "
        f"{n_components} components, {explained:.4%} variance"
    )
    out_meta["samples"]       = str(cols)
    out_meta["lines"]         = str(rows)
    out_meta["bands"]         = str(n_components)
    out_meta["header offset"] = "0"
    out_meta["data type"]     = str(envi_typecode(out_dtype))
    out_meta["interleave"]    = out_interleave
    out_meta["byte order"]    = "0"

    for key in ["map info", "coordinate system string", "projection info",
                "x start", "y start", "default stretch"]:
        if key in meta:
            out_meta[key] = meta[key]

    out_meta["band names"] = [
        f"PC{i+1} ({eigenvalues[i]/total_var*100:.2f}%)"
        for i in range(n_components)
    ]
    out_meta["pca eigenvalues"]     = [f"{v:.8g}" for v in eigenvalues[:n_components]]
    out_meta["pca mean spectrum"]   = [f"{v:.8g}" for v in mean]
    out_meta["pca target variance"] = str(target_variance)
    out_meta["pca config file"]     = str(Path(cli.config).resolve())

    out_img = envi.create_image(
        out_hdr, metadata=out_meta, dtype=out_dtype,
        force=True, ext=out_ext,
    )

    print(f"\n[4/6] Projecting onto {n_components} principal components "
          f"(chunked) ...")
    project_and_write_chunked(
        img, mean, W, nodata_val, out_img, out_dtype, chunk_lines,
    )

    print()
    print("═" * 60)
    print(f"  [5/6] PCA output written")
    print(f"  Output data : {out_data}")
    print(f"  Output HDR  : {out_hdr}")
    print(f"  Shape       : {rows} × {cols} × {n_components}  "
          f"({out_dtype_str} {out_interleave.upper()})")
    print(f"  Variance    : {explained:.4%}  in {n_components} components")
    print("═" * 60)

    # ── 6. Top contributing bands (chunked) ──────────────────────────────
    if write_top:
        sorted_top = sorted(int(i) for i in top_indices)
        print(f"\n[6/6] Writing top {len(sorted_top)} contributing bands "
              f"(chunked) ...")
        write_top_bands_chunked(
            img, sorted_top, meta, out_stem,
            out_interleave, chunk_lines,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_all
    print()
    print("═" * 60)
    print(f"  All outputs written.  Total time: {elapsed:.1f}s")
    print("═" * 60)


if __name__ == "__main__":
    main()
