#!/usr/bin/env python3
"""
lut_to_steviapp.py — Convert LUT + smile model to Steviapp polynomial coefficients.

Workflow:
  1. Load lab LUT angles and smile polynomial parameters
  2. Convert across-track angles → pixel-space distortion Δx(u)
  3. Convert along-track smile angles → pixel-space distortion Δy(u)
  4. Fit degree-5 polynomials in ξ = (u − w/2) / w
  5. Convert fitted Δx, Δy back to angles
  6. Compare with originals (plot + error statistics)

Mathematical derivation:

  Steviapp forward model:
    u = f · (Xc/Zc) + ppx − Δx(u)          across-track
    v = f · (Yc/Zc)        − Δy(u) = 0      along-track constraint

  For a pixel u with lab-measured angle θ_ac(u):
    Xc/Zc = tan(θ_ac)   →   u = f·tan(θ_ac(u)) + ppx − Δx(u)
    Rearranging:   Δx(u) = f·tan(θ_ac(u)) + ppx − u

  For smile at pixel u with θ_at(θ_ac_deg):
    Yc/Zc = −tan(θ_at)  →  Δy(u) = f · Yc/Zc = −f·tan(θ_at(u))
    (negated: smile calibration Y-forward, camera Y-back)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from  camera_models.load_PSF import compute_focal_length

# =========================================================================
# Configuration — edit these to match your setup
# =========================================================================

ANGLE_LUT_PATH = '/media/addLidar/AVIRIS_4_Mission_Processing/AV4_Camera_Model_Data/AV4_acrosstrack_PSF_2024_valid_angles.csv'

# Smile polynomial: θ_at = a·θ_ac² + b·θ_ac + c  (all in degrees)
SMILE_A = -0.0036022
SMILE_B =  0.0003831
SMILE_C =  0.6025328

POLY_DEGREE = 5     # degree of Steviapp distortion polynomials


def load_lut(path):
    """Load the lab-measured per-pixel across-track angles [degrees]."""
    angles = np.loadtxt(path, dtype=np.float64)
    print(f"LUT: {len(angles)} pixels, "
          f"θ ∈ [{angles[0]:.4f}°, {angles[-1]:.4f}°]")
    return angles


def estimate_pinhole(angles_deg, w):
    """
    Estimate f and ppx that best fit the LUT in a least-squares sense.

    The ideal pinhole maps:  u = f · tan(θ) + ppx

    We solve for [f, ppx] via linear least squares on:
        u_i = f · tan(θ_i) + ppx
    """
    u = np.arange(w, dtype=np.float64)
    tan_theta = np.tan(np.deg2rad(angles_deg))

    # Design matrix: [tan(θ), 1]
    A = np.column_stack([tan_theta, np.ones(w)])
    x, residuals, _, _ = np.linalg.lstsq(A, u, rcond=None)
    f_fit, ppx_fit = x[0], x[1]

    print(f"Estimated pinhole: f = {f_fit:.4f} px, ppx = {ppx_fit:.4f} px")
    print(f"  LS residual RMS = {np.sqrt(np.mean((A @ x - u)**2)):.4f} px")
    return f_fit, ppx_fit


def angles_to_distortion(angles_deg, f, ppx):
    """
    Convert LUT angles + smile to pixel-space distortion Δx, Δy.

    Δx(u) = f · tan(θ_ac(u)) + ppx − u
    Δy(u) = −f · tan(θ_at(u))

    where θ_at(u) = smile(θ_ac_deg(u)).
    """
    w = len(angles_deg)
    u = np.arange(w, dtype=np.float64)
    tan_ac = np.tan(np.deg2rad(angles_deg))

    # Across-track distortion
    delta_x = f * tan_ac + ppx - u

    # Along-track: smile evaluated at each pixel's across-track angle
    smile_deg = SMILE_A * angles_deg**2 + SMILE_B * angles_deg + SMILE_C
    # Negate: smile calibration is Y-forward, camera is Y-back
    delta_y = f * np.tan(np.deg2rad(smile_deg))

    print(f"\nPixel-space distortion:")
    print(f"  Δx: [{delta_x.min():+.4f}, {delta_x.max():+.4f}] px, "
          f"ptp = {np.ptp(delta_x):.4f} px")
    print(f"  Δy: [{delta_y.min():+.4f}, {delta_y.max():+.4f}] px, "
          f"ptp = {np.ptp(delta_y):.4f} px")

    return delta_x, delta_y


def fit_polynomials(delta_x, delta_y, w, degree=5):
    """
    Fit degree-N polynomials to Δx(u) and Δy(u).

    ξ(u) = (u − w/2) / w
    Δ(u) ≈ Σ_{i=0}^{N} c_i · ξ^i
    """
    u = np.arange(w, dtype=np.float64)
    xi = (u - w / 2.0) / w

    # Vandermonde matrix: [ξ⁰, ξ¹, ..., ξⁿ]
    V = np.column_stack([xi**i for i in range(degree + 1)])

    a_coeffs, _, _, _ = np.linalg.lstsq(V, delta_x, rcond=None)
    b_coeffs, _, _, _ = np.linalg.lstsq(V, delta_y, rcond=None)

    # Evaluate fits
    dx_fit = V @ a_coeffs
    dy_fit = V @ b_coeffs

    # Residual statistics
    dx_resid = delta_x - dx_fit
    dy_resid = delta_y - dy_fit

    print(f"\nPolynomial fit (degree {degree}):")
    print(f"  a_coeffs (Δx): {np.array2string(a_coeffs, precision=10, separator=', ')}")
    print(f"  b_coeffs (Δy): {np.array2string(b_coeffs, precision=10, separator=', ')}")
    print(f"\n  Δx fit residual: RMS = {rms(dx_resid):.6f} px, "
          f"max = {np.max(np.abs(dx_resid)):.6f} px")
    print(f"  Δy fit residual: RMS = {rms(dy_resid):.6f} px, "
          f"max = {np.max(np.abs(dy_resid)):.6f} px")

    return a_coeffs, b_coeffs, dx_fit, dy_fit


def distortion_to_angles(u, dx_fit, dy_fit, f, ppx):
    """
    Convert polynomial distortion back to angles for comparison.

    θ_ac_recovered = arctan( (u + Δx(u) − ppx) / f )
    θ_at_recovered = −arctan( Δy(u) / f )       (negate back to Y-forward)
    """
    tan_ac = (u + dx_fit - ppx) / f
    theta_ac_recovered = np.rad2deg(np.arctan(tan_ac))

    # Negate Δy back: Δy = −f·tan(θ_at) → θ_at = −arctan(Δy/f)
    theta_at_recovered = -np.rad2deg(np.arctan(dy_fit / f))

    return theta_ac_recovered, theta_at_recovered


def rms(x):
    return np.sqrt(np.mean(x**2))


def plot_validation(angles_deg, smile_deg_lut, theta_ac_rec, theta_at_rec,
                    delta_x, delta_y, dx_fit, dy_fit, a_coeffs, b_coeffs,
                    f, ppx, w, output_path):
    """
    6-panel validation figure.

    Top row:    Across-track (LUT angles)
    Bottom row: Along-track  (smile)

    Left:   Δ in pixel space (data + polynomial fit)
    Centre: Recovered angles vs original
    Right:  Error (residual in arcseconds)
    """
    u = np.arange(w, dtype=np.float64)
    xi = (u - w/2.0) / w

    # Error in arcseconds
    err_ac_arcsec = (theta_ac_rec - angles_deg) * 3600.0
    err_at_arcsec = (theta_at_rec - smile_deg_lut) * 3600.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'LUT → Steviapp polynomial validation   '
        f'(f={f:.2f}, ppx={ppx:.2f}, w={w}, degree {len(a_coeffs)-1})',
        fontsize=13, fontweight='bold'
    )

    # ---- Top row: Across-track ----
    ax = axes[0, 0]
    ax.plot(u, delta_x, 'k-', lw=0.8, label='Δx from LUT')
    ax.plot(u, dx_fit, 'r--', lw=1.2, label='Polynomial fit')
    ax.set_xlabel('Pixel u')
    ax.set_ylabel('Δx [pixels]')
    ax.set_title('Across-track distortion Δx(u)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(u, angles_deg, 'k-', lw=0.8, label='LUT θ_ac')
    ax.plot(u, theta_ac_rec, 'r--', lw=1.2, label='Recovered from Δx')
    ax.set_xlabel('Pixel u')
    ax.set_ylabel('θ_ac [°]')
    ax.set_title('Across-track angle: LUT vs recovered')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(u, err_ac_arcsec, 'b-', lw=0.8)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Pixel u')
    ax.set_ylabel('Error [arcsec]')
    ax.set_title(f'Across-track error: '
                 f'RMS={rms(err_ac_arcsec):.4f}″, '
                 f'max={np.max(np.abs(err_ac_arcsec)):.4f}″')
    ax.grid(True, alpha=0.3)

    # ---- Bottom row: Along-track ----
    ax = axes[1, 0]
    ax.plot(u, delta_y, 'k-', lw=0.8, label='Δy from smile')
    ax.plot(u, dy_fit, 'r--', lw=1.2, label='Polynomial fit')
    ax.set_xlabel('Pixel u')
    ax.set_ylabel('Δy [pixels]')
    ax.set_title('Along-track distortion Δy(u) (smile)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(u, smile_deg_lut, 'k-', lw=0.8, label='Smile model')
    ax.plot(u, theta_at_rec, 'r--', lw=1.2, label='Recovered from Δy')
    ax.set_xlabel('Pixel u')
    ax.set_ylabel('θ_at [°]')
    ax.set_title('Along-track angle: smile vs recovered')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(u, err_at_arcsec, 'b-', lw=0.8)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Pixel u')
    ax.set_ylabel('Error [arcsec]')
    ax.set_title(f'Along-track error: '
                 f'RMS={rms(err_at_arcsec):.4f}″, '
                 f'max={np.max(np.abs(err_at_arcsec)):.4f}″')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()


def plot_coefficient_bar(a_coeffs, b_coeffs, output_path):
    """Bar chart of fitted polynomial coefficients."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    degree = len(a_coeffs) - 1
    idx = np.arange(degree + 1)

    ax1.bar(idx, a_coeffs, color='steelblue', edgecolor='navy', alpha=0.8)
    ax1.set_xlabel('Coefficient index i')
    ax1.set_ylabel('aᵢ [pixels]')
    ax1.set_title('Δx coefficients (across-track)')
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f'a{i}' for i in idx])
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(a_coeffs):
        ax1.text(i, v, f'{v:.4f}', ha='center',
                 va='bottom' if v >= 0 else 'top', fontsize=8)

    ax2.bar(idx, b_coeffs, color='coral', edgecolor='darkred', alpha=0.8)
    ax2.set_xlabel('Coefficient index i')
    ax2.set_ylabel('bᵢ [pixels]')
    ax2.set_title('Δy coefficients (along-track / smile)')
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f'b{i}' for i in idx])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(b_coeffs):
        ax2.text(i, v, f'{v:.4f}', ha='center',
                 va='bottom' if v >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    print(f"Coefficient plot saved to {output_path}")
    plt.close()


def ground_error_table(err_ac_arcsec, err_at_arcsec, agls=[500, 1000, 2000]):
    """Print ground displacement errors at various AGL heights."""
    print(f"\n{'='*70}")
    print(f" Ground displacement error at various AGL (from polynomial fit error)")
    print(f"{'='*70}")
    print(f"{'AGL [m]':>10}  {'AC RMS [m]':>12}  {'AC max [m]':>12}  "
          f"{'AT RMS [m]':>12}  {'AT max [m]':>12}")
    print(f"{'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for agl in agls:
        ac_rms_m = agl * np.tan(np.deg2rad(rms(err_ac_arcsec) / 3600))
        ac_max_m = agl * np.tan(np.deg2rad(np.max(np.abs(err_ac_arcsec)) / 3600))
        at_rms_m = agl * np.tan(np.deg2rad(rms(err_at_arcsec) / 3600))
        at_max_m = agl * np.tan(np.deg2rad(np.max(np.abs(err_at_arcsec)) / 3600))
        print(f"{agl:>10}  {ac_rms_m:>12.4f}  {ac_max_m:>12.4f}  "
              f"{at_rms_m:>12.4f}  {at_max_m:>12.4f}")


def roundtrip_test(angles_deg, f, ppx, a_coeffs, b_coeffs, w):
    """
    Full roundtrip: pixel → ray → project → pixel.

    Verifies the Steviapp camera model (with fitted coefficients)
    gives a self-consistent forward-inverse pair.
    """
    from camera_model import PushbroomCamera, DistortionPoly

    cam = PushbroomCamera(
        f=f, ppx=ppx, w=w,
        delta_x_coeffs=a_coeffs.tolist(),
        delta_y_coeffs=b_coeffs.tolist(),
        angle_lut_path=ANGLE_LUT_PATH,
    )

    u_test = np.linspace(0, w-1, 500)
    rays = cam.pixel_to_ray(u_test)
    # Scale to some depth
    d_cam = rays * 500.0
    d_cam[:, 2] = 500.0
    u_recovered = cam.project(d_cam)
    resid_at = cam.along_track_residual(d_cam, u_recovered)

    err_px = np.abs(u_recovered - u_test)
    print(f"\nRoundtrip test (pixel → ray → project):")
    print(f"  Across-track: max pixel error = {err_px.max():.2e} px")
    print(f"  Along-track:  max residual    = {np.abs(resid_at).max():.2e}")


def main():
    print("="*70)
    print(" LUT + Smile → Steviapp Polynomial Coefficient Derivation")
    print("="*70)

    # ---- Step 1: Load LUT ----
    angles_deg = load_lut(ANGLE_LUT_PATH)
    # change order of angles_deg
    #angles_deg = angles_deg[::-1]
    fov = angles_deg[-1] - angles_deg[0]
    w = len(angles_deg)
    u = np.arange(w, dtype=np.float64)

    # ---- Step 2: Estimate optimal pinhole parameters ----
    f, ppx = estimate_pinhole(angles_deg, w)
    #f *= -1
    #f = compute_focal_length(fov, w)

    # ---- Step 3: Convert angles → pixel-space distortion ----
    delta_x, delta_y = angles_to_distortion(angles_deg, f, ppx)

    # ---- Step 4: Fit degree-5 polynomials ----
    a_coeffs, b_coeffs, dx_fit, dy_fit = fit_polynomials(
        delta_x, delta_y, w, degree=POLY_DEGREE)

    # ---- Step 5: Convert back to angles ----
    theta_ac_rec, theta_at_rec = distortion_to_angles(
        u, dx_fit, dy_fit, f, ppx)

    # Original smile at each pixel's LUT angle
    smile_deg_lut = SMILE_A * angles_deg**2 + SMILE_B * angles_deg + SMILE_C

    # ---- Step 6: Error statistics ----
    err_ac_arcsec = (theta_ac_rec - angles_deg) * 3600.0
    err_at_arcsec = (theta_at_rec - smile_deg_lut) * 3600.0

    print(f"\n{'='*70}")
    print(f" Angular error: polynomial ↔ original")
    print(f"{'='*70}")
    print(f"  Across-track:  RMS = {rms(err_ac_arcsec):.4f}″, "
          f"max = {np.max(np.abs(err_ac_arcsec)):.4f}″")
    print(f"  Along-track:   RMS = {rms(err_at_arcsec):.4f}″, "
          f"max = {np.max(np.abs(err_at_arcsec)):.4f}″")

    # At sub-pixel: how many arcseconds is 1 pixel?
    mean_ifov_arcsec = np.mean(np.diff(angles_deg)) * 3600
    print(f"\n  Reference: 1 pixel IFOV = {mean_ifov_arcsec:.1f}″")
    print(f"  AC fit error = {rms(err_ac_arcsec)/mean_ifov_arcsec:.4f} × IFOV (RMS)")
    print(f"  AT fit error = {rms(err_at_arcsec)/mean_ifov_arcsec:.4f} × IFOV (RMS)")

    ground_error_table(err_ac_arcsec, err_at_arcsec)

    # ---- Step 7: Plots ----
    plot_validation(
        angles_deg, smile_deg_lut, theta_ac_rec, theta_at_rec,
        delta_x, delta_y, dx_fit, dy_fit, a_coeffs, b_coeffs,
        f, ppx, w, output_path='validation_angles.png')

    plot_coefficient_bar(a_coeffs, b_coeffs,
                         output_path='validation_coefficients.png')

    # ---- Step 8: Roundtrip self-consistency ----
    import sys
    sys.path.insert(0, 'pipeline')
    roundtrip_test(angles_deg, f, ppx, a_coeffs, b_coeffs, w)

    # ---- Step 9: Print config snippet ----
    print(f"\n{'='*70}")
    print(f" config.yaml snippet")
    print(f"{'='*70}")
    print(f"camera:")
    print(f"  model: steviapp")
    print(f"  steviapp:")
    print(f"    f:     {f:.4f}")
    print(f"    ppx:   {ppx:.4f}")
    print(f"    w:     {w}")
    print(f"    # Δx across-track distortion [a0, a1, ..., a{POLY_DEGREE}]")
    a_str = ', '.join(f'{c:.10e}' for c in a_coeffs)
    print(f"    delta_x_coeffs: [{a_str}]")
    print(f"    # Δy along-track smile [b0, b1, ..., b{POLY_DEGREE}]")
    b_str = ', '.join(f'{c:.10e}' for c in b_coeffs)
    print(f"    delta_y_coeffs: [{b_str}]")
    print(f"  first_valid_pixel: 0")

    print(f"\n{'='*70}")
    print(f" Done. Check validation_angles.png and validation_coefficients.png")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
