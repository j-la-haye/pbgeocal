"""
camera_model.py — Decoupled pushbroom camera with in-flight correction.

=============================================================================
 Physical Model
=============================================================================

The lab calibration measures the across-track viewing angle θ_lab(i) for each
pixel i at reference conditions (temperature T₀, pressure P₀).  On the
detector focal plane, pixel i sits at physical position:

    u_det(i) = f₀ · tan(θ_lab(i)) + cx₀

where f₀ is the lab focal length and cx₀ the lab principal point.

In flight at (T, P), thermal expansion and pressure changes shift the optics:

    u_det(i) = f · tan(θ_true(i)) + cx

Same pixel, different angle.  Equating:

    f₀ · tan(θ_lab(i)) + cx₀ = f · tan(θ_true(i)) + cx

Solving for the lab-equivalent tangent (what we look up in the LUT):

    tan(θ_lab(i)) = (f/f₀) · tan(θ_true(i)) + (cx − cx₀)/f₀

Define:
    s   = f / f₀           focal length ratio       (≈ 1, e.g. 1.0001)
    Δcx = (cx − cx₀) / f₀  normalised PP shift, XT  (tangent-space units)
    Δcy = (cy − cy₀) / f₀  normalised PP shift, AT  (tangent-space units)

Then the **tangent-space affine correction** is:

    x_lab = s · x_true + Δcx        (across-track)
    y_lab = s · y_true + Δcy        (along-track)

where:
    x_true = Xc / Zc = tan(θ_xt_true)    observed tangent from 3D point
    y_true = Yc / Zc = tan(θ_at_true)    observed tangent from 3D point
    x_lab  = tan(θ_xt_lab)               what we interpolate in the LUT

Properties:
  • s = 1, Δcx = Δcy = 0 recovers the pure lab calibration exactly
  • The correction is linear in tangent space (physically exact, not an approximation)
  • All per-pixel irregularities in the LUT are preserved
  • Only 3 free parameters (s, Δcx, Δcy) to estimate from tie points
  • Typical magnitudes:  s − 1 ~ 50–200 ppm,  Δcx/Δcy ~ 1e-5 to 1e-4

=============================================================================
 Decoupled Calibration
=============================================================================

1. Across-Track (X):
   x_lab = s · (Xc/Zc) + Δcx
   pixel = interp(arctan(x_lab), θ_LUT, pixel_indices)

2. Along-Track (Y):
   y_lab = s · (Yc/Zc) + Δcy
   residual = y_lab − tan(smile(x_lab))  →  0

   where smile(xt) = deg2rad(a·xt² + b·xt + c)

=============================================================================
 Frame Convention
=============================================================================
Camera frame: X-right (across-track), Y-back (along-track), Z-down (nadir).
Angles in degrees for LUT storage, radians for all internal math.
Pixel indices: 0-based, integer at pixel centres.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass


# =========================================================================
# Smile model
# =========================================================================

@dataclass
class SmileModel:
    """
    Along-track smile polynomial in degrees:
        θ_at_deg = a · xt² + b · xt + c
    where xt = tan(θ_across_track) in the lab-corrected tangent space.
    """
    a: float    # quadratic (smile curvature)
    b: float    # linear (smile tilt)
    c: float    # constant (boresight along-track offset, degrees)

    def evaluate_deg(self, xt: np.ndarray) -> np.ndarray:
        """Along-track smile angle [degrees] given xt = tan(θ_xt)."""
        return self.a * xt**2 + self.b * xt + self.c

    def evaluate_rad(self, xt: np.ndarray) -> np.ndarray:
        """Along-track smile angle [radians]."""
        return np.deg2rad(self.evaluate_deg(xt))

    def evaluate_tan(self, xt: np.ndarray) -> np.ndarray:
        """Along-track smile as tangent value: tan(smile(xt))."""
        return np.tan(self.evaluate_rad(xt))


# =========================================================================
# In-flight optics correction
# =========================================================================

@dataclass
class OpticsCorrection:
    """
    In-flight focal length and principal point correction.

    User-facing parameters (physically intuitive):
        f_lab : lab focal length [pixels]  (from LUT mean IFOV)
        f     : in-flight focal length [pixels]
        cx    : across-track PP shift [pixels] FROM LAB OPTICAL AXIS
        cy    : along-track  PP shift [pixels] FROM LAB OPTICAL AXIS

    cx/cy convention:
        The lab angle LUT already encodes the optical axis position (the
        zero-angle crossing, e.g. pixel 627.78).  cx and cy are DELTAS
        from that lab-calibrated position — NOT absolute pixel coordinates
        and NOT relative to the detector geometric center.

        cx = 0  → optical axis is where the lab measured it
        cx = +1 → optical axis shifted 1 pixel rightward since lab cal
        cy = +1 → optical axis shifted 1 pixel in +Y camera direction

    Internally converted to the tangent-space affine:
        s   = f / f_lab                      focal ratio
        Δcx = cx * pixel_pitch_tan           XT PP shift in tangent units
        Δcy = cy * pixel_pitch_tan           AT PP shift in tangent units

    The focal scaling pivots around the lab optical axis (zero-angle pixel),
    which is physically correct: thermal expansion scales angles from the
    optical axis outward.
    """
    f_lab: float = 1762.2    # lab focal length [pixels]
    f: float = 1762.2        # in-flight focal length [pixels]
    cx: float = 0.0          # across-track PP shift [pixels] from lab
    cy: float = 0.0          # along-track  PP shift [pixels] from lab

    # Derived tangent-space parameters (set by bind())
    s: float = 1.0
    dcx: float = 0.0
    dcy: float = 0.0
    _bound: bool = False

    def bind(self, pixel_pitch_tan: float):
        """
        Compute tangent-space affine parameters from f/cx/cy.

        Called automatically by PushbroomCamera.__init__() once the
        pixel pitch is known from the angle LUT.

        Parameters
        ----------
        pixel_pitch_tan : mean tangent-space pixel pitch from LUT
        """
        self.s = self.f / self.f_lab
        self.dcx = self.cx * pixel_pitch_tan
        self.dcy = self.cy * pixel_pitch_tan
        self._bound = True


# =========================================================================
# Main camera model
# =========================================================================

class PushbroomCamera:
    """
    Decoupled pushbroom camera model with in-flight correction.

    Across-track: lab-measured angle LUT + tangent-space affine correction
    Along-track:  quadratic smile model + same affine correction

    Parameters
    ----------
    angle_lut_path : path to CSV with one across-track angle (degrees) per line
    smile          : SmileModel with polynomial coefficients
    correction     : OpticsCorrection for focal length / PP drift
                     (default: identity, i.e. pure lab calibration)
    """

    def __init__(self, angle_lut_path: str, smile: SmileModel,
                 correction: OpticsCorrection = None):
        path = Path(angle_lut_path)
        if not path.exists():
            raise FileNotFoundError(f"Angle LUT not found: {angle_lut_path}")

        # ---- Load lab-measured across-track angles [degrees] ----
        self.angles_deg = np.loadtxt(str(path), dtype=np.float64)
        self.num_pixels = len(self.angles_deg)
        self.pixel_indices = np.arange(self.num_pixels, dtype=np.float64)

        # Radians and tangents for internal use
        self.angles_rad = np.deg2rad(self.angles_deg)
        self.tan_angles = np.tan(self.angles_rad)    # x_lab values for each pixel

        # Verify monotonicity
        if not np.all(np.diff(self.angles_deg) > 0):
            raise ValueError(
                "Across-track angle LUT must be monotonically increasing"
            )

        # ---- Mean pixel pitch in tangent space (for unit conversions) ----
        self.pixel_pitch_tan = float(np.mean(np.diff(self.tan_angles)))

        # ---- Smile model ----
        self.smile = smile

        # Pre-compute smile tangent at each pixel's lab-LUT position
        self.smile_tan_per_pixel = self.smile.evaluate_tan(self.tan_angles)

        # ---- In-flight optics correction ----
        self.correction = correction or OpticsCorrection()
        self.correction.bind(self.pixel_pitch_tan)

        # ---- Diagnostics ----
        fov = self.angles_deg[-1] - self.angles_deg[0]
        mean_ifov = np.mean(np.diff(self.angles_deg))
        smile_amp_arcsec = (np.ptp(self.smile.evaluate_deg(self.tan_angles))
                            * 3600.0)

        # Report the zero-angle crossing (optical axis position from lab LUT)
        zero_pixel = float(np.interp(0.0, self.angles_rad, self.pixel_indices))
        geom_center = (self.num_pixels - 1) / 2.0

        c = self.correction
        print(f"  Camera: {self.num_pixels} px, FOV={fov:.2f}°, "
              f"IFOV={mean_ifov*1000:.1f} mrad, "
              f"smile={smile_amp_arcsec:.1f}\"")
        print(f"  Optical axis (lab): pixel {zero_pixel:.2f} "
              f"(detector center={geom_center:.1f}, "
              f"offset={zero_pixel - geom_center:+.2f} px)")
        if c.f != c.f_lab or c.cx != 0.0 or c.cy != 0.0:
            delta_f_ppm = (c.s - 1.0) * 1e6
            print(f"  Optics correction: f={c.f:.2f} px "
                  f"(Δf={delta_f_ppm:+.1f} ppm), "
                  f"cx={c.cx:+.3f} px, cy={c.cy:+.3f} px")

    # =================================================================
    # Update correction parameters (e.g. after tie-point optimisation)
    # =================================================================

    def set_correction(self, correction: OpticsCorrection):
        """
        Update the in-flight optics correction parameters.

        Automatically recomputes the tangent-space affine parameters.
        """
        correction.bind(self.pixel_pitch_tan)
        self.correction = correction

    # =================================================================
    # Forward projection:  3D camera point → (pixel, along-track tangent)
    # =================================================================

    def project(self, P_cam: np.ndarray):
        """
        Project 3D camera-frame point(s) to across-track pixel and
        along-track tangent.

        The projection chain:
          1. x_true = Xc/Zc,  y_true = Yc/Zc     (observed tangents)
          2. x_lab = s·x_true + Δcx                (affine correction)
          3. pixel = interp(arctan(x_lab), θ_LUT)  (LUT lookup)
          4. y_lab = s·y_true + Δcy                (affine correction)

        Parameters
        ----------
        P_cam : (..., 3) point(s) in camera frame [X-right, Y-back, Z-down]

        Returns
        -------
        u_pixel : across-track pixel index (fractional)
        y_lab   : along-track lab-corrected tangent [dimensionless]
                  The Newton solver drives  y_lab − smile_tan(x_lab) → 0
        """
        Xc = P_cam[..., 0]
        Yc = P_cam[..., 1]
        Zc = P_cam[..., 2]

        s = self.correction.s
        dcx = self.correction.dcx
        dcy = self.correction.dcy

        # Observed tangent-plane coordinates
        x_true = Xc / Zc
        y_true = Yc / Zc

        # Affine correction to lab-equivalent tangent space
        x_lab = s * x_true + dcx
        y_lab = s * y_true + dcy

        # Across-track pixel via angle LUT
        # arctan(x_lab) → angle → interpolate in LUT
        theta_lab = np.arctan(x_lab)
        u_pixel = np.interp(theta_lab, self.angles_rad, self.pixel_indices)

        return u_pixel, y_lab

    # =================================================================
    # Along-track residual for Newton solver
    # =================================================================

    def along_track_residual(self, P_cam: np.ndarray,
                             u_pixel: np.ndarray) -> np.ndarray:
        """
        Compute the along-track residual for the Newton iteration.

        After applying the in-flight correction, the corrected along-track
        tangent should equal the smile model prediction at the pixel position:

            residual = y_lab − smile_tan(x_lab)

        where y_lab = s·(Yc/Zc) + Δcy
        and   smile_tan is the tangent of the smile angle at x_lab.

        Parameters
        ----------
        P_cam   : (..., 3) points in camera frame
        u_pixel : (...,) across-track pixel indices (from project())

        Returns
        -------
        residual : (...,) along-track residual [tangent-space units]
        """
        Yc = P_cam[..., 1]
        Zc = P_cam[..., 2]

        s = self.correction.s
        dcy = self.correction.dcy

        y_lab = s * (Yc / Zc) + dcy

        # Smile prediction at this pixel (interpolate pre-computed LUT)
        smile_tan = np.interp(u_pixel, self.pixel_indices,
                              self.smile_tan_per_pixel)

        return y_lab - smile_tan

    # =================================================================
    # Convenience: get smile tangent at pixel positions
    # =================================================================

    def get_smile_tangent(self, u_pixel: np.ndarray) -> np.ndarray:
        """
        Smile tangent value at fractional pixel positions.
        """
        return np.interp(u_pixel, self.pixel_indices,
                         self.smile_tan_per_pixel)

    # =================================================================
    # Validity check
    # =================================================================

    def is_valid_pixel(self, u_pixel: np.ndarray) -> np.ndarray:
        """True if pixel is within sensor bounds [0, num_pixels)."""
        return (u_pixel >= 0) & (u_pixel < self.num_pixels)

    # =================================================================
    # Inverse:  pixel → look direction  (for ray casting / tie-point analysis)
    # =================================================================

    def pixel_to_look_direction(self, u_pixel: np.ndarray) -> np.ndarray:
        """
        Convert pixel indices to unit look-direction vectors in camera frame,
        accounting for in-flight correction.

        The inverse of the projection chain:
          1. θ_lab = interp(pixel, LUT)
          2. x_lab = tan(θ_lab);  y_lab = smile_tan(x_lab)
          3. x_true = (x_lab − Δcx) / s;  y_true = (y_lab − Δcy) / s
          4. look = [x_true, y_true, 1] / norm

        Parameters
        ----------
        u_pixel : (N,) across-track pixel indices (fractional)

        Returns
        -------
        look : (N, 3) unit vectors in camera frame [X, Y, Z]
        """
        s = self.correction.s
        dcx = self.correction.dcx
        dcy = self.correction.dcy

        # Lab tangent coordinates from LUT
        theta_lab = np.interp(u_pixel, self.pixel_indices, self.angles_rad)
        x_lab = np.tan(theta_lab)
        y_lab = self.smile.evaluate_tan(x_lab)

        # Invert the affine correction to get true tangents
        x_true = (x_lab - dcx) / s
        y_true = (y_lab - dcy) / s

        # Look direction: [x_true, y_true, 1] normalised
        look = np.column_stack([x_true, y_true, np.ones_like(x_true)])
        norms = np.linalg.norm(look, axis=1, keepdims=True)
        return look / norms
