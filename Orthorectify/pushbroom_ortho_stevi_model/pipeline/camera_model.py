"""
camera_model.py — Steviapp pushbroom camera with polynomial lens distortion.

=============================================================================
 Projection Model  (Steviapp, eq. 1)
=============================================================================

    ┌ u ┐   ┌ f  0  ppx ┐  1          ┌ Δx(u) ┐
    │ v │ = │ 0  f   0  │ ── Γ · P  − │ Δy(u) │        (1)
    └ 1 ┘   └ 0  0   1  ┘ Zc          └   0   ┘

where Γ encapsulates camera pose (R, t), and P is the world point.

Expanded in camera frame:

    u  =  f · (Xc / Zc) + ppx  −  Δx(u)        across-track pixel
    v  =  f · (Yc / Zc)        −  Δy(u)         along-track residual → 0

v = 0 is the pushbroom scanline constraint:  at the correct observation
time t*, the point projects exactly onto the linear detector.

=============================================================================
 Distortion Polynomials  (degree 5, normalised)
=============================================================================

    Δx(u) = Σ_{i=0}^{5}  aᵢ · ξⁱ         across-track distortion  [px]
    Δy(u) = Σ_{i=0}^{5}  bᵢ · ξⁱ         along-track (smile)      [px]

    ξ  =  (u − w/2) / w                    normalised pixel coordinate

where w is the detector width (number of valid pixels on the line).

The coefficients {aᵢ} and {bᵢ} are jointly optimised with focal length
and boresight in a factor-graph bundle adjustment using image tie points,
as described in:

    Barmettler, Burkhard, Monnerat et al. (2025).
    "Quality assessment of Airborne Image Spectrometry Data for AVIRIS-4."
    ResearchGate, DOI: 10.13140/RG.2.2.17556.17287

=============================================================================
 Inverse Projection  (for bottom-up orthorectification)
=============================================================================

Given camera-frame direction d = [Xc, Yc, Zc]:

  1.  u_ideal  =  f · (Xc / Zc) + ppx
  2.  Fixed-point iteration:  u ← u_ideal − Δx(u)   [converges in ≈ 5 iter]
  3.  Along-track residual:   r = Yc / Zc − Δy(u) / f  →  0

=============================================================================
 BIL Sample Mapping
=============================================================================

When the calibration covers only a subset of BIL detector columns:

    bil_sample  =  u  +  first_valid_pixel

=============================================================================
 Frame Convention
=============================================================================

Camera frame:  X-right (across-track), Y-back (along-track), Z-down (nadir).
Pixel index u :  0-based, 0 at left edge, w−1 at right edge.
ppx :  principal point in pixel units (= optical axis position ≈ 627.78).
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


# =========================================================================
# Distortion polynomial
# =========================================================================

@dataclass
class DistortionPoly:
    """
    Degree-5 polynomial distortion in normalised pixel coordinates.

        Δ(u)  =  Σ_{i=0}^{5}  cᵢ · ξⁱ        [pixels]
        ξ     =  (u − w/2) / w

    Parameters
    ----------
    coeffs : [c0, c1, c2, c3, c4, c5]  — polynomial coefficients
    w      : detector width for normalisation
    """
    coeffs: List[float]
    w: float

    def __post_init__(self):
        # Pad to exactly 6 coefficients (degree 5)
        c = list(self.coeffs)
        while len(c) < 6:
            c.append(0.0)
        self._c = np.array(c[:6], dtype=np.float64)
        self._half_w = self.w / 2.0

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Evaluate Δ(u) at pixel position(s) u.  Returns value in pixels."""
        xi = (u - self._half_w) / self.w
        # Horner's method for degree-5 polynomial
        c = self._c
        val = c[5]
        val = val * xi + c[4]
        val = val * xi + c[3]
        val = val * xi + c[2]
        val = val * xi + c[1]
        val = val * xi + c[0]
        return val

    def evaluate_derivative(self, u: np.ndarray) -> np.ndarray:
        """Evaluate dΔ/du at pixel position(s) u.  Returns [px/px]."""
        xi = (u - self._half_w) / self.w
        c = self._c
        # Derivative of Σ cᵢ ξⁱ w.r.t. u = (1/w) · Σ i·cᵢ ξⁱ⁻¹
        dval = 5 * c[5]
        dval = dval * xi + 4 * c[4]
        dval = dval * xi + 3 * c[3]
        dval = dval * xi + 2 * c[2]
        dval = dval * xi + 1 * c[1]
        return dval / self.w


# =========================================================================
# Main camera model
# =========================================================================

class PushbroomCamera:
    """
    Steviapp-compatible pushbroom camera with polynomial lens distortion.

    Parameters
    ----------
    f     : focal length [pixels]
    ppx   : principal point [pixels] (across-track optical axis position)
    w     : detector width [pixels] (normalisation base for polynomials)
    delta_x_coeffs : [a0, ..., a5] across-track distortion polynomial
    delta_y_coeffs : [b0, ..., b5] along-track (smile) distortion polynomial
    first_valid_pixel : BIL sample index of pixel u=0  (default 0)
    angle_lut_path    : optional lab angle LUT for validation / FOV limits
    """

    def __init__(self, f: float, ppx: float, w: int,
                 delta_x_coeffs: list, delta_y_coeffs: list,
                 first_valid_pixel: int = 0,
                 angle_lut_path: Optional[str] = None):

        self.f = float(f)
        self.ppx = float(ppx)
        self.w = int(w)
        self.num_pixels = self.w
        self.first_valid_pixel = first_valid_pixel

        # Distortion polynomials
        self.dist_x = DistortionPoly(delta_x_coeffs, float(self.w))
        self.dist_y = DistortionPoly(delta_y_coeffs, float(self.w))

        # FOV limits — compute from the distortion-corrected edge pixels
        self._compute_fov()

        # Optional: load angle LUT for validation / diagnostics
        self.angles_deg = None
        if angle_lut_path is not None:
            path = Path(angle_lut_path)
            if path.exists():
                self.angles_deg = np.loadtxt(str(path), dtype=np.float64)

        # ---- Diagnostics ----
        u_edges = np.array([0.0, self.w - 1.0])
        theta_edges = np.rad2deg(np.arctan(self._u_to_tan_xt(u_edges)))
        fov = theta_edges[1] - theta_edges[0]
        mean_ifov = fov / (self.w - 1)  # degrees per pixel

        # Smile amplitude from Δy
        u_all = np.arange(self.w, dtype=np.float64)
        dy_all = self.dist_y.evaluate(u_all)
        smile_amp_px = np.ptp(dy_all)
        smile_amp_arcsec = np.rad2deg(np.arctan(smile_amp_px / self.f)) * 3600

        print(f"  Camera [Steviapp]: {self.w} px, "
              f"f={self.f:.1f}, ppx={self.ppx:.2f}, "
              f"FOV={fov:.2f}°, IFOV={mean_ifov*1000:.1f} mrad")
        print(f"  Δx range: [{self.dist_x.evaluate(u_edges)[0]:+.1f}, "
              f"{self.dist_x.evaluate(u_edges)[1]:+.1f}] px")
        print(f"  Δy smile: {smile_amp_px:.1f} px amplitude "
              f"({smile_amp_arcsec:.0f}\")")
        if self.first_valid_pixel != 0:
            print(f"  BIL offset: first_valid_pixel={self.first_valid_pixel}")
        if self.angles_deg is not None:
            self._validate_against_lut()

    # =================================================================
    # Internal helpers
    # =================================================================

    def _u_to_tan_xt(self, u: np.ndarray) -> np.ndarray:
        """
        Pixel → across-track tangent (Xc/Zc) via Steviapp model.

        From:  u = f · (Xc/Zc) + ppx − Δx(u)
        Solve: Xc/Zc = (u + Δx(u) − ppx) / f
        """
        dx = self.dist_x.evaluate(u)
        return (u + dx - self.ppx) / self.f

    def _compute_fov(self):
        """Compute FOV angular limits from edge pixels."""
        u_left = np.array([0.0])
        u_right = np.array([float(self.w - 1)])
        tan_left = self._u_to_tan_xt(u_left)[0]
        tan_right = self._u_to_tan_xt(u_right)[0]
        self.theta_min_rad = np.arctan(tan_left)
        self.theta_max_rad = np.arctan(tan_right)
        self.tan_xt_min = tan_left
        self.tan_xt_max = tan_right

    def _validate_against_lut(self):
        """Compare polynomial model against lab angle LUT."""
        u_all = np.arange(len(self.angles_deg), dtype=np.float64)
        tan_poly = self._u_to_tan_xt(u_all)
        tan_lut = np.tan(np.deg2rad(self.angles_deg))
        diff_px = (tan_poly - tan_lut) * self.f  # difference in pixel units
        print(f"  LUT validation: Δx model vs LUT: "
              f"RMS={np.sqrt(np.mean(diff_px**2)):.4f} px, "
              f"max={np.max(np.abs(diff_px)):.4f} px")

    # =================================================================
    # Forward model:  pixel → ray direction  (for ray-casting / tie points)
    # =================================================================

    def pixel_to_ray(self, u_pixel: np.ndarray) -> np.ndarray:
        """
        Convert pixel index to 3D look-direction vector in camera frame.

        From the Steviapp model at v = 0:
            Xc/Zc = (u + Δx(u) − ppx) / f
            Yc/Zc = Δy(u) / f
            ray   = [Xc/Zc,  Yc/Zc,  1]

        Parameters
        ----------
        u_pixel : (N,) across-track pixel indices (may be fractional)

        Returns
        -------
        ray : (N, 3) direction vectors in camera frame (not normalised)
        """
        u = np.asarray(u_pixel, dtype=np.float64)
        dx = self.dist_x.evaluate(u)
        dy = self.dist_y.evaluate(u)

        Xc_Zc = (u + dx - self.ppx) / self.f
        Yc_Zc = dy / self.f

        return np.column_stack([Xc_Zc, Yc_Zc, np.ones_like(u)])

    # =================================================================
    # Inverse model:  ray → pixel  (for bottom-up orthorectification)
    # =================================================================

    def project(self, d_cam: np.ndarray) -> np.ndarray:
        """
        Project camera-frame direction(s) to across-track pixel index.

        Solves:  u = f · (Xc/Zc) + ppx − Δx(u)

        Uses Newton iteration (not just fixed-point) for fast convergence:
            g(u) = u + Δx(u) − u_ideal = 0
            u_{n+1} = u_n − g(u_n) / g'(u_n)
            g'(u) = 1 + dΔx/du

        Parameters
        ----------
        d_cam : (..., 3) direction(s) in camera frame

        Returns
        -------
        u_pixel : (...,) across-track pixel index (fractional)
        """
        Xc = d_cam[..., 0]
        Zc = d_cam[..., 2]

        u_ideal = self.f * (Xc / Zc) + self.ppx

        # Newton iteration:  g(u) = u + Δx(u) − u_ideal = 0
        u = u_ideal.copy()
        for _ in range(8):
            dx = self.dist_x.evaluate(u)
            ddx = self.dist_x.evaluate_derivative(u)
            g = u + dx - u_ideal
            gp = 1.0 + ddx
            u = u - g / gp

        return u

    # =================================================================
    # Along-track residual for Newton solver
    # =================================================================

    def along_track_residual(self, d_cam: np.ndarray,
                             u_pixel: np.ndarray) -> np.ndarray:
        """
        Along-track residual for the Newton time-iteration.

        From the Steviapp model:
            v = f · (Yc/Zc) − Δy(u)  →  0

        In tangent-space units (dividing by f):
            r = Yc/Zc − Δy(u)/f

        Parameters
        ----------
        d_cam   : (..., 3) points in camera frame
        u_pixel : (...,) across-track pixel indices (from project())

        Returns
        -------
        residual : (...,) [tangent-space units]
        """
        Yc = d_cam[..., 1]
        Zc = d_cam[..., 2]

        Y_obs = Yc / Zc
        Y_model = self.dist_y.evaluate(u_pixel) / self.f

        return Y_obs - Y_model

    # =================================================================
    # BIL sample mapping
    # =================================================================

    def pixel_to_bil_sample(self, u_pixel: np.ndarray) -> np.ndarray:
        """
        Map model pixel index to BIL sample index.

            bil_sample = u_pixel + first_valid_pixel
        """
        return u_pixel + self.first_valid_pixel

    # =================================================================
    # FOV validity check
    # =================================================================

    def is_within_fov(self, d_cam: np.ndarray) -> np.ndarray:
        """
        Check if camera-frame direction falls within the sensor FOV.

        The FOV is defined by the across-track angle range of the edge
        pixels (u=0 and u=w−1), including their distortion offsets.
        """
        Xc = d_cam[..., 0]
        Zc = d_cam[..., 2]
        tan_xt = Xc / Zc

        return ((tan_xt >= self.tan_xt_min) &
                (tan_xt <= self.tan_xt_max))

    def is_valid_pixel(self, u_pixel: np.ndarray) -> np.ndarray:
        """True if pixel is within detector bounds [0, w)."""
        return (u_pixel >= 0) & (u_pixel < self.w)
