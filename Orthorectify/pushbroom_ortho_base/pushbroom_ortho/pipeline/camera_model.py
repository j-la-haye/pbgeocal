"""
camera_model.py — Pushbroom camera with physical focal-plane correction.

=============================================================================
 Three-Step Physical Model
=============================================================================

Step 1 · Laboratory Baseline Model (Static Distortion)
-------------------------------------------------------
The lab calibration measures the across-track viewing angle θ_ac(px) for each
pixel.  We project these into normalised 2D coordinates on an ideal focal
plane at Z = 1:

    X_lab(px) = tan(θ_ac(px))                   from the angle CSV LUT

For the along-track smile, the polynomial is evaluated at the across-track
angle IN DEGREES (the natural calibration domain):

    θ_at_deg(px) = a · θ_ac_deg(px)² + b · θ_ac_deg(px) + c

    Y_lab(px) = tan(θ_at_deg(px) · π/180)

The 3D look vector in the lab frame is V_lab = [X_lab, Y_lab, 1]^T.

Step 2 · In-Flight Physical Affine Correction (Dynamic Distortion)
-------------------------------------------------------------------
Temperature and pressure changes during flight shift the focal length
f_lab → f_flt and the principal point.  The physical position of pixel px
on the silicon sensor does not change:

    x_sensor(px) = f_lab · X_lab(px) + cx_lab
    y_sensor(px) = f_lab · Y_lab(px) + cy_lab

The in-flight viewing ray through that same sensor position:

    X_flt(px) = (f_lab/f_flt) · X_lab(px) − Δcx/f_flt
    Y_flt(px) = (f_lab/f_flt) · Y_lab(px) − Δcy/f_flt

Step 3 · Optimisation Parameterisation
---------------------------------------
Three stable parameters (from bundle adjustment or GCPs):

    kf = f_lab / f_flt       focal length scale factor    (≈ 1)
    δx = Δcx / f_flt         across-track PP angular shift
    δy = Δcy / f_flt         along-track PP angular shift

    X_flt(px) = kf · X_lab(px) − δx
    Y_flt(px) = kf · Y_lab(px) − δy

With kf=1, δx=δy=0 → pure lab calibration.

Config mapping:
    kf = f_lab / f      (f_lab, f both in pixel units)
    δx = cx / f          (cx in pixels, f in pixels)
    δy = cy / f

=============================================================================
 Bottom-Up Inverse (ray → pixel) for Orthorectification
=============================================================================

Given a camera-frame direction d_cam = [Xc, Yc, Zc]:

    X_obs = Xc / Zc
    X_lab = (X_obs + δx) / kf
    θ_ac  = arctan(X_lab)
    u_pixel = LUT_inverse(θ_ac)

Along-track residual:

    θ_ac_deg_at_u = interp(u_pixel, pixel_indices, angles_deg)
    Y_lab_at_u    = tan(smile(θ_ac_deg_at_u) · π/180)
    Y_flt_pred    = kf · Y_lab_at_u − δy
    residual      = Y_obs − Y_flt_pred

BIL sample mapping (when LUT covers only a subset of BIL detector):

    bil_sample = u_pixel + first_valid_pixel

=============================================================================
 Frame Convention
=============================================================================
Camera frame:  X-right (across-track), Y-back (along-track), Z-down (nadir).
Angles in degrees for LUT storage and smile evaluation, radians internally.
Pixel indices:  0-based, integer at pixel centres.
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

        θ_at_deg(θ_ac_deg) = a · θ_ac_deg² + b · θ_ac_deg + c

    where θ_ac_deg is the across-track angle IN DEGREES from the LUT.

    IMPORTANT: The polynomial input is the angle in degrees, NOT the tangent.
    This matches the lab calibration convention where the polynomial was fit
    to angular positions across the detector.

    c : boresight along-track offset (sensor line not at θ_y = 0)
    a : quadratic smile curvature (< 0 = concave / smile)
    b : linear asymmetric tilt
    """
    a: float    # quadratic (smile curvature)  [deg / deg²]
    b: float    # linear (smile tilt)           [deg / deg]
    c: float    # constant (boresight offset)   [deg]

    def evaluate_deg(self, theta_ac_deg: np.ndarray) -> np.ndarray:
        """Along-track smile angle [degrees] given θ_ac in degrees."""
        return self.a * theta_ac_deg**2 + self.b * theta_ac_deg + self.c

    def evaluate_rad(self, theta_ac_deg: np.ndarray) -> np.ndarray:
        """Along-track smile angle [radians]."""
        return np.deg2rad(self.evaluate_deg(theta_ac_deg))

    def evaluate_tan(self, theta_ac_deg: np.ndarray) -> np.ndarray:
        """
        Along-track Y_lab = tan(θ_at · π/180).

        This is Y_lab from Step 1 of the formulation.
        """
        return np.tan(self.evaluate_rad(theta_ac_deg))


# =========================================================================
# In-flight optics correction
# =========================================================================

@dataclass
class OpticsCorrection:
    """
    In-flight physical affine correction on the focal plane.

    User-facing config parameters:
        f_lab : lab focal length [pixels]
        f     : in-flight focal length [pixels]  (= f_lab if no correction)
        cx    : across-track PP shift [pixels] from lab optical axis
        cy    : along-track  PP shift [pixels] from lab optical axis

    Derived parameters (computed by bind()):
        kf      = f_lab / f         focal length scale factor  (≈ 1)
        delta_x = cx / f            across-track PP angular shift
        delta_y = cy / f            along-track PP angular shift

    Forward: X_flt = kf · X_lab − delta_x
    Inverse: X_lab = (X_obs + delta_x) / kf
    """
    f_lab: float = 1762.2
    f: float = 1762.2
    cx: float = 0.0
    cy: float = 0.0

    # Derived (set by bind())
    kf: float = 1.0
    delta_x: float = 0.0
    delta_y: float = 0.0
    _bound: bool = False

    def bind(self):
        """Compute derived affine parameters.  Called by PushbroomCamera.__init__."""
        if self.f <= 0:
            raise ValueError(f"In-flight focal length f must be > 0, got {self.f}")
        self.kf = self.f_lab / self.f
        self.delta_x = self.cx / self.f
        self.delta_y = self.cy / self.f
        self._bound = True


# =========================================================================
# Main camera model
# =========================================================================

class PushbroomCamera:
    """
    Pushbroom camera model with physical focal-plane correction.

    Parameters
    ----------
    angle_lut_path : path to CSV with one across-track angle [deg] per line
    smile          : SmileModel with polynomial coefficients
    correction     : OpticsCorrection (default: identity = pure lab)
    first_valid_pixel : offset from LUT pixel 0 to BIL sample index
                        (0 if LUT covers the full detector,
                         >0 if LUT covers a subset starting at this BIL sample)
    """

    def __init__(self, angle_lut_path: str, smile: SmileModel,
                 correction: OpticsCorrection = None,
                 first_valid_pixel: int = 0):
        path = Path(angle_lut_path)
        if not path.exists():
            raise FileNotFoundError(f"Angle LUT not found: {angle_lut_path}")

        # ---- Load lab-measured across-track angles [degrees] ----
        self.angles_deg = np.loadtxt(str(path), dtype=np.float64)
        self.num_pixels = len(self.angles_deg)
        self.pixel_indices = np.arange(self.num_pixels, dtype=np.float64)

        # Pre-compute radians and tangents (X_lab per pixel)
        self.angles_rad = np.deg2rad(self.angles_deg)
        self.X_lab = np.tan(self.angles_rad)

        # Verify monotonicity (required for LUT inverse via np.interp)
        if not np.all(np.diff(self.angles_deg) > 0):
            raise ValueError("Angle LUT must be monotonically increasing")

        # ---- FOV angular bounds ----
        self.theta_min_rad = self.angles_rad[0]
        self.theta_max_rad = self.angles_rad[-1]

        # ---- Mean pixel pitch in tangent space ----
        self.pixel_pitch_tan = float(np.mean(np.diff(self.X_lab)))

        # ---- BIL sample offset ----
        # When the LUT covers only valid pixels (a subset of the BIL detector),
        # first_valid_pixel maps LUT pixel 0 → BIL sample first_valid_pixel.
        self.first_valid_pixel = first_valid_pixel

        # ---- Smile model ----
        self.smile = smile

        # Pre-compute Y_lab for each pixel (Step 1):
        #   Y_lab(px) = tan(smile(θ_ac_deg(px)) · π/180)
        #
        # SIGN CONVENTION:  The smile polynomial was calibrated in the
        # standard photogrammetric frame where Y points FORWARD (toward
        # flight direction).  Our camera frame has Y-BACK (opposite to
        # flight).  The sign of Y_lab must therefore be NEGATED.
        #
        # Without negation:  edges look forward → convex toward flight (WRONG)
        # With negation:     edges look backward → concave toward flight (CORRECT)
        self.Y_lab = -self.smile.evaluate_tan(self.angles_deg)

        # ---- In-flight optics correction ----
        self.correction = correction or OpticsCorrection()
        self.correction.bind()

        # ---- Diagnostics ----
        fov = self.angles_deg[-1] - self.angles_deg[0]
        mean_ifov = np.mean(np.diff(self.angles_deg))
        smile_deg = self.smile.evaluate_deg(self.angles_deg)
        smile_amp_arcsec = np.ptp(smile_deg) * 3600.0

        zero_pixel = float(np.interp(0.0, self.angles_rad, self.pixel_indices))
        geom_center = (self.num_pixels - 1) / 2.0

        c = self.correction
        print(f"  Camera: {self.num_pixels} px, FOV={fov:.2f}°, "
              f"IFOV={mean_ifov*1000:.1f} mrad, "
              f"smile={smile_amp_arcsec:.1f}\"")
        print(f"  Optical axis (lab): pixel {zero_pixel:.2f} "
              f"(detector center={geom_center:.1f}, "
              f"offset={zero_pixel - geom_center:+.2f} px)")
        if self.first_valid_pixel != 0:
            print(f"  BIL offset: first_valid_pixel={self.first_valid_pixel} "
                  f"(LUT pixel 0 → BIL sample {self.first_valid_pixel})")
        if c.kf != 1.0 or c.delta_x != 0.0 or c.delta_y != 0.0:
            delta_f_ppm = (1.0 / c.kf - 1.0) * 1e6
            print(f"  Flight correction: kf={c.kf:.6f} "
                  f"(Δf={delta_f_ppm:+.1f} ppm), "
                  f"δx={c.delta_x:+.4e}, δy={c.delta_y:+.4e}")

    # =================================================================
    # Forward model:  pixel → look direction (for ray casting / tie points)
    # =================================================================

    def pixel_to_ray(self, u_pixel: np.ndarray) -> np.ndarray:
        """
        Convert LUT pixel indices to 3D look-direction vectors in camera frame.

        Steps 1→2→3:
            1.  X_lab = tan(θ_ac(u)) from LUT
                Y_lab = tan(smile(θ_ac_deg(u)) · π/180)
            2–3.  X_flt = kf · X_lab − δx
                  Y_flt = kf · Y_lab − δy
            4.  ray = [X_flt, Y_flt, 1]

        Parameters
        ----------
        u_pixel : (N,) across-track LUT pixel indices (may be fractional)

        Returns
        -------
        ray : (N, 3) direction vectors in camera frame (NOT normalised)
        """
        c = self.correction

        # Step 1: Lab coordinates from LUT
        theta_lab_rad = np.interp(u_pixel, self.pixel_indices, self.angles_rad)
        theta_lab_deg = np.interp(u_pixel, self.pixel_indices, self.angles_deg)
        X_lab = np.tan(theta_lab_rad)
        Y_lab = -self.smile.evaluate_tan(theta_lab_deg)  # negated: cal Y-fwd → cam Y-back

        # Steps 2–3: Flight correction
        X_flt = c.kf * X_lab - c.delta_x
        Y_flt = c.kf * Y_lab - c.delta_y

        return np.column_stack([X_flt, Y_flt, np.ones_like(X_flt)])

    # =================================================================
    # Inverse model:  ray → pixel (for bottom-up orthorectification)
    # =================================================================

    def project(self, d_cam: np.ndarray):
        """
        Project camera-frame direction(s) to across-track pixel index.

        Inverts the forward model:
            X_obs = Xc / Zc
            X_lab = (X_obs + δx) / kf
            θ_ac  = arctan(X_lab)
            u_pixel = LUT_inverse(θ_ac)

        Parameters
        ----------
        d_cam : (..., 3) direction(s) in camera frame [X-right, Y-back, Z-down]

        Returns
        -------
        u_pixel : (...,) across-track LUT pixel index (fractional)
                  NOTE: np.interp clamps — use is_within_fov() for bounds.
        """
        Xc = d_cam[..., 0]
        Zc = d_cam[..., 2]
        c = self.correction

        # Observed across-track tangent
        X_obs = Xc / Zc

        # Invert flight correction: X_lab = (X_obs + δx) / kf
        X_lab = (X_obs + c.delta_x) / c.kf

        # Map X_lab → angle → pixel via LUT
        theta_ac = np.arctan(X_lab)
        u_pixel = np.interp(theta_ac, self.angles_rad, self.pixel_indices)

        return u_pixel

    # =================================================================
    # Along-track residual for Newton solver
    # =================================================================

    def along_track_residual(self, d_cam: np.ndarray,
                             u_pixel: np.ndarray) -> np.ndarray:
        """
        Along-track residual for Newton iteration.

        At the correct time t*, the observed along-track tangent must match
        the flight-corrected smile prediction:

            Y_obs = Y_flt(u) = kf · Y_lab(u) − δy

        Residual:
            r = Y_obs − Y_flt(u)
              = Yc/Zc − [kf · Y_lab(u) − δy]
              = Yc/Zc + δy − kf · Y_lab(u)

        where Y_lab(u) = tan(smile(θ_ac_deg(u)) · π/180).

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
        c = self.correction

        Y_obs = Yc / Zc

        # Smile at the projected pixel position.
        # Get θ_ac in degrees at u_pixel, then evaluate smile (negated for Y-back frame).
        theta_ac_deg = np.interp(u_pixel, self.pixel_indices, self.angles_deg)
        Y_lab_at_u = -self.smile.evaluate_tan(theta_ac_deg)

        # Predicted Y_flt
        Y_flt_predicted = c.kf * Y_lab_at_u - c.delta_y

        return Y_obs - Y_flt_predicted

    # =================================================================
    # BIL sample mapping
    # =================================================================

    def pixel_to_bil_sample(self, u_pixel: np.ndarray) -> np.ndarray:
        """
        Map LUT pixel indices to BIL sample indices.

        When the angle LUT covers only a subset of the BIL detector
        (e.g. excluding dark pixels at detector edges), the LUT pixel
        index 0 corresponds to BIL sample first_valid_pixel.

            bil_sample = u_pixel + first_valid_pixel

        Parameters
        ----------
        u_pixel : (...,) LUT pixel indices (fractional)

        Returns
        -------
        bil_sample : (...,) BIL sample indices (fractional)
        """
        return u_pixel + self.first_valid_pixel

    # =================================================================
    # FOV validity check (defeats np.interp clamping)
    # =================================================================

    def is_within_fov(self, d_cam: np.ndarray) -> np.ndarray:
        """
        Check if a camera-frame direction falls within the sensor FOV.

        Parameters
        ----------
        d_cam : (..., 3) points in camera frame

        Returns
        -------
        within : (...,) boolean
        """
        Xc = d_cam[..., 0]
        Zc = d_cam[..., 2]
        c = self.correction

        X_obs = Xc / Zc
        X_lab = (X_obs + c.delta_x) / c.kf
        theta_ac = np.arctan(X_lab)

        return ((theta_ac >= self.theta_min_rad) &
                (theta_ac <= self.theta_max_rad))

    # =================================================================
    # Convenience
    # =================================================================

    def is_valid_pixel(self, u_pixel: np.ndarray) -> np.ndarray:
        """True if pixel is within sensor bounds [0, num_pixels)."""
        return (u_pixel >= 0) & (u_pixel < self.num_pixels)
