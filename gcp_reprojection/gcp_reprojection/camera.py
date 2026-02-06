"""
Camera model module for projecting 3D points to image coordinates.

Implements the pinhole camera model with optional lens distortion correction.

Coordinate System:
    - Camera frame: X-right, Y-back, Z-down (looking along +Z)
    - Image frame: u-right, v-down (origin at top-left corner)
    
Projection Model:
    1. Perspective projection: x' = X/Z, y' = Y/Z
    2. Distortion (optional): Apply radial and tangential distortion
    3. Pixel mapping: u = fx*x' + cx, v = fy*y' + cy
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

from .config import CameraIntrinsics

logger = logging.getLogger(__name__)


class CameraModel:
    """
    Camera projection model implementing pinhole projection with distortion.
    
    The distortion model follows OpenCV conventions:
        - Radial distortion: k1, k2, k3
        - Tangential distortion: p1, p2
        
    Distortion equations (applied to normalized coordinates x', y'):
        r² = x'² + y'²
        x'' = x'(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x'*y' + p2*(r² + 2*x'²)
        y'' = y'(1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y'²) + 2*p2*x'*y'
    """
    
    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initialize camera model with intrinsic parameters.
        
        Args:
            intrinsics: Camera intrinsic parameters
        """
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.cx
        self.cy = intrinsics.cy
        
        # Distortion coefficients
        self.k1 = intrinsics.k1
        self.k2 = intrinsics.k2
        self.k3 = intrinsics.k3
        self.p1 = intrinsics.p1
        self.p2 = intrinsics.p2
        
        # Image dimensions
        self.image_width = intrinsics.image_width
        self.image_height = intrinsics.image_height
        
        # Check if distortion is significant
        self.has_distortion = not np.allclose(
            [self.k1, self.k2, self.k3, self.p1, self.p2], 0
        )
        
        # Camera matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        logger.debug(f"Camera model initialized: fx={self.fx}, fy={self.fy}")
        logger.debug(f"Principal point: ({self.cx}, {self.cy})")
        logger.debug(f"Distortion enabled: {self.has_distortion}")
    
    def project_point(
        self,
        point_camera: np.ndarray,
        apply_distortion: bool = True,
    ) -> Tuple[float, float, bool]:
        """
        Project a 3D point in camera frame to image coordinates.
        
        Args:
            point_camera: 3D point in camera frame (X-right, Y-back, Z-down)
            apply_distortion: Whether to apply lens distortion
            
        Returns:
            Tuple of (u, v, valid) where:
                - u: Horizontal pixel coordinate
                - v: Vertical pixel coordinate  
                - valid: True if point is in front of camera and within image
        """
        X, Y, Z = point_camera
        
        # Check if point is behind camera
        if Z <= 0:
            logger.debug(f"Point behind camera: Z={Z}")
            return 0.0, 0.0, False
        
        # Perspective projection to normalized coordinates
        x_norm = X / Z
        y_norm = Y / Z
        
        # Apply distortion if enabled
        if apply_distortion and self.has_distortion:
            x_dist, y_dist = self._apply_distortion(x_norm, y_norm)
        else:
            x_dist, y_dist = x_norm, y_norm
        
        # Map to pixel coordinates
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        
        # Check if within image bounds
        valid = True
        if self.image_width > 0 and self.image_height > 0:
            valid = (0 <= u < self.image_width) and (0 <= v < self.image_height)
        
        return u, v, valid
    
    def _apply_distortion(
        self, x_norm: float, y_norm: float
    ) -> Tuple[float, float]:
        """
        Apply lens distortion to normalized coordinates.
        
        Uses the Brown-Conrady distortion model (OpenCV convention).
        
        Args:
            x_norm: Normalized x coordinate (X/Z)
            y_norm: Normalized y coordinate (Y/Z)
            
        Returns:
            Distorted (x, y) normalized coordinates
        """
        r2 = x_norm ** 2 + y_norm ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3
        
        # Radial distortion factor
        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        
        # Tangential distortion
        x_tangential = 2 * self.p1 * x_norm * y_norm + self.p2 * (r2 + 2 * x_norm ** 2)
        y_tangential = self.p1 * (r2 + 2 * y_norm ** 2) + 2 * self.p2 * x_norm * y_norm
        
        # Combined distortion
        x_dist = x_norm * radial + x_tangential
        y_dist = y_norm * radial + y_tangential
        
        return x_dist, y_dist
    
    def project_points_batch(
        self,
        points_camera: np.ndarray,
        apply_distortion: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project multiple 3D points to image coordinates.
        
        Args:
            points_camera: Nx3 array of camera frame coordinates
            apply_distortion: Whether to apply lens distortion
            
        Returns:
            Tuple of:
                - u_coords: N-element array of u coordinates
                - v_coords: N-element array of v coordinates
                - valid: N-element boolean array indicating valid projections
        """
        n_points = len(points_camera)
        u_coords = np.zeros(n_points)
        v_coords = np.zeros(n_points)
        valid = np.zeros(n_points, dtype=bool)
        
        for i, point in enumerate(points_camera):
            u_coords[i], v_coords[i], valid[i] = self.project_point(
                point, apply_distortion
            )
        
        return u_coords, v_coords, valid
    
    def undistort_point(
        self,
        u: float,
        v: float,
        max_iterations: int = 10,
        tolerance: float = 1e-8,
    ) -> Tuple[float, float]:
        """
        Remove distortion from pixel coordinates (inverse distortion).
        
        Uses iterative refinement to solve for undistorted coordinates.
        
        Args:
            u: Distorted u coordinate
            v: Distorted v coordinate
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Undistorted (u, v) pixel coordinates
        """
        if not self.has_distortion:
            return u, v
        
        # Convert to normalized coordinates
        x_dist = (u - self.cx) / self.fx
        y_dist = (v - self.cy) / self.fy
        
        # Initial guess: distorted = undistorted
        x_norm = x_dist
        y_norm = y_dist
        
        # Iterative refinement
        for _ in range(max_iterations):
            x_curr, y_curr = self._apply_distortion(x_norm, y_norm)
            
            # Error
            dx = x_dist - x_curr
            dy = y_dist - y_curr
            
            if abs(dx) < tolerance and abs(dy) < tolerance:
                break
            
            # Update estimate
            x_norm += dx
            y_norm += dy
        
        # Convert back to pixel coordinates
        u_undist = self.fx * x_norm + self.cx
        v_undist = self.fy * y_norm + self.cy
        
        return u_undist, v_undist
    
    def get_projection_jacobian(
        self,
        point_camera: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the Jacobian of the projection function.
        
        Useful for error propagation and uncertainty analysis.
        
        Args:
            point_camera: 3D point in camera frame
            
        Returns:
            2x3 Jacobian matrix d(u,v)/d(X,Y,Z)
        """
        X, Y, Z = point_camera
        
        if Z <= 0:
            return np.zeros((2, 3))
        
        Z2 = Z ** 2
        
        # For undistorted pinhole model:
        # u = fx * X/Z + cx
        # v = fy * Y/Z + cy
        # 
        # du/dX = fx/Z, du/dY = 0, du/dZ = -fx*X/Z²
        # dv/dX = 0, dv/dY = fy/Z, dv/dZ = -fy*Y/Z²
        
        J = np.array([
            [self.fx / Z, 0, -self.fx * X / Z2],
            [0, self.fy / Z, -self.fy * Y / Z2]
        ])
        
        return J
    
    def compute_reprojection_error(
        self,
        point_camera: np.ndarray,
        measured_u: float,
        measured_v: float,
        apply_distortion: bool = True,
    ) -> float:
        """
        Compute reprojection error for a single point.
        
        Args:
            point_camera: 3D point in camera frame
            measured_u: Measured u pixel coordinate
            measured_v: Measured v pixel coordinate
            apply_distortion: Whether to apply distortion in projection
            
        Returns:
            Euclidean distance between projected and measured points (pixels)
        """
        u_proj, v_proj, valid = self.project_point(point_camera, apply_distortion)
        
        if not valid:
            return float('inf')
        
        error = np.sqrt((u_proj - measured_u) ** 2 + (v_proj - measured_v) ** 2)
        return error
