"""

camera.py`
    Manages the camera matrix and lens distortion.

"""

import numpy as np

class CameraModel:
    def __init__(self, width, height, fx, fy, cx, cy, distortion=None):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.dist = np.array(distortion) if distortion else np.zeros(5)

    @classmethod
    def from_dict(cls, cfg):
        """Factory method to handle mm to pixel conversion."""
        if 'focal_length_mm' in cfg and 'pixel_size_um' in cfg:
            f_pix = cfg['focal_length_mm'] / (cfg['pixel_size_um'] / 1000.0)
            fx, fy = f_pix, f_pix
        else:
            fx, fy = cfg.get('fx'), cfg.get('fy')

        return cls(
            width=cfg['width'], height=cfg['height'],
            fx=fx, fy=fy,
            cx=cfg.get('cx', cfg['width']/2),
            cy=cfg.get('cy', cfg['height']/2),
            distortion=cfg.get('distortion')
        )

