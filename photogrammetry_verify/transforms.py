#### `transforms.py`
"""
Handles coordinate frame transformations. This is where you manage the **Body-to-Camera** logic.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

class Transform:
    def __init__(self, translation, rotation_euler, order='xyz', degrees=True):
        self.r = R.from_euler(order, rotation_euler, degrees=degrees)
        self.t = np.array(translation)

    def apply(self, points):
        """P_target = R * P_source + t"""
        return self.r.apply(points) + self.t

    def inverse(self):
        inv_r = self.r.inv()
        inv_t = -inv_r.apply(self.t)
        # Construct via rotation matrix to maintain precision
        obj = Transform([0,0,0], [0,0,0])
        obj.r = inv_r
        obj.t = inv_t
        return obj
