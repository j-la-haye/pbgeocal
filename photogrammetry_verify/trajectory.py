import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class TrajectoryInterpolator:
    def __init__(self, csv_path):
        """
        Expects CSV with columns: timestamp, x, y, z, roll, pitch, yaw
        Assumes Euler angles are in degrees.
        """
        self.df = pd.read_csv(csv_path).sort_values('timestamp')
        self.times = self.df['timestamp'].values
        self.positions = self.df[['x', 'y', 'z']].values
        
        # Create Rotation object for all poses (for SLERP)
        # Assuming input is degrees. Adjust 'xyz' order if needed.
        self.rotations = R.from_euler('xyz', self.df[['roll', 'pitch', 'yaw']].values, degrees=True)
        
        # create Slerp object
        self.slerp = Slerp(self.times, self.rotations)

    def get_pose_at_time(self, query_time):
        """
        Returns interpolated position (np.array) and rotation (scipy Rotation object).
        """
        if query_time < self.times[0] or query_time > self.times[-1]:
            raise ValueError(f"Time {query_time} is out of trajectory bounds.")

        # 1. Linear Interpolation for Position
        # Find index for linear interp
        idx = np.searchsorted(self.times, query_time)
        if idx == 0: return self.positions[0], self.rotations[0]
        
        t0, t1 = self.times[idx-1], self.times[idx]
        ratio = (query_time - t0) / (t1 - t0)
        
        p0, p1 = self.positions[idx-1], self.positions[idx]
        pos_interp = p0 + (p1 - p0) * ratio

        # 2. SLERP for Rotation
        rot_interp = self.slerp([query_time])[0]

        return pos_interp, rot_interp
