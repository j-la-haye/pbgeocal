from raytracer2D3D import CameraProjector
import numpy as np
from liblibor import rotations
import matplotlib.pyplot as plt

def build_rotation_matrix( r, p, y):
        """Z-Y-X Euler sequence."""
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        R_z = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        R_y = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        R_x = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        #return R_x @ R_y @ R_z
        return R_z @ R_y @ R_x

rpy = np.array([-0.051027,0.06024340000000001,-2.2001398])  # roll, pitch, yaw in degrees
rpy_deg = np.degrees(rpy)

lF_topo = np.array([-0.004,0.03,1.264])
lR_topo = np.array([-0.110,0.26,1.177])  # Local frame topographic 'up' vector

R_enu_to_body = build_rotation_matrix(
            rpy[0], rpy[1], rpy[2]
        )  # roll, pitch, yaw in degrees

Rned2b = rotations.R_ned2b(rpy[0], rpy[1], rpy[2])
Rb2ned = rotations.R_b2ned(rpy[0], rpy[1], rpy[2]) 

lR_body = R_enu_to_body @ lR_topo
lF_body = R_enu_to_body @ lF_topo

lR_body_1 = Rned2b @ lR_topo
lF_body_1 = Rned2b @ lF_topo

lR_body_2 = Rb2ned @ lR_topo
lF_body_2 = Rb2ned @ lF_topo

print('Atlans-A7:','lR_body:', lR_body, 'lR_body_1:', lR_body_1, 'lR_body_2:', lR_body_2)
print('Riegl:''lF_body:', lF_body, 'lF_body_1:', lF_body_1, 'lF_body_2:', lF_body_2)

lR_body_SFS = np.array([-0.076, -0.378, 1.216])
lF_gsm_center_sfs = np.array([0.0650,0.0779,1.3395])


# Data from Table 2
# AIS - AVIRIS-4 (R)
ais_SFS = np.array([-0.202, 0.0758, 1.216])
ais_Cal = np.array([-0.333, -0.0121, 1.104])

# ALS - Riegl (F)
als_SFS = np.array([0.075, 0.112, 0.942])
als_Cal = np.array([0.108, 0.101, 0.900])

# Calculating Absolute Differences |Cal - Ref|
ais_abs_diff = np.abs(ais_Cal - ais_SFS)
als_abs_diff = np.abs(als_Cal - als_SFS)
# Labels and setup
labels = ['Right (R)', 'Forward (F)', 'Up (U)']
x = np.arange(len(labels))
width = 0.35

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
#rects1 = ax.bar(x - width/2, ais_abs_diff, width, label='AIS - AVIRIS-4 (R)', color="#d612d6")
rects2 = ax.bar(x + width/2, als_abs_diff, width, label='ALS - Riegl (F)', color="#0e62ff")

# Formatting
ax.set_ylabel('Absolute Difference |Calibration - Operator| [m]', fontsize=14)
ax.set_title('Absolute Difference between Operator and Calibrated Lever Arm Values', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adding value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

#autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('abs_lever_arm_diff.png')
plt.show()