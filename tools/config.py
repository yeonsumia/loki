
import copy
import os

import numpy as np


# original value: [0.10]
HEAD_RADIUS_MIN = 0.02
HEAD_RADIUS_MAX = 0.06

# original value: [500, 600, 700, 800, 900, 1000]
HEAD_DENSITY_MIN = 500.
HEAD_DENSITY_MAX = 1000.

# origianl value: [[-30, 0],[0, 30],[-30, 30],[-45, 45],[-45, 0],[0, 45],[-60, 0],[0, 60],[-60, 60],[-90, 0],[0, 90],[-60, 30],[-30, 60]]
JOINT_RANGE_MIN = 0.0
JOINT_RANGE_MAX = 90.0

# original value: [150, 200, 250, 300]
JOINT_GEAR_MIN = 150.
JOINT_GEAR_MAX = 300.

# original value: [0.05]
LIMB_RADIUS_MIN = 0.02
LIMB_RADIUS_MAX = 0.06

# original value: [0.2, 0.3, 0.4]
LIMB_HEIGHT_MIN = 0.2
LIMB_HEIGHT_MAX = 0.4

# original value: [500, 600, 700, 800, 900, 1000]
LIMB_DENSITY_MIN = 500.
LIMB_DENSITY_MAX = 1000.

LIMB_ORIENTATION_STEP_SIZE = 45
LIMB_ORIENTATION_PHI_MIN = np.radians(90)
LIMB_ORIENTATION_PHI_MAX = np.radians(180)
LIMB_ORIENTATION_THETA_MIN = np.radians(0)
LIMB_ORIENTATION_THETA_MAX = np.radians(360 - LIMB_ORIENTATION_STEP_SIZE)