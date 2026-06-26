"""Just view the exoskeleton in sim
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import matplotlib.pyplot as plt
import numpy as np
from mujoco.renderer import Renderer

from ExoConfigs import EXOSKELETON_CONFIGS, so100_adhesive
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, render_from_camera_pose, get_link_poses_from_robot
import pickle
from robot_models.so100_controller import Arm


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exo", type=str, default="so100_holemounts", choices=list(EXOSKELETON_CONFIGS.keys()), help="Exoskeleton configuration to use")
parser.add_argument("--just_sim_state", action="store_true", help="cam rerender but dont reset config")
parser.add_argument("--no_render", action="store_true", help="just render arm in sim")
parser.add_argument("--use_robot_state", action="store_true", help="just render arm in sim")
args = parser.parse_args()

so100_adhesive.exo_link_alpha = 0
robot_config = so100_adhesive.SO100AdhesiveConfig()#EXOSKELETON_CONFIGS[args.exo]

print(f"Using exoskeleton config: {args.exo} ({robot_config.name})")
image_path = '../redo_mujoco_calibration/random/tmpimgs/00000.png'

if args.use_robot_state: self=Arm( pickle.load(open("robot_models/arm_offsets/middleservo_calib_redo_fromimg.pkl", 'rb')) )

# Load model from config
model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

data.qpos=data.ctrl= np.array([0, -1.57, 1.57, 1.57, 1.57, 0]) 

mujoco.mj_forward(model, data)

print("\nLaunching interactive viewer...")
viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
while viewer.is_running():
    if args.use_robot_state:
        pos=self.get_pos()
        data.qpos[:] = data.ctrl[:] = pos
        print(self.get_pos())
    position_exoskeleton_meshes(robot_config, model, data, get_link_poses_from_robot(robot_config, model, data))
    mujoco.mj_step(model, data)
    viewer.sync()