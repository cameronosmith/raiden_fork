"""View the Piper arm with base-only exoskeleton in sim.

Uses robot_models/agilex_piper and ExoConfigs agilex_piper_base_only.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import numpy as np

from ExoConfigs import EXOSKELETON_CONFIGS
from exo_utils import position_exoskeleton_meshes, get_link_poses_from_robot

import argparse
parser = argparse.ArgumentParser(description="Piper arm sim with base-only exoskeleton")
parser.add_argument("--exo", type=str, default="agilex_piper_base_only", choices=list(EXOSKELETON_CONFIGS.keys()), help="Exoskeleton config (default: Piper base only)")
args = parser.parse_args()

robot_config = EXOSKELETON_CONFIGS[args.exo]
# Optional: hide exo link mesh alpha for cleaner view
if hasattr(robot_config, 'exo_link_alpha'):
    robot_config.exo_link_alpha = 0

print(f"Using exoskeleton config: {args.exo} ({robot_config.name})")
print(f"Model: {robot_config.base_xml_path}")

# Load model from config (includes robot_models/agilex_piper/piper.xml + exo)
model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)

# Piper home pose: 7 DOF (joint1–6 + gripper; joint8 = -joint7 via equality)
# Keyframe ctrl: "0 1.57 -1.3485 0 0 0 0"
#PIPER_HOME = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0,0.0])
#data.qpos[:] = PIPER_HOME
#mujoco.mj_forward(model, data)

print("\nLaunching interactive viewer...")
viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
while viewer.is_running():
    position_exoskeleton_meshes(robot_config, model, data, get_link_poses_from_robot(robot_config, model, data))
    mujoco.mj_step(model, data)
    viewer.sync()
