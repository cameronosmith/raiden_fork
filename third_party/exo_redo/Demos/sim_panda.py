"""View the Panda arm with base-only exoskeleton in sim.

Uses robot_models/franka_emika_panda and ExoConfigs panda_base_only.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import numpy as np

from ExoConfigs import EXOSKELETON_CONFIGS
from ExoConfigs.panda_exo import PANDA_BASE_ONLY_CONFIG
from exo_utils import position_exoskeleton_meshes, get_link_poses_from_robot

import argparse
parser = argparse.ArgumentParser(description="Panda arm sim with base-only exoskeleton")
args = parser.parse_args()

robot_config = PANDA_BASE_ONLY_CONFIG
# Make exo and link overlay visible in the viewer (defaults are semi-transparent)
if hasattr(robot_config, "exo_link_alpha"):
    robot_config.exo_link_alpha = 1
if hasattr(robot_config, "exo_alpha"):
    robot_config.exo_alpha = 1
# Regenerate XML so the new alphas are used when building the model
robot_config.xml = robot_config._generate_xml()

print(f"Model: {robot_config.base_xml_path}")

# Load model from config (includes robot + exo mocap bodies)
model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)

# One forward pass so data.xpos/data.xquat are set; otherwise get_link_poses_from_robot
# reads zeros and the exo mocap bodies end up at the wrong pose (or origin).
mujoco.mj_forward(model, data)

print("\nLaunching interactive viewer...")
viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
while viewer.is_running():
    position_exoskeleton_meshes(robot_config, model, data, get_link_poses_from_robot(robot_config, model, data))
    mujoco.mj_step(model, data)
    viewer.sync()
