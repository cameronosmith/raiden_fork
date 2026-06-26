"""View SO100 robot with large 8x8 alignment board in simulation.

This demo loads the SO100 robot with adhesive ArUco markers and places a large
8x8 ArUco board in front of it for pointcloud alignment demonstrations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import mujoco.viewer
import numpy as np

from ExoConfigs.so100_adhesive import SO100_ADHESIVE_CONFIG
from ExoConfigs.alignment_board import ALIGNMENT_BOARD_CONFIG
from exo_utils import position_exoskeleton_meshes, get_link_poses_from_robot

# Load robot config and add alignment board
robot_config = SO100_ADHESIVE_CONFIG

# Combine XMLs
import xml.etree.ElementTree as ET

# Parse robot XML
robot_root = ET.fromstring(robot_config.xml)

# Parse alignment board XML addition
board_xml = ALIGNMENT_BOARD_CONFIG.get_xml_addition()
board_root = ET.fromstring(f"<mujoco>{board_xml}</mujoco>")

# Merge assets
robot_asset = robot_root.find('asset')
board_asset = board_root.find('asset')
if robot_asset is not None and board_asset is not None:
    for child in board_asset:
        robot_asset.append(child)

# Merge worldbody
robot_worldbody = robot_root.find('worldbody')
board_worldbody = board_root.find('worldbody')
if robot_worldbody is not None and board_worldbody is not None:
    for child in board_worldbody:
        robot_worldbody.append(child)

# Convert back to string
combined_xml = ET.tostring(robot_root, encoding='unicode')

# Load model
print(f"Loading {robot_config.name} with {ALIGNMENT_BOARD_CONFIG.name}")
model = mujoco.MjModel.from_xml_string(combined_xml)
data = mujoco.MjData(model)

# Set robot to a reasonable pose
data.qpos = data.ctrl = np.array([0, -1.57, 1.57, 1.57, 1.57, 0])
mujoco.mj_forward(model, data)

# Launch viewer
print("\nLaunching interactive viewer...")
print("The alignment board is positioned in front of the robot on the table")
viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)

while viewer.is_running():
    # Update exoskeleton meshes to match robot pose
    position_exoskeleton_meshes(robot_config, model, data, get_link_poses_from_robot(robot_config, model, data))
    mujoco.mj_step(model, data)
    viewer.sync()

