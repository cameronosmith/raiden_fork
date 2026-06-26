"""Estimate robot state from a single image.

This demo shows how to:
1. Load an RGB image
2. Detect ArUco markers
3. Estimate robot joint configuration
4. Render the estimated pose alongside the original image
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import matplotlib.pyplot as plt
import numpy as np
from mujoco.renderer import Renderer

from ExoConfigs import EXOSKELETON_CONFIGS
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, render_from_camera_pose, get_link_poses_from_robot

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

from ExoConfigs.panda_exo import PANDA_BASE_ONLY_CONFIG as robot_config

print(f"Using exoskeleton config: {robot_config.name})")
#image_path = '../redo_mujoco_calibration/random/tmpimgs/pretty_robot.png'
image_path = "scratch/panda3.png"

# Load model from config
model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)

# Use the same arm pose as at capture (e.g. all zeros). The Panda XML has a "home"
# keyframe with non-zero joint4-7; without this, the sim arm can be in a different pose.
n_arm = 7
data.qpos[:n_arm] = -.1
# optional: close gripper to match typical pose
# data.qpos[n_arm] = data.qpos[n_arm + 1] = 0.0

# Set virtual robot state from image
rgb = plt.imread(image_path)[..., :3]
if rgb.max() <= 1.0: rgb = (rgb * 255).astype(np.uint8)

# Detect link poses from ArUco markers

cam_K=None if 1 else np.array([[1.58847596e+03,0.00000000e+00, 9.59500000e+02], [0.00000000e+00, 1.58847596e+03, 5.39500000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, model, data, robot_config)
mujoco.mj_forward(model, data)
position_exoskeleton_meshes(robot_config, model, data, link_poses)
mujoco.mj_forward(model, data)

# Render from estimated camera pose and show on top of image
rendered = render_from_camera_pose(model, data, camera_pose_world, cam_K, *rgb.shape[:2])

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, img in zip(axes, [rgb, rendered, (rgb * 0.5 + rendered * 0.5).astype(np.uint8)]): ax.imshow(img);ax.axis('off')
plt.tight_layout()
plt.show()