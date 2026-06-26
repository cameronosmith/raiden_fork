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
parser.add_argument("--exo", type=str, default="so100_holemounts", choices=list(EXOSKELETON_CONFIGS.keys()), help="Exoskeleton configuration to use")
parser.add_argument("--just_sim_state", action="store_true", help="cam rerender but dont reset config")
parser.add_argument("--no_render", action="store_true", help="just render arm in sim")
args = parser.parse_args()

#robot_config = EXOSKELETON_CONFIGS[args.exo]

from ExoConfigs.arx import ARXConfig
ARXConfig.exo_alpha = 0.2
ARXConfig.aruco_alpha = .9  # Set to 0.
ARXConfig.exo_link_alpha = 0.2
robot_config = ARXConfig()

print(f"Using exoskeleton config: {args.exo} ({robot_config.name})")
#image_path = '../redo_mujoco_calibration/random/tmpimgs/pretty_robot.png'
image_path = "scratch/arx_frames/frame_00255.png"

# Load model from config
model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)

# Set virtual robot state from image
if not args.just_sim_state:    
    rgb = plt.imread(image_path)[..., :3]
    if rgb.max() <= 1.0: rgb = (rgb * 255).astype(np.uint8)

    # DEBUG: Only use base and first link for estimation
    original_links = robot_config.links
    robot_config.links = {k: v for k, v in original_links.items() if k in ["larger_base", "link3","link4","link6"]}
    print(f"DEBUG: Using only links: {list(robot_config.links.keys())}")

    # Detect link poses from ArUco markers
    link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, model, data, robot_config)
    
    # Estimate robot state using only the filtered links
    configuration = estimate_robot_state( model, data, robot_config, link_poses, ik_iterations=295)
    
    # Restore original links for rendering
    robot_config.links = original_links
    data.qpos[:] = configuration.q
    data.ctrl[:] = configuration.q[:len(data.ctrl)]
    mujoco.mj_forward(model, data)
    position_exoskeleton_meshes(robot_config, model, data, link_poses)
else:
    link_poses = get_link_poses_from_robot(robot_config, model, data)
    position_exoskeleton_meshes(robot_config, model, data, link_poses)
    mujoco.mj_forward(model, data)

# Render from estimated camera pose and show on top of image
if not args.no_render:
    rendered = render_from_camera_pose(model, data, camera_pose_world, cam_K, *rgb.shape[:2])
    plt.imshow((rgb * 0.5 + rendered * 0.5).astype(np.uint8))
    plt.show()
    zz
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img in zip(axes, [rgb, rendered, (rgb * 0.5 + rendered * 0.5).astype(np.uint8)]): ax.imshow(img);ax.axis('off')
    plt.tight_layout()
    plt.show()
# just Launch interactive viewer
else:
    print("\nLaunching interactive viewer...")
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()