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
import mink
from scipy.spatial.transform import Rotation as R

from ExoConfigs.so100_holemounts import SO100_CONFIG
from ExoConfigs.puck import PUCK_CONFIG
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, detect_and_position_puck, render_from_camera_pose, combine_xmls, get_link_poses_from_robot, position_puck_aruco_from_mocap, track_aruco_with_ik

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--just_sim_state",action="store_true") # cam rerender but dont reset config
parser.add_argument("--no_render",action="store_true") # just render arm in sim
args = parser.parse_args()


robot_config = SO100_CONFIG
puck_config = PUCK_CONFIG
image_path = '../redo_mujoco_calibration/random/tmpimgs/00001.png'

# Load model from config
model = mujoco.MjModel.from_xml_string(combine_xmls(robot_config.xml, puck_config.get_xml_addition()))
data = mujoco.MjData(model)
data.qpos[:] = data.ctrl[:] = np.array([0, -1.48, 1.57, 1.51, -1.57, 1.75]) 
mujoco.mj_forward(model, data)

 # Setup IK for moving arm to ArUco offset
from loop_rate_limiters import RateLimiter
rate = RateLimiter(frequency=200.0, warn=False)
configuration = mink.Configuration(model)
configuration.update(data.qpos)

if not args.just_sim_state:    
    print("\nEstimating robot state from image...")
    rgb = plt.imread(image_path)[..., :3]
    if rgb.max() <= 1.0: rgb = (rgb * 255).astype(np.uint8)
    # Detect link poses from ArUco markers

    link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, model, data, robot_config)
    puck_pose = detect_and_position_puck(rgb, model, data, puck_config, cam_K, camera_pose_world, corners_cache, visualize=0)[0]
    configuration_joints_from_image = estimate_robot_state( model, data, robot_config, link_poses, ik_iterations=15)

    # Update simulation with optimized joint state
    data.qpos[:] = data.ctrl[:] = configuration_joints_from_image.q

    # Set puck mocap position to detected pose
    data.mocap_pos[model.body_mocapid[model.body("grabbable_puck").id]] = puck_pose[:3, 3]
    data.mocap_quat[model.body_mocapid[model.body("grabbable_puck").id]] = R.from_matrix(puck_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    
else:
    link_poses = get_link_poses_from_robot(robot_config, model, data)
    position_exoskeleton_meshes(robot_config, model, data, link_poses)
    
    # Position puck ArUco marker based on current puck pose in simulation
    aruco_pose = position_puck_aruco_from_mocap(model, data, puck_config)
    
    model.site_pos[model.site("end_effector").id] = np.array([0.015, -0.1, 0])
    mujoco.mj_forward(model, data)


data.qpos[-1]=data.ctrl[-1]=1.75
mujoco.mj_forward(model, data)

    
if 0: # show IK to grasp puck before rendering
    for _ in range(20): track_aruco_with_ik(model, data, puck_config, configuration, rate.dt)
    position_exoskeleton_meshes(robot_config, model, data, get_link_poses_from_robot(robot_config, model, data))
    mujoco.mj_forward(model, data)

# Render from estimated camera pose
if not args.no_render and not args.just_sim_state:
    rendered = render_from_camera_pose(model, data, camera_pose_world, cam_K, *rgb.shape[:2])
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img in zip(axes, [rgb, rendered, (rgb * 0.5 + rendered * 0.5).astype(np.uint8)]): ax.imshow(img);ax.axis('off')
    plt.tight_layout();plt.show()
# Launch interactive viewer
else:
   
    print("\nLaunching interactive viewer...")
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    while viewer.is_running():
        
        if 1:
            track_aruco_with_ik(model, data, puck_config, configuration, rate.dt)
            position_exoskeleton_meshes(robot_config, model, data, get_link_poses_from_robot(robot_config, model, data))
        rate.sleep()

        mujoco.mj_step(model, data)
            
        viewer.sync()