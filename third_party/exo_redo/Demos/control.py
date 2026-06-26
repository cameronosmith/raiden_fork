"""Live camera-based robot state estimation with puck grasping.

This demo continuously:
1. Captures frames from camera
2. Detects ArUco markers on robot and puck
3. Estimates robot joint configuration
4. Runs IK to grasp the puck
5. Displays the result
"""
import sys
import os
from turtle import done
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import cv2,pickle
import numpy as np

# Suppress OpenCV warnings
cv2.setLogLevel(0)
from mujoco.renderer import Renderer
import mink
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from ExoConfigs import EXOSKELETON_CONFIGS
from ExoConfigs.puck import PUCK_CONFIG
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, detect_and_position_puck, render_from_camera_pose, combine_xmls, get_link_poses_from_robot, position_puck_aruco_from_mocap, track_aruco_with_ik
from robot_models.so100_controller import Arm,rest_pos,targ_joint_state_to_match,sensor_to_offset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exo", type=str, default="so100_holemounts", choices=list(EXOSKELETON_CONFIGS.keys()), help="Exoskeleton configuration to use")
parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
args = parser.parse_args()

# Configuration
robot_config = EXOSKELETON_CONFIGS[args.exo]
camera_device = 0  # Change to match your camera

self=Arm( pickle.load(open("robot_models/arm_offsets/so300_home_adhesive.pkl", 'rb')) )

cam_K=None#np.array([[1.19087964e+03, 0.00000000e+00, 9.59500000e+02], [0.00000000e+00, 1.19087964e+03, 5.39500000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Load model from config
model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)

# Initialize camera
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened(): raise RuntimeError(f"Failed to open camera device {camera_device}")

# Get first frame to determine resolution
ret, frame = cap.read()
if not ret: raise RuntimeError("Failed to read from camera")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
height, width = rgb.shape[:2]
print(f"Camera resolution: {width}x{height}")

i=0
target_pos = targ_joint_state_to_match if 1 else rest_pos
already_written_target,already_written_delta,already_written_sensor_offset=False,0,False
while True:
    i+=1
    print(i)

    self.data.qpos[:] = self.data.ctrl[:] = self.get_pos(); 

    # Capture frame
    ret, frame = cap.read()
    if not ret: print("Failed to read frame from camera");continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect robot and puck ArUco markers

    try: link_poses, camera_pose_world, cam_K, corners_cache, corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, model, data, robot_config,cam_K=cam_K)
    except: print("pose est error");continue
    configuration_joints_from_image = estimate_robot_state( model, data, robot_config, link_poses, ik_iterations=15)
    data.qpos[:] = data.ctrl[:] = target_pos #configuration_joints_from_image.q
    mujoco.mj_forward(model, data)

    done_moving = not any([self.sts.ReadMoving(i+1)[0] for i in range(6)])

    print(target_pos-configuration_joints_from_image.q)
    delta_pos = target_pos-(configuration_joints_from_image.q-target_pos)
    if done_moving and not already_written_target: self.write_pos(target_pos);already_written_target=True
    elif done_moving and already_written_delta==0: print("moving delta",delta_pos);self.write_pos(delta_pos,slow=True);already_written_delta+=1
    elif done_moving and not already_written_sensor_offset:print("updating sensor offset");self.calib["sensor_offset"] = sensor_to_offset(configuration_joints_from_image.q,self.calib["signs"],self.get_pos(raw=True));already_written_sensor_offset=True

    rendered = render_from_camera_pose(model, data, camera_pose_world, cam_K, *rgb.shape[:2])
    
    # Create overlay
    display = np.hstack([corners_vis, rendered, (rgb.astype(float) * 0.5 + rendered.astype(float) * 0.5).astype(np.uint8)])
    cv2.imshow('Live Puck Grasping', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
    
    # Check for quit
    waitkey = cv2.waitKey(1) & 0xFF
    if waitkey == ord('q'): break
    elif waitkey == ord('s'): plt.imsave("../redo_mujoco_calibration/random/tmpimgs/00001.png", rgb);print("saved img")

self.write_pos(rest_pos,slow=False)
while any([self.sts.ReadMoving(i+1)[0] for i in range(6)]):pass

self.emergency_stop()
cap.release()
cv2.destroyAllWindows()
    
print("Done!")