"""Live camera-based robot state estimation with rendering.

This demo continuously:
1. Captures frames from camera
2. Detects ArUco markers
3. Estimates robot joint configuration
4. Renders and displays the result
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import cv2,pickle
import numpy as np
from mujoco.renderer import Renderer

from ExoConfigs import EXOSKELETON_CONFIGS
from ExoConfigs.so100_adhesive import SO100AdhesiveConfig
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, render_from_camera_pose, get_link_poses_from_robot
from robot_models.so100_controller import Arm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exo", type=str, default="so100_adhesive", choices=list(EXOSKELETON_CONFIGS.keys()), help="Exoskeleton configuration to use")
parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
parser.add_argument("--use_robot_state",action="store_true") # use raw robot motors as init
args = parser.parse_args()

# Configuration
robot_config = EXOSKELETON_CONFIGS[args.exo]
camera_device = args.camera

print(f"Using exoskeleton config: {args.exo} ({robot_config.name})")
print(f"Initializing camera device {camera_device}...")

if args.use_robot_state: self=Arm( pickle.load(open("robot_models/arm_offsets/middleservo_calib_redo_fromimg.pkl", 'rb')) )

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

# Initialize renderer
renderer = Renderer(model, height=height, width=width)

# Set initial configuration
data.qpos[:] = data.ctrl[:] = np.array([0, -1.57, 1.57, 1.57, -1.57, 0])
mujoco.mj_forward(model, data)

cam_K=None#np.array([[1.19087964e+03, 0.00000000e+00, 9.59500000e+02], [0.00000000e+00, 1.19087964e+03, 5.39500000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    print("reading frame")
    if not ret: print ("Failed to read frame from camera");continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    try: link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, model, data, robot_config,cam_K=cam_K)
    except: 
        print("pose est error");

        cv2.imshow("display", rgb[...,::-1])
        waitkey=cv2.waitKey(1)& 0xFF
        if waitkey==ord('q'): break
        continue
    configuration = estimate_robot_state( model, data, robot_config, link_poses, ik_iterations=15)
    data.qpos[:] = data.ctrl[:] = configuration.q
    mujoco.mj_forward(model, data)
    rendered = render_from_camera_pose(model, data, camera_pose_world, cam_K, *rgb.shape[:2])
    overlay = (rgb.astype(float) * 0.5 + rendered.astype(float)  * 0.5).astype(np.uint8)
    display = np.hstack([corners_vis, rendered, overlay])

    cv2.imshow("display", display[...,::-1])
    waitkey=cv2.waitKey(1)& 0xFF
    if waitkey==ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Done!")