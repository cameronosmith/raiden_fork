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

from ExoConfigs.panda_exo import PANDA_BASE_ONLY_CONFIG as robot_config
from exo_utils import estimate_robot_state, detect_and_set_link_poses, render_from_camera_pose
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
args = parser.parse_args()

# Configuration
camera_device = args.camera

print(f"Using exoskeleton config: {robot_config.name})")
print(f"Initializing camera device {camera_device}...")

# Load model from config
model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)

# Initialize camera
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened(): raise RuntimeError(f"Failed to open camera device {camera_device}")

# Get first frame to determine resolution
for _ in range(10): ret, frame = cap.read()
if not ret: raise RuntimeError("Failed to read from camera")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
height, width = rgb.shape[:2]
print(f"Camera resolution: {width}x{height}")

# Initialize renderer
renderer = Renderer(model, height=height, width=width)

cam_K=None if 0 else np.array([[1.58847596e+03,0.00000000e+00, 9.59500000e+02], [0.00000000e+00, 1.58847596e+03, 5.39500000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
import cv2
cap = cv2.VideoCapture(0)
median_K=[]
while True:
    ret, frame = cap.read()
    print("reading frame")
    if not ret: print ("Failed to read frame from camera");continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try: link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, model, data, robot_config,cam_K=cam_K)
    except Exception as e: 
        print(e)
        print("pose est error");

        cv2.imshow("display", rgb[...,::-1])
        waitkey=cv2.waitKey(1)& 0xFF
        if waitkey==ord('q'): break
        continue
    #median_K.append(cam_K)
    #cam_K=np.median(median_K, axis=0)
    #print(cam_K)
    rendered = render_from_camera_pose(model, data, camera_pose_world, cam_K, *rgb.shape[:2])
    overlay = (rgb.astype(float) * 0.5 + rendered.astype(float)  * 0.5).astype(np.uint8)
    display = np.hstack([corners_vis, rendered, overlay])

    cv2.imshow("display", display[...,::-1])
    waitkey=cv2.waitKey(1)& 0xFF
    if waitkey==ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Done!")