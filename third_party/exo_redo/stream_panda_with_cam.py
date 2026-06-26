#!/usr/bin/env python3
"""Stream Panda joint states and visualize them in MuJoCo.

Combines stream_panda.py (socket client for joint states) with sim_panda.py
(MuJoCo viewer). At each step, the sim's joint state is set from the latest
streamed positions.
"""
import socket
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mujoco
import numpy as np

from ExoConfigs.panda_exo import PANDA_BASE_ONLY_CONFIG
from exo_utils import position_exoskeleton_meshes, get_link_poses_from_robot
from exo_utils import estimate_robot_state, detect_and_set_link_poses, render_from_camera_pose

if len(sys.argv) < 3:
    print("Usage: python3 stream_panda_with_vis.py <host> <port>")
    print("Example: python3 stream_panda_with_vis.py 0.tcp.ngrok.io 12345")
    sys.exit(1)

host = sys.argv[1]
port = int(sys.argv[2])

robot_config = PANDA_BASE_ONLY_CONFIG
if hasattr(robot_config, "exo_link_alpha"):
    robot_config.exo_link_alpha = 1

print(f"Model: {robot_config.base_xml_path}")
model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)

# Panda: 7 arm joints + gripper. Gripper is one ctrl [0, 255] and two qpos (finger slides 0–0.04 m).
N_ARM_JOINTS = 7
GRIPPER_CTRL_RANGE = (0, 255)  # actuator8 ctrlrange
GRIPPER_POS_MAX = 0.04         # finger joint range 0–0.04 m (each finger)

n_arm = min(N_ARM_JOINTS, data.qpos.size)
# Initial pose until first message
latest_positions = np.zeros(n_arm)
# Gripper width: 0=closed, 1=open (or use gripper_width_percent 0–100 / raw gripper 0–255 from stream)
latest_gripper_width = 1.0  # default open
has_gripper = data.ctrl.size > N_ARM_JOINTS and data.qpos.size >= N_ARM_JOINTS + 2

print(f"Connecting to {host}:{port}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))
sock.settimeout(0.01)
print("Connected! Streaming joint states into MuJoCo viewer...\n")

buffer = ""

# Initialize camera
cap = cv2.VideoCapture(0)

# Get first frame to determine resolution
for _ in range(10): ret, frame = cap.read()
if not ret: raise RuntimeError("Failed to read from camera")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
height, width = rgb.shape[:2]
print(f"Camera resolution: {width}x{height}")

# Initialize renderer
from mujoco.renderer import Renderer
renderer = Renderer(model, height=height, width=width)

cam_K=None if 1 else np.array([[1.58847596e+03,0.00000000e+00, 9.59500000e+02], [0.00000000e+00, 1.58847596e+03, 5.39500000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
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
    
    # After each camera frame, drain the socket and keep only the latest state.
    # We loop recv() until it times out, overwriting latest_positions each time.
    while True:
        try:
            data_recv = sock.recv(4096).decode("utf-8")
            if not data_recv:
                # Remote closed connection
                sock.close()
                raise SystemExit("Joint stream closed.")
            buffer += data_recv
        except socket.timeout:
            # No more data pending for this frame
            break

    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        if line.strip():
            try:
                msg = json.loads(line)
                pos = msg.get("positions", [])
                latest_positions = np.array(pos[:n_arm], dtype=np.float64)
                latest_gripper_width = float(msg["gripper_width_percent"]) # ranges between [0,1]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # Drive sim from latest streamed positions (kinematics only, no physics step)
    data.qpos[:n_arm] =  latest_positions
    # Single gripper width -> ctrl[7] in [0,255] and both finger qpos in [0, 0.04] m

    # NOTE need gripper separate in model don't have it in mujoco here may not need it tbh
    #g_ctrl = latest_gripper_width * GRIPPER_CTRL_RANGE[1]
    #g_pos_m = latest_gripper_width * GRIPPER_POS_MAX
    #print(g_ctrl, g_pos_m)
    ##data.ctrl[N_ARM_JOINTS] = g_ctrl
    #data.qpos[N_ARM_JOINTS] = data.qpos[N_ARM_JOINTS + 1] = g_pos_m

    for _ in range(10): mujoco.mj_forward(model, data)

    data.qpos[0]=0

    print(data.qpos)
    rendered = render_from_camera_pose(model, data, camera_pose_world, cam_K, *rgb.shape[:2])
    overlay = (rgb.astype(float) * 0.5 + rendered.astype(float)  * 0.5).astype(np.uint8)
    display = np.hstack([corners_vis, rendered, overlay])

    cv2.imshow("display", display[...,::-1])
    waitkey=cv2.waitKey(1)& 0xFF
    if waitkey==ord('q'): break





sock.close()
