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

import mujoco
import numpy as np

from ExoConfigs.panda_exo import PANDA_BASE_ONLY_CONFIG
from exo_utils import position_exoskeleton_meshes, get_link_poses_from_robot

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
viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)

while viewer.is_running():
    # Try to read one or more lines from the stream
    try:
        data_recv = sock.recv(4096).decode("utf-8")
        if not data_recv:
            break
        buffer += data_recv
    except socket.timeout:
        pass

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
    data.qpos[:n_arm] = data.ctrl[:n_arm] = latest_positions

    # Single gripper width -> ctrl[7] in [0,255] and both finger qpos in [0, 0.04] m
    g_ctrl = latest_gripper_width * GRIPPER_CTRL_RANGE[1]
    g_pos_m = latest_gripper_width * GRIPPER_POS_MAX
    print(g_ctrl, g_pos_m)
    data.ctrl[N_ARM_JOINTS] = g_ctrl
    data.qpos[N_ARM_JOINTS] = data.qpos[N_ARM_JOINTS + 1] = g_pos_m

    mujoco.mj_forward(model, data)
    position_exoskeleton_meshes(
        robot_config, model, data, get_link_poses_from_robot(robot_config, model, data)
    )
    viewer.sync()

viewer.close()
sock.close()
