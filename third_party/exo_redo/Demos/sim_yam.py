"""View the i2rt YAM arm with base-only exoskeleton in sim."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco

from ExoConfigs.yam_exo import YAM_BASE_ONLY_CONFIG
from exo_utils import position_exoskeleton_meshes, get_link_poses_from_robot

robot_config = YAM_BASE_ONLY_CONFIG
if hasattr(robot_config, "exo_link_alpha"):
    robot_config.exo_link_alpha = 1
if hasattr(robot_config, "exo_alpha"):
    robot_config.exo_alpha = 1
robot_config.xml = robot_config._generate_xml()

print(f"Model: {robot_config.base_xml_path}")

model = mujoco.MjModel.from_xml_string(robot_config.xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("\nLaunching interactive viewer...")
viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
while viewer.is_running():
    position_exoskeleton_meshes(
        robot_config, model, data, get_link_poses_from_robot(robot_config, model, data)
    )
    # Keep a static visualization like sim_panda startup; stepping can make joints sag.
    mujoco.mj_forward(model, data)
    viewer.sync()
