"""i2rt YAM arm with exoskeleton only on the base (no exos on links).

Uses a local copied model directory under robot_models/i2rt_yam_fidex so
wrapper tweaks do not modify the upstream menagerie files.
"""
import os
import numpy as np

from .exoskeleton import ExoskeletonConfig, LinkConfig
from .panda_exo import (
    BOARD_IMG_DIR,
    BOARD_LENGTH_COARSE,
    link_boards,
)

_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_this_dir)
YAM_MODEL_DIR = os.path.join(_repo_root, "robot_models", "i2rt_yam_fidex")
SO100_MODEL_DIR = os.path.join(_repo_root, "robot_models", "so100_model")
YAM_ASSETS_DIR = os.path.join(YAM_MODEL_DIR, "assets")
YAM_BASE_MESH = os.path.join(YAM_ASSETS_DIR, "model2.stl")
YAM_EXO_MESH = os.path.join(_repo_root, "so100_blender_testings", "yam_base_board_v2.stl")

class YamBaseOnlyConfig(ExoskeletonConfig):
    """YAM arm: only the base has an exoskeleton (ArUco board); no exos on links."""

    name = "YAM_BaseOnly"
    base_xml_path = f"{YAM_MODEL_DIR}/yam.xml"
    background_xml_path = f"{YAM_MODEL_DIR}/yam_background.xml"
    compiler_meshdir = f"{YAM_ASSETS_DIR}/"

    aruco_boards = {
        "larger_coarse_board": f"{BOARD_IMG_DIR}/larger_coarse_board.png",
    }
    aruco_board_objects = {
        "larger_coarse_board": link_boards["larger_coarse_board"],
    }

    # Base body in yam.xml is "arm"; base visual mesh is model2.stl.
    links = {
        "larger_coarse_board": LinkConfig(
            mujoco_name="arm",
            pybullet_name="arm",
            robot_mesh_path=YAM_BASE_MESH,
            exo_mesh_path=YAM_EXO_MESH,
            # Match base visual geom transform in yam.xml:
            # <geom ... pos="0 0 -0.0006" quat="1 0 0 1" .../>
            robot_mesh_pos=np.array([0.0, 0.0, -0.0006]),
            robot_mesh_quat=np.array([0.70710678, 0.0, 0.0, 0.70710678]),
            # v2 STL geometry is already at [107.7,-35.12,-2.56] in mesh frame; only apply mesh rotation.
            offsets_in_robot_mesh_frame=True,
            aruco_offset_pos=np.array([175.23, -19.6, -4.1]),
            aruco_offset_rot=np.array([0.0, 0.0, 0.0]),
            aruco_board_name="larger_coarse_board",
            board_length=BOARD_LENGTH_COARSE,
        ),
    }


YAM_BASE_ONLY_CONFIG = YamBaseOnlyConfig()
