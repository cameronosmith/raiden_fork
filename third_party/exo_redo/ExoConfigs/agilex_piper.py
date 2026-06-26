"""Agile X Piper arm with exoskeleton only on the base (no exos on links).

Single ArUco board on the robot base for camera/pose estimation; arm links
have no exo meshes or markers. Add your Piper MuJoCo model under robot_models/agilex_piper/.
"""
import numpy as np
from cv2 import aruco

from .exoskeleton import ExoskeletonConfig, LinkConfig

PIPER_MODEL_DIR = "robot_models/agilex_piper"
SO100_MODEL_DIR = "robot_models/so100_model"
BOARD_IMG_DIR = f"{SO100_MODEL_DIR}/../board_imgs"

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
n_aruco_row, n_aruco_col = 6, 6
x = (4.75 * 2) / ((n_aruco_row + (n_aruco_row - 1) / 10) * 100)
board_ids_base = np.arange(136, 172, dtype=np.int32).reshape(-1, 1)
link_boards = {
    "larger_base": aruco.GridBoard(
        size=(n_aruco_row, n_aruco_col),
        markerLength=x,
        markerSeparation=x / 10,
        dictionary=aruco_dict,
        ids=board_ids_base,
    ),
}


class AgileXPiperBaseOnlyConfig(ExoskeletonConfig):
    """Piper arm: only the base has an exoskeleton (ArUco board); no exos on links."""

    name = "AgileX_Piper_BaseOnly"
    base_xml_path = f"{PIPER_MODEL_DIR}/piper_exo.xml"
    background_xml_path = f"{SO100_MODEL_DIR}/background.xml"
    compiler_meshdir = f"{PIPER_MODEL_DIR}/assets/"

    aruco_boards = {"larger_base": f"{BOARD_IMG_DIR}/larger_base.png"}
    aruco_board_objects = link_boards

    # Only the base link has an exo; arm links have no markers.
    # Use "larger_base" so detect_link_poses uses it for camera pose.
    # pybullet_name must match the base body in your Piper MuJoCo XML (e.g. base_link).
    links = {
        "larger_base": LinkConfig(
            mujoco_name="base_link",
            pybullet_name="base_link",
            robot_mesh_path="base_link.stl",
            exo_mesh_path="../../so100_blender_testings/piper_base_camboard.stl",
            aruco_offset_pos=np.array([-3.74,-94.42,4.33]),  
            aruco_offset_rot=np.array([0, 0, 0]),
            aruco_board_name="larger_base",
            board_length=0.095,
        ),
    }


AGILEX_PIPER_BASE_ONLY_CONFIG = AgileXPiperBaseOnlyConfig()
