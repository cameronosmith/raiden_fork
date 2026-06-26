"""Grabbable puck with ArUco marker configuration."""
import numpy as np
import cv2
from cv2 import aruco
from .exoskeleton import BLENDER_STL_DIR, SO100_MODEL_DIR, BOARD_IMG_DIR

# Puck ArUco board
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
n_aruco_row, n_aruco_col = 3, 3
x = 4.5 / ((n_aruco_row + (n_aruco_row - 1) / 10) * 100)
board_ids_puck = np.arange(172, 181, dtype=np.int32).reshape(-1, 1)  # IDs 172-180
puck_board = aruco.GridBoard(size=(n_aruco_row, n_aruco_col), markerLength=x, markerSeparation=x/10, dictionary=aruco_dict, ids=board_ids_puck)


class PuckConfig:
    """Configuration for grabbable puck with ArUco marker."""
    
    name = "Grabbable_Puck"
    
    # ArUco configuration
    aruco_board = puck_board
    aruco_offset_pos = np.array([1.64, 0, 50.62]) / 1000  # meters
    aruco_offset_rot = np.array([0, np.pi/2, 0])  # radians
    board_length = 0.045
    
    # Grasp offset
    grasp_offset = np.array([9.4, 6.9, 57.8]) / 1000  # meters
    
    @classmethod
    def get_xml_addition(cls) -> str:
        """Get XML snippet to add puck to scene."""
        return f"""
          <asset>
            <mesh name="puck_top" file="{BLENDER_STL_DIR}/grabbable_puck_simpler_cube_top.stl" scale=".001 .001 .001"/>
            <mesh name="puck_bottom" file="{BLENDER_STL_DIR}/grabbable_puck_simpler_cube_bot.stl" scale=".001 .001 .001"/>
            <mesh name="aruco_plane_puck" file="{BLENDER_STL_DIR}/plane.obj" scale="0.15 0.15 .001" inertia="shell"/>
            <texture name="puck_aruco_tex" type="2d" file="{BOARD_IMG_DIR}/grabbable_plane.png"/>
            <material name="puck_aruco_mat" texture="puck_aruco_tex" rgba="1 1 1 1"/>
          </asset>

          <worldbody>
            <body mocap="true" name="grabbable_puck" pos="0 -0.25 0" quat="0 0 0 1">
              <geom name="puck_top_geom" type="mesh" mesh="puck_top" contype="1" conaffinity="1" rgba="0.8 0.2 0.2 0.8"/>
              <geom name="puck_bottom_geom" type="mesh" mesh="puck_bottom" contype="1" conaffinity="1" rgba="0.8 0.2 0.2 0.8"/>
            </body>
            <body mocap="true" name="grabbable_aruco_plane">
              <geom type="mesh" mesh="aruco_plane_puck" contype="0" conaffinity="0" material="puck_aruco_mat"/>
            </body>
          </worldbody>
        """


# Convenience instance
PUCK_CONFIG = PuckConfig()

