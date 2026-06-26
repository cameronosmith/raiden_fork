"""Large 8x8 ArUco board for pointcloud alignment configuration."""
import numpy as np
import cv2
from cv2 import aruco
from .exoskeleton import BLENDER_STL_DIR, SO100_MODEL_DIR, BOARD_IMG_DIR

# Large 8x8 ArUco board for alignment
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
n_aruco_row, n_aruco_col = 6, 6
x = (4.75 * 2) / ((n_aruco_row + (n_aruco_row - 1) / 10) * 100)
board_ids_alignment = np.arange(181, 181 + 36, dtype=np.int32).reshape(-1, 1)  # IDs 181-216 (36 markers)
alignment_board = aruco.GridBoard( size=(n_aruco_row, n_aruco_col), markerLength=x, markerSeparation=x/10, dictionary=aruco_dict, ids=board_ids_alignment)
board_length=0.095

if 0: # Generate image for alignment board
  from PIL import Image
  marker = alignment_board.generateImage((800, 800))
  marker_path = "robot_models/board_imgs/alignment_board.png"
  cv2.imwrite(marker_path, marker)
  img = Image.open(marker_path)
  width_px, height_px = img.size
  board_length_in = board_length / 0.0254 
  dpi = int(round(width_px / board_length_in))
  img.save(marker_path, dpi=(dpi, dpi))

class AlignmentBoardConfig:
    """Configuration for large 8x8 ArUco alignment board."""
    
    name = "Alignment_Board"
    
    # ArUco configuration
    aruco_board = alignment_board
    aruco_offset_pos = np.array([0, 0, 0])  # Center of the plane
    aruco_offset_rot = np.array([0, 0, 0])  # Flat on table (rotate to face up)
    board_length = board_length
    
    @classmethod
    def get_xml_addition(cls) -> str:
        """Get XML snippet to add alignment board to scene."""
        return f"""
          <asset>
            <mesh name="alignment_aruco_plane" file="{BLENDER_STL_DIR}/plane.obj" scale="0.3166666 0.3166666 .001" inertia="shell"/>
            <texture name="alignment_aruco_tex" type="2d" file="{BOARD_IMG_DIR}/alignment_board.png"/>
            <material name="alignment_aruco_mat" texture="alignment_aruco_tex" rgba="1 1 1 1"/>
          </asset>

          <worldbody>
            <body mocap="true" name="alignment_board" pos="0 -0.3 0.001" quat="1 0 0 0">
              <geom name="alignment_board_geom" type="mesh" mesh="alignment_aruco_plane" contype="0" conaffinity="0" material="alignment_aruco_mat"/>
            </body>
          </worldbody>
        """


# Convenience instance
ALIGNMENT_BOARD_CONFIG = AlignmentBoardConfig()

