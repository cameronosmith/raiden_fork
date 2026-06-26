"""Panda arm with exoskeleton only on the base (no exos on links).

Single ArUco board on the robot base for camera/pose estimation; arm links
have no exo meshes or markers.
"""
import numpy as np
from cv2 import aruco

from .exoskeleton import ExoskeletonConfig, LinkConfig

PANDA_MODEL_DIR = "/Users/cameronsmith/Projects/robotics_testing/exo_redo/robot_models/franka_emika_panda"
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
BOARD_LENGTH_LARGE = n_aruco_row * x + (n_aruco_row - 1) * (x / 10)  # 6x6 board, ~0.095 m

# 3x3 coarse board: half the resolution, twice the marker length (and separation)
n_coarse_row, n_coarse_col = 3, 3
x_coarse = 2 * x  # twice the marker length; separation scales the same
board_ids_coarse = np.arange(217, 217 + 9, dtype=np.int32).reshape(-1, 1)  # 217–225, no conflict
link_boards["larger_coarse_board"] = aruco.GridBoard(
    size=(n_coarse_row, n_coarse_col),
    markerLength=x_coarse,
    markerSeparation=x_coarse / 10,
    dictionary=aruco_dict,
    ids=board_ids_coarse,
)
BOARD_LENGTH_COARSE = n_coarse_row * x_coarse + (n_coarse_row - 1) * (x_coarse / 10)  # ~0.094 m

# 4x4 board: physical size = coarse board edge × (75/50) — exo plane was 50 units, new is 75
EVEN_LARGER_PLANE_RATIO = 75 / 50
BOARD_LENGTH_EVEN_LARGER = BOARD_LENGTH_COARSE * EVEN_LARGER_PLANE_RATIO
n_even_row, n_even_col = 4, 4
_x_even = BOARD_LENGTH_EVEN_LARGER / (n_even_row + (n_even_row - 1) / 10)
board_ids_even_larger = np.arange(226, 226 + n_even_row * n_even_col, dtype=np.int32).reshape(-1, 1)
link_boards["even_larger_board"] = aruco.GridBoard(
    size=(n_even_row, n_even_col),
    markerLength=_x_even,
    markerSeparation=_x_even / 10,
    dictionary=aruco_dict,
    ids=board_ids_even_larger,
)

if 0:  # Generate PNG for larger_coarse_board (set to 1 and run this module to export)
    import cv2
    from pathlib import Path
    board_length_coarse = n_coarse_row * x_coarse + (n_coarse_row - 1) * (x_coarse / 10)
    out_dir = Path(__file__).resolve().parents[1] / "robot_models" / "board_imgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "larger_coarse_board.png"
    img = link_boards["larger_coarse_board"].generateImage((800, 800))
    cv2.imwrite(str(out_path), img)
    try:
        from PIL import Image
        pil_img = Image.open(out_path)
        board_length_in = board_length_coarse / 0.0254
        dpi = int(round(pil_img.width / board_length_in))
        pil_img.save(str(out_path), dpi=(dpi, dpi))
    except ImportError:
        pass
    print(f"Wrote {out_path}")

if 0:  # Generate PNG for even_larger_board (set to 1 and run this module to export)
    import cv2
    from pathlib import Path
    out_dir = Path(__file__).resolve().parents[1] / "robot_models" / "board_imgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "even_larger_board.png"
    img = link_boards["even_larger_board"].generateImage((800, 800))
    cv2.imwrite(str(out_path), img)
    try:
        from PIL import Image
        pil_img = Image.open(out_path)
        board_length_in = BOARD_LENGTH_EVEN_LARGER / 0.0254
        dpi = int(round(pil_img.width / board_length_in))
        pil_img.save(str(out_path), dpi=(dpi, dpi))
    except ImportError:
        pass
    print(f"Wrote {out_path}")


class PandaBaseOnlyConfig(ExoskeletonConfig):
    """Panda arm: only the base has an exoskeleton (ArUco board); no exos on links."""

    name = "Panda_BaseOnly"
    base_xml_path = f"{PANDA_MODEL_DIR}/panda.xml"
    background_xml_path = f"{SO100_MODEL_DIR}/background.xml"
    compiler_meshdir = f"{PANDA_MODEL_DIR}/assets/"

    aruco_boards = {
        "larger_base": f"{BOARD_IMG_DIR}/larger_base.png",
        "larger_coarse_board": f"{BOARD_IMG_DIR}/larger_coarse_board.png",
        "even_larger_board": f"{BOARD_IMG_DIR}/even_larger_board.png",
    }
    # Base link: 4×4 even_larger_board (physical size 75/50 × coarse); PnP uses same board_length
    aruco_board_objects = {
        "larger_base": link_boards["even_larger_board"],
        "larger_coarse_board": link_boards["larger_coarse_board"],
        "even_larger_board": link_boards["even_larger_board"],
    }

    # Only the base link has an exo; arm links have no markers.
    # Use "larger_base" so detect_link_poses uses it for camera pose.
    # pybullet_name MUST be the robot base body in the included XML (panda.xml -> "link0").
    links = {
        "larger_base": LinkConfig(
            mujoco_name="link0_8",
            pybullet_name="link0",
            robot_mesh_path="link0_8.obj",
            exo_mesh_path="../../so100_blender_testings/panda_base_redo_grip_align_larger.stl",
            aruco_offset_pos=np.array([46.6,-152.2,44.0]),  
            aruco_offset_rot=np.array([0, 0, 0]),
            aruco_board_name="even_larger_board",
            board_length=BOARD_LENGTH_EVEN_LARGER,
        ),
    }


PANDA_BASE_ONLY_CONFIG = PandaBaseOnlyConfig()
