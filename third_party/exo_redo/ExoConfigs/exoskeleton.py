"""Base exoskeleton configuration framework."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation as R

BLENDER_STL_DIR = "../../so100_blender_testings"
SO100_MODEL_DIR = "robot_models/so100_model"
BOARD_IMG_DIR = f"{SO100_MODEL_DIR}/../board_imgs"


@dataclass
class LinkConfig:
    """Configuration for a single robot link with ArUco marker."""
    mujoco_name: str          # Name in MuJoCo XML
    pybullet_name: str        # Name for PyBullet/IK solver
    robot_mesh_path: str      # Path to robot's visual mesh
    exo_mesh_path: str        # Path to exoskeleton/mount mesh
    aruco_offset_pos: np.ndarray  # [x, y, z] offset from link to ArUco (mm)
    aruco_offset_rot: np.ndarray  # [rx, ry, rz] euler angles (radians)
    aruco_board_name: str     # Name of ArUco board pattern
    board_length: float       # Physical size of board (meters)
    # Optional local transform for the red robot visual overlay geom.
    # Use this when the source robot mesh is attached with a local geom pose in base XML.
    robot_mesh_pos: Optional[np.ndarray] = None   # meters
    robot_mesh_quat: Optional[np.ndarray] = None  # wxyz
    # Optional offset for green exo mesh (mm / euler xyz radians).
    exo_mesh_offset_pos: Optional[np.ndarray] = None
    exo_mesh_offset_rot: Optional[np.ndarray] = None
    # When True, exo/aruco offsets are measured in the robot visual mesh frame
    # (robot_mesh_pos / robot_mesh_quat), not the raw link body frame.
    offsets_in_robot_mesh_frame: bool = False


def _robot_mesh_frame(cfg: LinkConfig):
    mesh_pos = np.zeros(3) if cfg.robot_mesh_pos is None else np.asarray(cfg.robot_mesh_pos)
    mesh_quat_wxyz = (
        np.array([1.0, 0.0, 0.0, 0.0])
        if cfg.robot_mesh_quat is None
        else np.asarray(cfg.robot_mesh_quat)
    )
    mesh_rot = R.from_quat(mesh_quat_wxyz[[1, 2, 3, 0]])
    return mesh_pos, mesh_quat_wxyz, mesh_rot


def _child_pose_in_link_frame(
    cfg: LinkConfig,
    offset_pos_mm: np.ndarray,
    offset_rot: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return child pos (m) and quat (wxyz) in link-body frame."""
    offset_rot = np.zeros(3) if offset_rot is None else np.asarray(offset_rot)
    if cfg.offsets_in_robot_mesh_frame:
        mesh_pos, _, mesh_rot = _robot_mesh_frame(cfg)
        pos = mesh_pos + mesh_rot.apply(offset_pos_mm / 1000)
        rot = mesh_rot * R.from_euler("xyz", offset_rot)
    else:
        pos = offset_pos_mm / 1000
        rot = R.from_euler("xyz", offset_rot)
    return pos, rot.as_quat()[[3, 0, 1, 2]]


def link_to_aruco_transform(cfg: LinkConfig) -> np.ndarray:
    """4x4 transform from link body frame to ArUco board frame."""
    pos, quat_wxyz = _child_pose_in_link_frame(cfg, cfg.aruco_offset_pos, cfg.aruco_offset_rot)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    return T


class ExoskeletonConfig:
    """Base configuration class for robots with ArUco marker exoskeletons."""
    
    # Override these in subclass
    name: str = "Robot"
    base_xml_path: str = ""
    background_xml_path: str = "robot_models/so100_model/background.xml"
    compiler_meshdir: str = "./"
    
    # ArUco board patterns (override in subclass)
    aruco_boards: Dict[str, str] = {}
    
    # Physical board sizes
    board_length_small: float = 0.045  # 3x3 markers
    board_length_large: float = 0.095  # 6x6 markers
    
    # Link configurations (override in subclass)
    links: Dict[str, LinkConfig] = {}

    exo_alpha: float = .3
    exo_link_alpha: float = 0.8*1
    aruco_alpha: float = 1.0  # Transparency for ArUco planes (0.0 = invisible, 1.0 = opaque)
    
    def __init__(self):
        """Initialize and generate XML."""
        self.xml = self._generate_xml()
    
    def _generate_xml(self) -> str:
        """Generate complete MuJoCo XML with exoskeletons and ArUco planes."""
        
        # Header
        xml = f"""<mujoco>
                    <compiler angle="radian" meshdir="{self.compiler_meshdir}"/>
                    <include file="{self.base_xml_path}"/>
                    <include file="{self.background_xml_path}"/>
                    <visual> <global offheight="4100" offwidth="4100"/> </visual>
                    <asset>"""
        
        # Robot and exoskeleton meshes
        for name, cfg in self.links.items():
            # Robot meshes from SO100 model are already in correct units (meters), no scaling needed
            # Exoskeleton meshes from Blender need .001 scale (mm to m conversion)
            exo_scale = ' scale=".001 .001 .001"' if cfg.exo_mesh_path.endswith('.stl') else ''
            xml += f"""
                    <mesh name="{name}_link_stl" file="{cfg.robot_mesh_path}" inertia="shell"/>
                    <mesh name="{name}_exo_stl" file="{cfg.exo_mesh_path}"{exo_scale} inertia="shell"/>"""
        
        # ArUco plane meshes (even_larger: same base scale as large × 75/50 for Panda even_larger_board)
        xml += f"""<mesh name="aruco_plane_small" file="{BLENDER_STL_DIR}/plane.obj" scale="0.15 0.15 .001" inertia="shell"/>
                   <mesh name="aruco_plane_large" file="{BLENDER_STL_DIR}/plane.obj" scale="0.3166666 0.3166666 .001" inertia="shell"/>
                   <mesh name="aruco_plane_even_larger" file="{BLENDER_STL_DIR}/plane.obj" scale="0.475 0.475 .001" inertia="shell"/>"""
        
        # Textures and materials
        for board_name, img_path in self.aruco_boards.items():
            xml += f"""
                <texture name="{board_name}_tex" type="2d" file="{img_path}"/>
                <material name="{board_name}_mat" texture="{board_name}_tex" rgba="1 1 1 1"/>"""
        xml += """ </asset>
                <worldbody>
                    <camera name="estimated_camera" pos="0 0 0" quat="1 0 0 0" fovy="95"/>
                """
        
        # Mocap bodies for each link
        for name, cfg in self.links.items():
            if cfg.board_length > 0.11:
                plane_size = "even_larger"
            elif cfg.board_length > 0.05:
                plane_size = "large"
            else:
                plane_size = "small"
            link_pos_attr = (
                f' pos="{" ".join(map(str, cfg.robot_mesh_pos))}"'
                if cfg.robot_mesh_pos is not None
                else ""
            )
            link_quat_attr = (
                f' quat="{" ".join(map(str, cfg.robot_mesh_quat))}"'
                if cfg.robot_mesh_quat is not None
                else ""
            )
            exo_pos_attr = ""
            exo_quat_attr = ""
            if cfg.offsets_in_robot_mesh_frame:
                exo_mm = (
                    cfg.exo_mesh_offset_pos
                    if cfg.exo_mesh_offset_pos is not None
                    else np.zeros(3)
                )
                exo_pos, exo_quat_wxyz = _child_pose_in_link_frame(
                    cfg, exo_mm, cfg.exo_mesh_offset_rot
                )
                exo_pos_attr = f' pos="{" ".join(map(str, exo_pos))}"'
                exo_quat_attr = f' quat="{" ".join(map(str, exo_quat_wxyz))}"'
            elif cfg.exo_mesh_offset_pos is not None:
                exo_pos, exo_quat_wxyz = _child_pose_in_link_frame(
                    cfg, cfg.exo_mesh_offset_pos, cfg.exo_mesh_offset_rot
                )
                exo_pos_attr = f' pos="{" ".join(map(str, exo_pos))}"'
                exo_quat_attr = f' quat="{" ".join(map(str, exo_quat_wxyz))}"'
            elif cfg.robot_mesh_pos is not None or cfg.robot_mesh_quat is not None:
                exo_pos_attr = link_pos_attr
                exo_quat_attr = link_quat_attr
            xml += f"""
                    <!-- {name} -->
                    <body mocap="true" name="{name}_link_mesh">
                    <geom type="mesh" mesh="{name}_link_stl" contype="0" conaffinity="0" rgba="1 0 0 {self.exo_link_alpha}"{link_pos_attr}{link_quat_attr} />
                    </body>
                    <body mocap="true" name="{name}_exo_mesh">
                    <geom type="mesh" mesh="{name}_exo_stl" contype="0" conaffinity="0" rgba="0 1 0 {self.exo_alpha}"{exo_pos_attr}{exo_quat_attr} />
                    </body>
                    <body mocap="true" name="{name}_exo_plane">
                    <geom type="mesh" mesh="aruco_plane_{plane_size}" contype="0" conaffinity="0" material="{cfg.aruco_board_name}_mat" rgba="1 1 1 {self.aruco_alpha}" />
                    </body>
                """
        xml += """  </worldbody> </mujoco>"""
        
        return xml

