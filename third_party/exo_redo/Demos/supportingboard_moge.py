import torch
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append("/Users/cameronsmith/Projects/robotics_testing/random/vggt")
sys.path.append("/Users/cameronsmith/Projects/robotics_testing/random/MoGe")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from demo_viser import *
from demo_utils import preprocess_numpy_images,procrustes_alignment
from moge.model.v2 import MoGeModel # Let's try MoGe-2
import utils3d
import viser

#from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

import mujoco
from mujoco.renderer import Renderer
from ExoConfigs import EXOSKELETON_CONFIGS, PUCK_CONFIG
from ExoConfigs.alignment_board import ALIGNMENT_BOARD_CONFIG
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, render_from_camera_pose, get_link_poses_from_robot, detect_and_position_puck, combine_xmls, detect_and_position_alignment_board

imgpath="/Users/cameronsmith/Downloads/pre_grasp_apple.png"#"scratch/randimgs/helper1.png"
frame1 = cv2.imread(imgpath)[...,[2,1,0]]  # BGR to RGB
imgpath2="/Users/cameronsmith/Downloads/grasp_apple.png"#"scratch/randimgs/helper1.png"
frame2 = cv2.imread(imgpath2)[...,[2,1,0]]  # BGR to RGB
frame1=frame2
imgpath=imgpath2

robot_config = EXOSKELETON_CONFIGS["so100_holemounts"]
mj_model= mujoco.MjModel.from_xml_string(combine_xmls(robot_config.xml, ALIGNMENT_BOARD_CONFIG.get_xml_addition()))
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)
rgb=frame1
if rgb.max() <= 1.0: rgb = (rgb * 255).astype(np.uint8)
if frame2.max() <= 1.0: frame2 = (frame2 * 255).astype(np.uint8)
# Detect link poses from ArUco markers
link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(frame2, mj_model, mj_data, robot_config)

configuration = estimate_robot_state( mj_model, mj_data, robot_config, link_poses, ik_iterations=55)

# Detect and position alignment board, add to obj_img_pts for VGGT alignment
board_result = detect_and_position_alignment_board(rgb, mj_model, mj_data, ALIGNMENT_BOARD_CONFIG, cam_K, camera_pose_world, corners_cache, visualize=False)
if board_result is not None:
    board_pose, board_pts = board_result
    obj_img_pts["alignment_board"] = board_pts
    print(f"Alignment board detected and added to VGGT alignment ({len(board_pts[1])} points)")
else:
    print("Warning: Alignment board not detected")

obj_img_pts={"larger_base":obj_img_pts["larger_base"],"alignment_board":obj_img_pts["alignment_board"]}

mj_data.qpos[:] = mj_data.ctrl[:] = configuration.q
mujoco.mj_forward(mj_model, mj_data)
rendered = render_from_camera_pose(mj_model, mj_data, camera_pose_world, cam_K, *rgb.shape[:2])

if 0:
    # Display render
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img in zip(axes, [corners_vis, rendered, (frame2 * 0.5 + rendered * 0.5).astype(np.uint8)]): ax.imshow(img);ax.axis('off')
    plt.tight_layout()
    plt.show()
    zz


device = torch.device("mps")

# Load the model from huggingface hub (or load from local).
# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer 
if 1: 
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)                             
    output = model.infer(input_image)
    torch.save(output, "output.pt")
else:
    output = torch.load("output.pt")

points = output["points"].cpu().numpy()  # (H, W, 3)
mask=output["mask"].cpu().numpy()&~utils3d.np.depth_map_edge(points[:,:,2], rtol=0.04)

# Flatten and filter valid points
points_flat = points.reshape(-1, 3)
mask_flat = mask.reshape(-1)
valid_points = points_flat[mask_flat > 0.5]

# Get colors from input image
colors = input_image.cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
colors_flat = colors.reshape(-1, 3)
valid_colors = colors_flat[mask_flat > 0.5]

# Create viser server and add pointcloud
server = viser.ViserServer()
#server.scene.add_point_cloud( name="/moge_pointcloud", points=valid_points, colors=valid_colors, point_size=0.001,)

# Sample ArUco 3D points from MOGE pointmap (no coordinate transforms needed!)
moge_aruco_corners = []
aruco_corners_robot_frame = []
aruco_img_coords = []

for obj_img_pts_, img_pts in obj_img_pts.values():
    # Get 3D points in robot frame
    aruco_3d = (np.linalg.inv(camera_pose_world) @ np.hstack([obj_img_pts_, np.ones((obj_img_pts_.shape[0], 1))]).T).T[:, :3]
    aruco_corners_robot_frame.extend(aruco_3d)
    
    # Sample from MOGE pointmap - direct pixel coordinates, no transforms!
    for pt in img_pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < points.shape[0] and 0 <= x < points.shape[1]:
            moge_aruco_corners.append(points[y, x])
            aruco_img_coords.append([x, y])

aruco_corners_robot_frame = np.array(aruco_corners_robot_frame)
moge_aruco_corners = np.array(moge_aruco_corners)
aruco_img_coords = np.array(aruco_img_coords)

print(f"Sampled {len(moge_aruco_corners)} ArUco corners from MOGE pointmap")

# Procrustes alignment: align MOGE pointcloud to robot frame
T_procrustes, scale, rotation, translation = procrustes_alignment(aruco_corners_robot_frame, moge_aruco_corners)
points_homogeneous = np.hstack([valid_points, np.ones((len(valid_points), 1))])
moge_aligned = (T_procrustes @ points_homogeneous.T).T[:, :3]

# Add aligned pointcloud to viser
server.scene.add_point_cloud(
    name="/moge_aligned",
    points=moge_aligned,
    colors=valid_colors,
    point_size=0.001,
)

# Add ArUco correspondence points
server.scene.add_point_cloud( name="/aruco_robot", points=aruco_corners_robot_frame, colors=np.tile([255, 0, 0], (len(aruco_corners_robot_frame), 1)), point_size=0.003,)
#server.scene.add_point_cloud( name="/aruco_moge", points=moge_aruco_corners, colors=np.tile([0, 255, 0], (len(moge_aruco_corners), 1)), point_size=0.003,)

# Visualize sampling locations on the image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image with ArUco detections
axes[0].imshow(rgb)
for coords in aruco_img_coords:
    axes[0].plot(coords[0], coords[1], 'ro', markersize=5, alpha=0.7)
axes[0].set_title(f"MOGE Input Image with ArUco Samples\n{rgb.shape[1]}x{rgb.shape[0]}")
axes[0].axis('off')

# Show depth map with ArUco samples
if 0:
    depth_vis = output["depth"].cpu().numpy()
    depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
    axes[1].imshow(depth_vis, cmap='viridis')
    for coords in aruco_img_coords:
        axes[1].plot(coords[0], coords[1], 'r+', markersize=8, markeredgewidth=2)
    axes[1].set_title("MOGE Depth Map with Samples")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("scratch/moge_aruco_sampling.png", dpi=150, bbox_inches='tight')
    print("Saved sampling visualization to scratch/moge_aruco_sampling.png")
    plt.show()

# Load URDF robot arm
from viser.extras import ViserUrdf
import yourdfpy
urdf_path = "/Users/cameronsmith/Projects/robotics_testing/calibration_testing/so_100_arm/urdf/so_100_arm.urdf"
urdf = yourdfpy.URDF.load(urdf_path)
viser_urdf = ViserUrdf(
    server,
    urdf_or_path=urdf,
    load_meshes=True,
    load_collision_meshes=False,
    collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
)
mujoco_so100_offset = np.array([0, -1.57, 1.57, 1.57, -1.57, 0])
viser_urdf.update_cfg(np.array(mj_data.qpos - mujoco_so100_offset))
print("Loaded robot URDF into viser")

print("Viser server running at http://localhost:8080")
print("Press Ctrl+C to exit")
import pdb; pdb.set_trace()