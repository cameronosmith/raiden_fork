import torch
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append("/Users/cameronsmith/Projects/robotics_testing/random/vggt")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from demo_viser import *
from demo_utils import preprocess_numpy_images,procrustes_alignment
import viser

from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

import mujoco
from mujoco.renderer import Renderer
from ExoConfigs import EXOSKELETON_CONFIGS, PUCK_CONFIG
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, render_from_camera_pose, get_link_poses_from_robot, detect_and_position_puck, combine_xmls


#frame1 = cv2.imread("/Users/cameronsmith/Downloads/IMG_9358.png")[...,[2,1,0]]  # BGR to RGB
#frame2 = cv2.imread("/Users/cameronsmith/Downloads/IMG_9359.png")[...,[2,1,0]]  # BGR to RGB
frame1 = cv2.imread("../redo_mujoco_calibration/random/tmpimgs/kitchen_logi1_black.png")[...,[2,1,0]]  # BGR to RGB
frame2 = cv2.imread("../redo_mujoco_calibration/random/tmpimgs/kitchen_logi2_black.png")[...,[2,1,0]]  # BGR to RGB

robot_config = EXOSKELETON_CONFIGS["so100_holemounts"]
#mj_model = mujoco.MjModel.from_xml_string(combine_xmls(robot_config.xml, PUCK_CONFIG.get_xml_addition()))
mj_model = mujoco.MjModel.from_xml_string(robot_config.xml)
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = mj_data.ctrl[:] = np.array([0, -1.57, 1.57, 1.57, -1.57, 0])
mujoco.mj_forward(mj_model, mj_data)
rgb=frame1
if rgb.max() <= 1.0: rgb = (rgb * 255).astype(np.uint8)
# Detect link poses from ArUco markers
link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, mj_model, mj_data, robot_config)
#puck_pose,puck_img_pts = detect_and_position_puck(rgb, mj_model, mj_data, PUCK_CONFIG, cam_K, camera_pose_world, corners_cache, visualize=0)
#obj_img_pts["puck"] = puck_img_pts

configuration = estimate_robot_state( mj_model, mj_data, robot_config, link_poses, ik_iterations=15)
mj_data.qpos[:] = mj_data.ctrl[:] = configuration.q
mujoco.mj_forward(mj_model, mj_data)
rendered = render_from_camera_pose(mj_model, mj_data, camera_pose_world, cam_K, *rgb.shape[:2])

if 0:
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img in zip(axes, [corners_vis, rendered, (rgb * 0.5 + rendered * 0.5).astype(np.uint8)]): ax.imshow(img);ax.axis('off')
    plt.tight_layout()
    plt.show()
    zz

"""
Main function for the VGGT demo with viser for 3D visualization.
Hardcoded to use frame1 and frame2 from twoviews.npy
"""
device = "mps" 
torch.set_grad_enabled(False)
images = preprocess_numpy_images([frame1,frame2], mode="crop").to(device)
dtype = torch.float16
images = images.to(dtype)
if 1:
    print("loading model")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval()
    model = model.to(device).to(torch.float16)
    print("doing prediction")
    with torch.amp.autocast(device_type="mps", dtype=torch.float16): predictions = model(images)
    print("done prediction")
    torch.save(predictions,"../random/vggt_predictions.pt")
else: predictions=torch.load("../random/vggt_predictions.pt")

print("Processing model outputs...")
predictions["extrinsic"], predictions["intrinsic"] = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
for key in predictions.keys():
    if isinstance(predictions[key], torch.Tensor): predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
print("Starting simple viser visualization...")

# Use depth-based points
images = predictions["images"]  # (S, 3, H, W)
depth_map = predictions["depth"]  # (S, H, W, 1)
world_points = predictions["world_points"]

depth_map = predictions["depth"]  # (S, H, W, 1)
depth_conf = predictions["depth_conf"]  # (S, H, W)
extrinsics_cam = predictions["extrinsic"]
intrinsics_cam = predictions["intrinsic"]
world_points = torch.from_numpy(unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)).float().numpy()
conf = depth_conf

# Convert images from (S, 3, H, W) to (S, H, W, 3) for colors

# Start viser server
server = viser.ViserServer(host="0.0.0.0", port=8080)
print(f"Viser server started on port 8080")

S, H, W, _ = world_points.shape
points = world_points.reshape(-1, 3)
colors_flat = (predictions["images"].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
if 0: conf_flat = predictions["depth_conf"].reshape(-1)
else: conf_flat = predictions["world_points_conf"].reshape(-1)
conf_mask = (conf_flat >= np.percentile(conf_flat, 20.0)) & (conf_flat > 1e-5)
#point_cloud = server.scene.add_point_cloud( name="simple_pcd", points=points[conf_mask], colors=colors_flat[conf_mask], point_size=0.0005, point_shape="circle",)

# Load URDF robot arm
urdf_path = "/Users/cameronsmith/Projects/robotics_testing/calibration_testing/so_100_arm/urdf/so_100_arm.urdf"
#print(f"Loading URDF from: {urdf_path}")

# Import required modules
from viser.extras import ViserUrdf
import yourdfpy
urdf = yourdfpy.URDF.load(urdf_path)
viser_urdf = ViserUrdf( server, urdf_or_path=urdf, load_meshes=True, load_collision_meshes=False, collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),)
mujoco_so100_offset = np.array([0, -1.57, 1.57, 1.57, -1.57, 0])
viser_urdf.update_cfg(np.array(mj_data.qpos-mujoco_so100_offset))

# Get aruco corners in robot frame
aruco_corners_all = []
img_pts_norm_all = []
for obj_img_pts, img_pts in obj_img_pts.values():
    aruco_corners_all.extend((np.linalg.inv(camera_pose_world)@ np.hstack([obj_img_pts, np.ones((obj_img_pts.shape[0], 1))]).T).T[:, :3])
    img_pts_norm_all.extend(img_pts / np.array([rgb.shape[1]-1, rgb.shape[0]-1]))

# Sample vggt aruco corners in vggt frame
vggt_corners_all = torch.nn.functional.grid_sample(torch.from_numpy(world_points)[[0]].permute(0,3,1,2), torch.from_numpy(np.stack(img_pts_norm_all))[None,None].float()*2-1, mode='nearest', padding_mode='border', align_corners=False).squeeze().T.numpy()
vggt_corners_conf_all = torch.nn.functional.grid_sample(torch.from_numpy(predictions["world_points_conf"][...,None])[[0]].permute(0,3,1,2), torch.from_numpy(np.stack(img_pts_norm_all))[None,None].float()*2-1, mode='nearest', padding_mode='border', align_corners=False).squeeze().T.numpy()

# Filter arucos with vggt confidence
vggt_aruco_conf = vggt_corners_conf_all > np.percentile(vggt_corners_conf_all, 5.0)
vggt_corners_all = vggt_corners_all[vggt_aruco_conf]
aruco_corners_all = np.array(aruco_corners_all)[vggt_aruco_conf]

# Ensure we have the same number of points for alignment
T_procrustes, scale, rotation, translation = procrustes_alignment(aruco_corners_all, vggt_corners_all)
points_homogeneous = np.hstack([world_points.reshape(-1, 3), np.ones((world_points.reshape(-1, 3).shape[0], 1))])
vggt_in_robot_frame = (T_procrustes @ points_homogeneous.T).T[:,:3]
server.scene.add_point_cloud( name="robot_frame_pcd", points=vggt_in_robot_frame, colors=colors_flat, point_size=0.001, point_shape="circle",)
print(f"Added {len(vggt_in_robot_frame)} VGGT corner points to viser (in robot coordinate frame)")

#aruco_point_cloud = server.scene.add_point_cloud( name="aruco_points", points=aruco_corners_all, colors=np.tile([255, 0, 0], (len(aruco_corners_all), 1)), point_size=0.002, point_shape="circle",)
#vggt_aruco_point_cloud = server.scene.add_point_cloud( name="vggt_aruco_points", points=np.array(vggt_corners_all), colors=np.tile([0,255,  0], (len(vggt_corners_all), 1)), point_size=0.004, point_shape="circle",)
print(f"Added {len(aruco_corners_all)} ArUco corner points to viser (in robot coordinate frame)")

print("Viser visualization ready. Open http://localhost:8080 in your browser")

# Keep server running
while True:
    import time
    time.sleep(0.01)
