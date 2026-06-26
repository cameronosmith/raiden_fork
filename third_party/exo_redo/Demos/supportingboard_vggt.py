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
from ExoConfigs.alignment_board import ALIGNMENT_BOARD_CONFIG
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, render_from_camera_pose, get_link_poses_from_robot, detect_and_position_puck, combine_xmls, detect_and_position_alignment_board


#frame1 = cv2.imread("/Users/cameronsmith/Downloads/IMG_9358.png")[...,[2,1,0]]  # BGR to RGB
#frame2 = cv2.imread("/Users/cameronsmith/Downloads/IMG_9359.png")[...,[2,1,0]]  # BGR to RGB
frame1 = cv2.imread("scratch/randimgs/helper1.png")[...,[2,1,0]]  # BGR to RGB
frame2 = cv2.imread("scratch/randimgs/helper2.png")[...,[2,1,0]]  # BGR to RGB

robot_config = EXOSKELETON_CONFIGS["so100_holemounts"]
#mj_model = mujoco.MjModel.from_xml_string(combine_xmls(robot_config.xml, PUCK_CONFIG.get_xml_addition()))
mj_model= mujoco.MjModel.from_xml_string(combine_xmls(robot_config.xml, ALIGNMENT_BOARD_CONFIG.get_xml_addition()))
#mj_model = mujoco.MjModel.from_xml_string(robot_config.xml)
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)
rgb=frame1
if rgb.max() <= 1.0: rgb = (rgb * 255).astype(np.uint8)
# Detect link poses from ArUco markers
link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, mj_model, mj_data, robot_config)
#puck_pose,puck_img_pts = detect_and_position_puck(rgb, mj_model, mj_data, PUCK_CONFIG, cam_K, camera_pose_world, corners_cache, visualize=0)
#obj_img_pts["puck"] = puck_img_pts

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
if 0:  # Changed to 1 to regenerate predictions with correct image order
    print("loading model")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval()
    model = model.to(device).to(torch.float16)
    print("doing prediction")
    with torch.amp.autocast(device_type="mps", dtype=torch.float16): predictions = model(images)
    print("done prediction")
    torch.save(predictions,"../random/vggt_predictions_correct_order.pt")
else: predictions=torch.load("../random/vggt_predictions_correct_order.pt")

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

vggt_img_index = 0

# Convert images from (S, 3, H, W) to (S, H, W, 3) for colors

# Start viser server
server = viser.ViserServer(host="0.0.0.0", port=8080)
print(f"Viser server started on port 8080")

S, H, W, _ = world_points.shape
points = world_points.reshape(-1, 3)
colors_flat = (predictions["images"].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
if 0: conf_flat = predictions["depth_conf"].reshape(-1)
else: conf_flat = predictions["world_points_conf"].reshape(-1)
conf_mask = (conf_flat >= np.percentile(conf_flat, 10.0)) & (conf_flat > 1e-5)
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

# Get aruco corners in robot frame and transform coordinates to VGGT space
aruco_corners_all = []
img_pts_norm_all = []

# Get original and VGGT preprocessed dimensions
orig_height, orig_width = frame1.shape[0], frame1.shape[1]
vggt_preprocessed_shape = predictions["images"][vggt_img_index].shape  # (C, H, W)
vggt_height, vggt_width = vggt_preprocessed_shape[1], vggt_preprocessed_shape[2]

# Compute transformation: VGGT resizes width to 518, maintains aspect ratio, center crops height
scale_factor = vggt_width / orig_width
scaled_height = int(orig_height * scale_factor)
crop_offset_y = (scaled_height - vggt_height) // 2 if scaled_height > vggt_height else 0

print(f"\n=== Coordinate Transformation ===")
print(f"Original image: {orig_width}x{orig_height}")
print(f"VGGT preprocessed: {vggt_width}x{vggt_height}")
print(f"Scale factor: {scale_factor:.4f}")
print(f"Scaled height before crop: {scaled_height}")
print(f"Crop offset Y: {crop_offset_y}")

for obj_img_pts_, img_pts in obj_img_pts.values():
    aruco_corners_all.extend((np.linalg.inv(camera_pose_world)@ np.hstack([obj_img_pts_, np.ones((obj_img_pts_.shape[0], 1))]).T).T[:, :3])
    
    # Transform ArUco pixel coords from original space to VGGT preprocessed space
    # Account for pixel centers: pixel coordinate x represents the center of pixel, which is at position (x + 0.5)
    for pt in img_pts:
        # Convert to continuous coordinates (accounting for pixel centers)
        continuous_x = (pt[0] + 0.5) / orig_width
        continuous_y = (pt[1] + 0.5) / orig_height
        
        # Apply scaling in continuous space
        scaled_continuous_y = continuous_y * orig_height * scale_factor
        
        # Apply crop offset
        cropped_continuous_y = scaled_continuous_y - crop_offset_y
        
        # Convert back to pixel coordinates in VGGT space (subtract 0.5 to get pixel index from center)
        vggt_pixel_x = continuous_x * vggt_width - 0.5
        vggt_pixel_y = cropped_continuous_y - 0.5
        
        # Normalize to [0, 1] range for grid_sample
        norm_x = vggt_pixel_x / (vggt_width - 1)
        norm_y = vggt_pixel_y / (vggt_height - 1)
        
        img_pts_norm_all.append([norm_x, norm_y])

print(f"Transformed {len(img_pts_norm_all)} ArUco corner coordinates\n")

# Visualize sampling locations on the image
vggt_img_index = 0
if 0:
    # Get VGGT preprocessed image
    vggt_image = predictions["images"][vggt_img_index].transpose(1, 2, 0)  # (H, W, 3)
    
    # Ensure images are in uint8 range [0, 255]
    if vggt_image.max() <= 1.0:
        vggt_image = (vggt_image * 255).astype(np.uint8)
    else:
        vggt_image = vggt_image.astype(np.uint8)

    if frame1.max() <= 1.0:
        rgb_display = (frame1 * 255).astype(np.uint8)
    else:
        rgb_display = frame1.astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show the original ArUco detection image with detections
    axes[0].imshow(rgb_display)
    for obj_img_pts_, img_pts in obj_img_pts.values():
        for pt in img_pts:
            axes[0].plot(pt[0], pt[1], 'ro', markersize=4, alpha=0.7)
    axes[0].set_title(f"Original Image with ArUco detections\n{orig_width}x{orig_height}")
    axes[0].axis('off')
    
    # Show the VGGT preprocessed image with transformed coordinates
    axes[1].imshow(vggt_image)
    for i, img_pt_norm in enumerate(img_pts_norm_all):
        # Convert normalized coords to VGGT pixel coords
        pixel_x = img_pt_norm[0] * (vggt_width - 1)

        pixel_y = img_pt_norm[1] * (vggt_height - 1)
        axes[1].plot(pixel_x, pixel_y, 'go', markersize=4, alpha=0.7)
        if i < 10:  # Label first 10
            axes[1].text(pixel_x + 3, pixel_y + 3, str(i), color='green', fontsize=7)
    axes[1].set_title(f"VGGT Preprocessed with transformed coords\n{vggt_width}x{vggt_height}")
    axes[1].axis('off')
    
    # Show overlay to verify alignment
    axes[2].imshow(vggt_image)
    for i, img_pt_norm in enumerate(img_pts_norm_all):
        pixel_x = img_pt_norm[0] * (vggt_width - 1)
        pixel_y = img_pt_norm[1] * (vggt_height - 1)
        # Draw a crosshair to make it easier to see
        axes[2].plot(pixel_x, pixel_y, 'r+', markersize=10, markeredgewidth=2, alpha=0.8)
    axes[2].set_title(f"Transformed points on VGGT image\n(verify alignment)")
    axes[2].axis('off')
    
    plt.tight_layout()
    print(f"ArUco detection image shape: {frame1.shape}")
    print(f"VGGT image shape: {vggt_image.shape}")
    print(f"VGGT world_points shape: {world_points.shape}")
    print(f"Image value ranges - RGB: [{rgb_display.min()}, {rgb_display.max()}], VGGT: [{vggt_image.min()}, {vggt_image.max()}]")
    plt.savefig("scratch/aruco_sampling_debug.png", dpi=150, bbox_inches='tight')
    print(f"Saved sampling visualization to scratch/aruco_sampling_debug.png")
    plt.show()
    plt.close()

# Sample vggt aruco corners in vggt frame
# Use vggt_img_index (should be 0 after regenerating predictions with correct image order)
vggt_corners_all = torch.nn.functional.grid_sample(torch.from_numpy(world_points)[[vggt_img_index]].permute(0,3,1,2), torch.from_numpy(np.stack(img_pts_norm_all))[None,None].float()*2-1, mode='nearest', padding_mode='border', align_corners=False).squeeze().T.numpy()
vggt_corners_conf_all = torch.nn.functional.grid_sample(torch.from_numpy(predictions["world_points_conf"][...,None])[[vggt_img_index]].permute(0,3,1,2), torch.from_numpy(np.stack(img_pts_norm_all))[None,None].float()*2-1, mode='nearest', padding_mode='border', align_corners=False).squeeze().T.numpy()

# Filter arucos with vggt confidence
vggt_aruco_conf = vggt_corners_conf_all > np.percentile(vggt_corners_conf_all, 10.0)
vggt_corners_all = vggt_corners_all[vggt_aruco_conf]
aruco_corners_all = np.array(aruco_corners_all)[vggt_aruco_conf]

# Ensure we have the same number of points for alignment
T_procrustes, scale, rotation, translation = procrustes_alignment(aruco_corners_all, vggt_corners_all)
points_homogeneous = np.hstack([world_points.reshape(-1, 3), np.ones((world_points.reshape(-1, 3).shape[0], 1))])
vggt_in_robot_frame = (T_procrustes @ points_homogeneous.T).T[:,:3]
server.scene.add_point_cloud( name="robot_vggt_points", points=vggt_in_robot_frame, colors=colors_flat, point_size=0.001, point_shape="circle",)
#server.scene.add_point_cloud( name="raw_frame_pcd", points=points_homogeneous[:,:3], colors=colors_flat, point_size=0.001, point_shape="circle",)
print(f"Added {len(vggt_in_robot_frame)} VGGT corner points to viser (in robot coordinate frame)")

aruco_point_cloud = server.scene.add_point_cloud( name="aruco_points", points=aruco_corners_all, colors=np.tile([255, 0, 0], (len(aruco_corners_all), 1)), point_size=0.002, point_shape="circle",)
vggt_aruco_point_cloud = server.scene.add_point_cloud( name="vggt_aruco_points", points=np.array(vggt_corners_all), colors=np.tile([0,255,  0], (len(vggt_corners_all), 1)), point_size=0.004, point_shape="circle",)
print(f"Added {len(aruco_corners_all)} ArUco corner points to viser (in robot coordinate frame)")

print("Viser visualization ready. Open http://localhost:8080 in your browser")

# Keep server running
while True:
    import time
    time.sleep(0.01)
