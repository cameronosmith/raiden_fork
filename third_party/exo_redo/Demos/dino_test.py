import torch
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append("/Users/cameronsmith/Projects/robotics_testing/random/vggt")
sys.path.append("/Users/cameronsmith/Projects/robotics_testing/random/MoGe")
sys.path.append("/Users/cameronsmith/Projects/robotics_testing/random/dinotool")
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

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA 
from scipy import signal

REPO_DIR = "/Users/cameronsmith/Projects/robotics_testing/random/dinov3" 
WEIGHTS_PATH = "/Users/cameronsmith/Projects/robotics_testing/random/dinov3/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
PATCH_SIZE = 16
IMAGE_SIZE = 768 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Load model from local weights

# image resize transform to dimensions divisible by patch size
def resize_transform( mask_image: Image.Image, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE,) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

# Load and preprocess image
img = Image.open(imgpath).convert("RGB")
img = img.resize(np.array(img.size)//4)  # Resize for faster processing
image_resized = resize_transform(img)
image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Extract features
n_layers = 12
with torch.inference_mode():
    with torch.autocast(device_type='mps', dtype=torch.float32):

        if 1:
            dinov3_model = torch.hub.load( REPO_DIR, 'dinov3_vits16plus', source='local', weights=WEIGHTS_PATH).to("mps")
            print(f"Loaded DINOv3 model from local weights: {WEIGHTS_PATH}")
            feats = dinov3_model.get_intermediate_layers( image_resized_norm.unsqueeze(0).to("mps"), n=range(n_layers), reshape=True, norm=True)
            torch.save(feats, "scratch/dinofeats.pt")
        else:feats=torch.load("scratch/dinofeats.pt")
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]

# Apply PCA with 12 components (to visualize 4 sets of RGB channels)
pca = PCA(n_components=12, whiten=True)
pca.fit(x)
pca_features_all = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 12)

# Create visualizations for each set of 3 components
pca_images = []
pca_upsampled_list = []

for i in range(4):  # 4 sets: [0-2], [3-5], [6-8], [9-11]
    start_idx = i * 3
    end_idx = start_idx + 3
    
    # Extract 3 components and apply sigmoid for vibrant colors
    pca_subset = pca_features_all[:, :, start_idx:end_idx]
    pca_rgb = torch.nn.functional.sigmoid(pca_subset.mul(2.0)).permute(2, 0, 1)
    
    # Upsample to original image size
    pca_upsampled = TF.resize(
        pca_rgb,  # CHW format
        img.size[::-1],  # (height, width)
        interpolation=TF.InterpolationMode.BILINEAR
    ).permute(1, 2, 0).numpy()  # Convert to HWC for display
    
    pca_upsampled_list.append(pca_upsampled)

# Create 3-row figure: original + 4 PCA visualizations, and overlays
fig, axes = plt.subplots(3, 5, figsize=(30, 18))

# Row 1: Pure PCA visualizations
axes[0, 0].imshow(img)
axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(pca_upsampled_list[0])
axes[0, 1].set_title('PCA Components 0-2', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(pca_upsampled_list[1])
axes[0, 2].set_title('PCA Components 3-5', fontsize=14, fontweight='bold')
axes[0, 2].axis('off')

axes[0, 3].imshow(pca_upsampled_list[2])
axes[0, 3].set_title('PCA Components 6-8', fontsize=14, fontweight='bold')
axes[0, 3].axis('off')

axes[0, 4].imshow(pca_upsampled_list[3])
axes[0, 4].set_title('PCA Components 9-11', fontsize=14, fontweight='bold')
axes[0, 4].axis('off')

# Row 2: Overlays with original image
img_normalized = np.array(img).astype(float) / 255.0

axes[1, 0].imshow(img)
axes[1, 0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')

for i in range(4):
    overlay = img_normalized * 0.5 + pca_upsampled_list[i] * 0.5
    axes[1, i+1].imshow(overlay)
    axes[1, i+1].set_title(f'Overlay: RGB + PCA {i*3}-{i*3+2}', fontsize=14, fontweight='bold')
    axes[1, i+1].axis('off')

# Row 3: Show explained variance for each component
explained_var = pca.explained_variance_ratio_
axes[2, 0].bar(range(12), explained_var[:12])
axes[2, 0].set_xlabel('PCA Component', fontsize=12)
axes[2, 0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[2, 0].set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
axes[2, 0].set_xticks(range(12))

# Show cumulative explained variance
axes[2, 1].plot(range(12), np.cumsum(explained_var[:12]), 'bo-', linewidth=2, markersize=8)
axes[2, 1].set_xlabel('Number of Components', fontsize=12)
axes[2, 1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[2, 1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes[2, 1].set_xticks(range(12))
axes[2, 1].grid(True, alpha=0.3)

# Hide unused subplots
axes[2, 2].axis('off')
axes[2, 3].axis('off')
axes[2, 4].axis('off')

plt.tight_layout()
plt.show()

import pdb; pdb.set_trace()


zz







robot_config = EXOSKELETON_CONFIGS["so100_holemounts"]
mj_model= mujoco.MjModel.from_xml_string(combine_xmls(robot_config.xml, ALIGNMENT_BOARD_CONFIG.get_xml_addition()))
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)
rgb=frame1
if rgb.max() <= 1.0: rgb = (rgb * 255).astype(np.uint8)
# Detect link poses from ArUco markers
link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, mj_model, mj_data, robot_config)
configuration = estimate_robot_state( mj_model, mj_data, robot_config, link_poses, ik_iterations=55)

# Detect and position alignment board, add to obj_img_pts for VGGT alignment
board_result = detect_and_position_alignment_board(rgb, mj_model, mj_data, ALIGNMENT_BOARD_CONFIG, cam_K, camera_pose_world, corners_cache, visualize=False)
board_pose, board_pts = board_result
obj_img_pts["alignment_board"] = board_pts
print(f"Alignment board detected and added to VGGT alignment ({len(board_pts[1])} points)")
obj_img_pts={"larger_base":obj_img_pts["larger_base"],"alignment_board":obj_img_pts["alignment_board"]} # just use larger base and alignment board for aruco point alignments (flat table surfaces)

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
points_flat = points.reshape(-1, 3)
mask_flat = mask.reshape(-1)
valid_points = points_flat[mask_flat > 0.5]
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
# Procrustes alignment: align MOGE pointcloud to robot frame
T_procrustes, scale, rotation, translation = procrustes_alignment(aruco_corners_robot_frame, moge_aruco_corners)
points_homogeneous = np.hstack([valid_points, np.ones((len(valid_points), 1))])
moge_aligned = (T_procrustes @ points_homogeneous.T).T[:, :3]

# Add aligned pointcloud to viser
server.scene.add_point_cloud( name="/moge_aligned", points=moge_aligned, colors=valid_colors, point_size=0.001,)
server.scene.add_point_cloud( name="/aruco_robot", points=aruco_corners_robot_frame, colors=np.tile([255, 0, 0], (len(aruco_corners_robot_frame), 1)), point_size=0.003,)

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