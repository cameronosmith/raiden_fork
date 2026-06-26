import torch
import numpy as np
from PIL import Image
from torchvision import transforms as TF
import viser


def preprocess_numpy_images(frame_list, mode="crop"):
    """
    Preprocess numpy arrays similar to load_and_preprocess_images from load_fn.py
    
    Args:
        frame_list (list): List of numpy arrays representing images in (H, W, 3) format
        mode (str): Preprocessing mode, either "crop" or "pad"
    
    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)
    """
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518
    
    for frame in frame_list:
        # Convert numpy array to PIL Image
        # Assuming frame is in (H, W, 3) format with values 0-255
        img = Image.fromarray(frame.astype(np.uint8))
        
        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14
        
        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)
        
        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
        
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
    
    # Check if we have different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        
        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    
    images = torch.stack(images)  # concatenate images
    
    # Ensure correct shape when single image
    if len(frame_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
    
    return images


def procrustes_alignment(X, Y):
    """
    Compute optimal rigid transformation (R, t, s) that aligns Y to X
    using Procrustes analysis
    """
    # Center the point clouds
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute rotation using SVD
    H = Y_centered.T @ X_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale using a more robust method
    # Scale based on the ratio of point cloud sizes
    X_scale = np.sqrt(np.sum(X_centered**2) / len(X_centered))
    Y_scale = np.sqrt(np.sum(Y_centered**2) / len(Y_centered))
    scale = X_scale / Y_scale
    
    # Alternative: use the original Procrustes scale but with better normalization
    # scale = np.trace(R.T @ H) / np.trace(Y_centered.T @ Y_centered)
    
    # Compute translation
    t = np.mean(X, axis=0) - scale * R @ np.mean(Y, axis=0)
    
    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    
    return T, scale, R, t