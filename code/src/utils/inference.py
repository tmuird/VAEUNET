import torch
import torch.nn.functional as F
import numpy as np

def create_gaussian_window(patch_size):
    """Create a 2D Gaussian window for weighted averaging of overlapping patches"""
    sigma = patch_size / 6  # Adjust this value to control the blending
    center = patch_size // 2
    
    y, x = np.ogrid[-center:patch_size-center, -center:patch_size-center]
    window = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    window = torch.from_numpy(window).float()
    return window

def sliding_window_inference(model, image, patch_size, stride, device):
    """
    Perform sliding window inference on large images
    
    Args:
        model: The neural network model
        image: Input image tensor [C, H, W]
        patch_size: Size of patches to process
        stride: Stride between patches
        device: Computation device
    
    Returns:
        Reconstructed full-size prediction
    """
    model.eval()
    _, h, w = image.shape
    
    # Initialize output tensor and count map for averaging
    output = torch.zeros((1, h, w), device=device)
    count = torch.zeros((h, w), device=device)
    
    # Create Gaussian window for weighted averaging
    window = create_gaussian_window(patch_size).to(device)
    
    with torch.no_grad():
        # Calculate number of windows
        n_h = max(1, (h - patch_size + stride) // stride)
        n_w = max(1, (w - patch_size + stride) // stride)
        
        for i in range(n_h):
            for j in range(n_w):
                # Calculate patch coordinates
                y = min(i * stride, h - patch_size)
                x = min(j * stride, w - patch_size)
                
                # Extract and process patch
                patch = image[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
                patch = patch.to(device)
                
                # Get prediction
                pred = model(patch)
                pred = torch.sigmoid(pred).squeeze(0)
                
                # Apply Gaussian window
                pred = pred * window
                
                # Add to output and count
                output[0, y:y+patch_size, x:x+patch_size] += pred
                count[y:y+patch_size, x:x+patch_size] += window
    
    # Average overlapping regions
    output = output / count.unsqueeze(0).clamp(min=1e-8)
    
    return output 