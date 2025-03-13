import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import logging
from pathlib import Path
from torchvision import transforms
from utils.data_loading import IDRIDDataset, load_image
from unet.unet_resnet import UNetResNet
import math
import argparse
from tqdm import tqdm
from collections import defaultdict
import shutil

def get_image_for_prediction(dataset, img_idx):
    """Get image and mask from dataset for prediction."""
    sample = dataset[img_idx]
    img = sample['image'].unsqueeze(0)  # Add batch dimension [1, C, H, W]
    mask = sample['mask'].unsqueeze(0)  # Add batch dimension [1, 1, H, W]
    img_id = sample['img_id']
    
    print(f"Image tensor shape: {img.shape}")
    print(f"Mask tensor shape: {mask.shape}")
    return img, mask, img_id

def predict_full_image(model, img, z):
    """Predict segmentation using the full image (no patching)."""
    model.eval()
    
    # Process the full image in one go
    with torch.no_grad():
        # Get encoder features
        features = model.encoder(img)
        x_enc = features[-1]
        
        # Interpolate z to match encoder output size
        z_full = F.interpolate(z, size=x_enc.shape[2:], mode='bilinear', align_corners=True)
        
        # Initial projection
        x = model.z_initial(z_full)
        
        # Decode with z injection at each stage
        for k, decoder_block in enumerate(model.decoder_blocks):
            skip = features[-(k+2)] if k < len(features)-1 else None
            x = decoder_block(x, skip, z_full)
        
        # Final conv and sigmoid
        output = torch.sigmoid(model.final_conv(x))
        
        print(f"Full image prediction shape: {output.shape}")
        
    return output

def get_patches_for_image(dataset, img_id):
    """Get all patches for a specific image ID."""
    patches = []
    patch_coords = []
    
    # Find all indices in the dataset that belong to the specified image ID
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample['img_id'] == img_id:
            img = sample['image'].unsqueeze(0)  # Add batch dimension
            # Load patch coordinates from the saved data
            patch_data = torch.load(dataset.patch_indices[idx][1])
            coords = patch_data['coords']
            patches.append(img)
            patch_coords.append(coords)
    
    return patches, patch_coords

def predict_with_dataset_patches(model, dataset, img_id, z, device):
    """Use the dataset's patch mechanism to predict on patches and reassemble."""
    model.eval()
    
    # Get all patches for this image
    patches, patch_coords = get_patches_for_image(dataset, img_id)
    
    if not patches:
        logging.error(f"No patches found for image ID: {img_id}")
        return None
    
    # Get original image shape from the first patch's metadata
    patch_data = torch.load(dataset.patch_indices[0][1])
    if 'original_shape' in patch_data:
        H, W = patch_data['original_shape']
    else:
        # If not available, use a large enough size based on coordinates
        max_y = max(y + p.shape[2] for p, (y, _) in zip(patches, patch_coords))
        max_x = max(x + p.shape[3] for p, (_, x) in zip(patches, patch_coords))
        H, W = max_y, max_x
    
    # Initialize output and weight masks
    output_mask = torch.zeros((1, 1, H, W), device=device)
    weight_mask = torch.zeros((1, 1, H, W), device=device)
    
    # Process each patch
    with torch.no_grad():
        for i, (patch, (y, x)) in enumerate(zip(patches, patch_coords)):
            patch = patch.to(device)
            
            # Get encoder features for this patch
            features = model.encoder(patch)
            x_enc = features[-1]
            
            # Interpolate z to match encoder output size
            z_patch = F.interpolate(z, size=x_enc.shape[2:], mode='bilinear', align_corners=True)
            
            # Initial projection
            x_proj = model.z_initial(z_patch)
            
            # Decode with z injection
            for k, decoder_block in enumerate(model.decoder_blocks):
                skip = features[-(k+2)] if k < len(features)-1 else None
                x_proj = decoder_block(x_proj, skip, z_patch)
            
            # Final prediction
            patch_pred = torch.sigmoid(model.final_conv(x_proj))
            
            # Create tapering weight mask for smooth blending
            patch_h, patch_w = patch.shape[2], patch.shape[3]
            weight = torch.ones_like(patch_pred)
            
            # Apply smooth tapering at edges (similar to the original function)
            overlap = 50  # Use a reasonable overlap value
            for axis in [2, 3]:
                if patch_pred.shape[axis] > 2*overlap:
                    ramp = torch.linspace(0, 1, overlap, device=device)
                    if axis == 2:
                        weight[:, :, :overlap, :] *= ramp.view(-1, 1)
                        weight[:, :, -overlap:, :] *= (1 - ramp).view(-1, 1)
                    else:
                        weight[:, :, :, :overlap] *= ramp.view(-1)
                        weight[:, :, :, -overlap:] *= (1 - ramp).view(-1)
            
            # Add weighted prediction to output in the correct position
            h, w = patch_pred.shape[2], patch_pred.shape[3]
            output_mask[:, :, y:y+h, x:x+w] += patch_pred * weight
            weight_mask[:, :, y:y+h, x:x+w] += weight
    
    # Average overlapping regions
    output_mask = output_mask / (weight_mask + 1e-8)
    
    return output_mask

def calculate_uncertainty_metrics(segmentations):
    """Calculate various uncertainty metrics from multiple segmentation samples."""
    # Calculate mean prediction
    mean_pred = segmentations.mean(dim=0)  # [B, C, H, W]
    
    # Standard deviation (aleatory uncertainty)
    std_dev = segmentations.std(dim=0)     # [B, C, H, W]
    
    # Entropy of the mean prediction (epistemic uncertainty)
    epsilon = 1e-7  # Small constant to avoid log(0)
    entropy = -(mean_pred * torch.log(mean_pred + epsilon) + 
               (1 - mean_pred) * torch.log(1 - mean_pred + epsilon))
    
    # Mutual information (total uncertainty)
    sample_entropies = -(segmentations * torch.log(segmentations + epsilon) + 
                        (1 - segmentations) * torch.log(1 - segmentations + epsilon))
    mean_entropy = sample_entropies.mean(dim=0)
    mutual_info = entropy - mean_entropy
    
    # Coefficient of variation
    coeff_var = std_dev / (mean_pred + epsilon)
    
    return {
        'mean': mean_pred.squeeze(1),      # Remove channel dim for visualization
        'std': std_dev.squeeze(1),
        'entropy': entropy.squeeze(1),
        'mutual_info': mutual_info.squeeze(1),
        'coeff_var': coeff_var.squeeze(1)
    }

def get_segmentation_distribution(model, img, img_id, dataset=None, num_samples=32, patch_dataset=None, temperature=1.0, enable_dropout=True):
    """Generate multiple segmentations using either full image or patches from dataset."""
    if enable_dropout:
        model.train()  # Enable dropout
    else:
        model.eval()
        
    device = img.device
    
    # Get latent distribution parameters
    with torch.no_grad():
        mu, logvar = model.encode(img)
    print(f"Latent distribution shapes - mu: {mu.shape}, logvar: {logvar.shape}")
    
    # Initialize list to store segmentations
    segmentations = []
    
    # Generate multiple samples
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Sample from latent space
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        
        # Choose prediction method based on available datasets
        if patch_dataset is not None:
            # Generate segmentation using patches from dataset
            seg = predict_with_dataset_patches(model, patch_dataset, img_id, z, device)
        else:
            # Generate segmentation using full image
            seg = predict_full_image(model, img, z)
            
        print(f"Sample {i} segmentation shape: {seg.shape}")
        segmentations.append(seg)
        
        # Free some memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Stack all segmentations
    segmentations = torch.cat(segmentations, dim=0)
    print(f"Final stacked segmentations shape: {segmentations.shape}")
    
    return segmentations, mu, logvar

def plot_reconstruction(model, img, mask, img_id, dataset=None, num_samples=32, patch_dataset=None, temperature=1.0, enable_dropout=True):
    """Plot the input image, ground truth mask, and uncertainty analysis from multiple sampled reconstructions."""
    # Ensure correct dimensions
    if img.dim() != 4:
        raise ValueError(f"Expected img to have 4 dimensions [B, C, H, W], got shape {img.shape}")
    if mask.dim() != 4:
        raise ValueError(f"Expected mask to have 4 dimensions [B, C, H, W], got shape {mask.shape}")
        
    print(f"Input shapes - img: {img.shape}, mask: {mask.shape}")
    
    # Get multiple segmentations
    segmentations, mu, logvar = get_segmentation_distribution(
        model, img, img_id, dataset=dataset, num_samples=num_samples,
        patch_dataset=patch_dataset, temperature=temperature, enable_dropout=enable_dropout
    )
    print(f"Segmentations shape after distribution: {segmentations.shape}")
    
    # Calculate uncertainty metrics
    metrics = calculate_uncertainty_metrics(segmentations)
    
    # Create figure with subplots
    plt.rcParams['figure.dpi'] = 500  # Higher DPI for better quality
    fig = plt.figure(figsize=(20, 16))  # Increased figure size for more plots
    gs = gridspec.GridSpec(3, 3, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)  # Increased spacing between plots
    
    # Original image (take first batch) with improved visualization
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get image from tensor and properly prepare it for visualization
    # Following the same approach as in data_loading.py but for visualization
    img_display = img[0].cpu().permute(1, 2, 0).numpy()  # Convert to [H, W, C] format
    
    # Convert normalized values to regular RGB range
    # Simply rescale from normalized 0-1 to 0-255 range
    img_display = np.clip(img_display, 0, 1)  # Ensure proper range
    
    # Display the properly scaled image
    ax1.imshow(img_display)
    ax1.set_title('Input Image', fontsize=12, pad=10)
    ax1.axis('off')
    
    # Ground truth mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask[0, 0].cpu(), cmap='gray')
    ax2.set_title('Ground Truth', fontsize=12, pad=10)
    ax2.axis('off')
    
    # Mean prediction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(metrics['mean'][0].cpu(), cmap='gray')
    ax3.set_title(f'Mean Prediction\n(T={temperature}, N={num_samples})', fontsize=12, pad=10)
    ax3.axis('off')
    
    # Standard deviation
    ax4 = fig.add_subplot(gs[1, 0])
    std_plot = ax4.imshow(metrics['std'][0].cpu(), cmap='hot')
    ax4.set_title('Std Deviation\n(Aleatory Uncertainty)', fontsize=12, pad=10)
    ax4.axis('off')
    plt.colorbar(std_plot, ax=ax4)
    
    # Entropy
    ax5 = fig.add_subplot(gs[1, 1])
    entropy_plot = ax5.imshow(metrics['entropy'][0].cpu(), cmap='hot')
    ax5.set_title('Entropy\n(Epistemic Uncertainty)', fontsize=12, pad=10)
    ax5.axis('off')
    plt.colorbar(entropy_plot, ax=ax5)
    
    # Mutual Information
    ax6 = fig.add_subplot(gs[1, 2])
    mi_plot = ax6.imshow(metrics['mutual_info'][0].cpu(), cmap='hot')
    ax6.set_title('Mutual Information\n(Total Uncertainty)', fontsize=12, pad=10)
    ax6.axis('off')
    plt.colorbar(mi_plot, ax=ax6)
    
    # Sample predictions
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(segmentations[i, 0].cpu(), cmap='gray')
        ax.set_title(f'Sample {i+1}', fontsize=12, pad=10)
        ax.axis('off')
    
    plt.suptitle('VAE-UNet Segmentation Analysis', fontsize=14, y=0.95)
    plt.tight_layout()
    return fig

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='best_model.pth', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--lesion_type', type=str, required=True, default='EX', choices=['EX', 'HE', 'MA','OD'], help='Lesion type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling (default: 1.0)')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for prediction. Set to 0 for full image inference')
    parser.add_argument('--overlap', type=int, default=100, help='Overlap between patches (default: 100)')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for resizing (default: 1.0)')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples for ensemble prediction (default: 10)')
    parser.add_argument('--attention', dest='use_attention', action='store_true', help='Enable attention mechanism (default)')
    parser.add_argument('--no-attention', dest='use_attention', action='store_false', help='Disable attention mechanism')
    parser.add_argument('--enable_dropout', action='store_true', help='Enable dropout during inference')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save output images')
    parser.add_argument('--max_images', type=int, default=3, help='Maximum number of test images to process')
    parser.set_defaults(use_attention=True, enable_dropout=False)
    return parser.parse_args()

def predict_with_patches(model, img, z, patch_size=512, overlap=100):
    """Predict segmentation using patches."""
    model.eval()
    device = img.device
    B, C, H, W = img.shape
    
    # Calculate number of patches needed
    stride = patch_size - overlap
    n_patches_h = math.ceil((H - overlap) / stride)
    n_patches_w = math.ceil((W - overlap) / stride)
    
    print(f"Input image shape: {img.shape}")
    print(f"Processing {n_patches_h}x{n_patches_w} patches")
    
    # Initialize output mask with zeros
    output_mask = torch.zeros((B, 1, H, W), device=device)
    weight_mask = torch.zeros((B, 1, H, W), device=device)
    
    # Process each patch
    with torch.no_grad():
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                start_h = i * stride
                start_w = j * stride
                
                # Handle last row and column
                if i == n_patches_h - 1:
                    end_h = H
                    start_h = max(0, end_h - patch_size)
                else:
                    end_h = min(start_h + patch_size, H)
                    
                if j == n_patches_w - 1:
                    end_w = W
                    start_w = max(0, end_w - patch_size)
                else:
                    end_w = min(start_w + patch_size, W)
                
                # Extract patch
                patch = img[:, :, start_h:end_h, start_w:end_w]
                patch_h, patch_w = patch.shape[2:]
                print(f"Patch {i},{j} original size: {patch.shape}")
                
                # Get encoder features for this patch
                features = model.encoder(patch)
                x_enc = features[-1]
                
                # Interpolate z to match encoder output size
                z_patch = F.interpolate(z, size=x_enc.shape[2:], mode='bilinear', align_corners=True)
                
                # Initial projection and decoding
                x = model.z_initial(z_patch)
                
                # Decode with z injection at each stage
                for k, decoder_block in enumerate(model.decoder_blocks):
                    skip = features[-(k+2)] if k < len(features)-1 else None
                    x = decoder_block(x, skip, z_patch)
                
                # Final conv and sigmoid
                patch_pred = torch.sigmoid(model.final_conv(x))
                
                # Resize prediction back to original patch size if needed
                if patch_pred.shape[2:] != patch.shape[2:]:
                    patch_pred = F.interpolate(patch_pred, size=(patch_h, patch_w), mode='bilinear', align_corners=True)
                
                # Create weight mask for blending
                weight = torch.ones_like(patch_pred)
                
                # Apply tapering at edges for smooth blending
                if overlap > 0:
                    for axis in [2, 3]:  # Height and width dimensions
                        if patch_pred.shape[axis] > overlap:
                            # Create linear ramp for overlap regions
                            ramp = torch.linspace(0, 1, overlap, device=device)
                            
                            # Apply ramp to start of patch if not at image boundary
                            if (i > 0 and axis == 2) or (j > 0 and axis == 3):
                                if axis == 2:
                                    weight[:, :, :overlap, :] *= ramp.view(-1, 1)
                                else:
                                    weight[:, :, :, :overlap] *= ramp.view(-1)
                            
                            # Apply ramp to end of patch if not at image boundary
                            if ((i < n_patches_h - 1 and axis == 2) or 
                                (j < n_patches_w - 1 and axis == 3)):
                                if axis == 2:
                                    weight[:, :, -overlap:, :] *= (1 - ramp).view(-1, 1)
                                else:
                                    weight[:, :, :, -overlap:] *= (1 - ramp).view(-1)
                
                # Add weighted prediction to output
                output_mask[:, :, start_h:end_h, start_w:end_w] += patch_pred * weight
                weight_mask[:, :, start_h:end_h, start_w:end_w] += weight
    
    # Average overlapping regions
    output_mask = output_mask / (weight_mask + 1e-8)
    print(f"Final mask shape: {output_mask.shape}")
    
    return output_mask

def get_image_and_mask(dataset, img_id):
    """Get full image and mask for a specific image ID by finding its full image or combining patches."""
    full_img = None
    full_mask = None
    coords = []
    original_shape = None
    
    # Try to find if there's a full image entry
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample['img_id'] == img_id:
            # Try to load the patch data to get coordinates and original shape
            patch_path = dataset.patch_indices[idx][1]
            patch_data = torch.load(patch_path)
            
            # If this is a full image (coords are (0,0)), use it directly
            if patch_data['coords'] == (0, 0) and 'original_shape' in patch_data:
                print(f"Found full image for {img_id}")
                full_img = sample['image']
                full_mask = sample['mask']
                original_shape = patch_data.get('original_shape', full_img.shape[1:])
                break
            elif dataset.patch_size is None:
                # If patch_size is None, we're using full images anyway
                print(f"Using full image mode for {img_id}")
                full_img = sample['image']
                full_mask = sample['mask']
                original_shape = patch_data.get('original_shape', full_img.shape[1:])
                break
            # Otherwise collect patch information
            coords.append(patch_data['coords'])
    
    # If we didn't find a full image, we need to stitch patches
    if full_img is None and coords:
        # Find the size of the original image from the patch coordinates
        max_h = max(y + dataset.patch_size for y, _ in coords)
        max_w = max(x + dataset.patch_size for _, x in coords)
        
        # Create tensors to hold the full image and mask
        full_img = torch.zeros((3, max_h, max_w), dtype=torch.float32)
        full_mask = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        weight = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        
        # Collect all patches for this image ID
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample['img_id'] == img_id:
                patch_path = dataset.patch_indices[idx][1]
                patch_data = torch.load(patch_path)
                y, x = patch_data['coords']
                
                # Get patch
                img_patch = sample['image']
                mask_patch = sample['mask']
                
                # Add patch to the full image with feathering for overlap
                patch_h, patch_w = img_patch.shape[1:]
                
                # Create weight mask for smooth blending
                patch_weight = torch.ones((1, patch_h, patch_w))
                overlap = dataset.patch_size // 4  # Use 1/4 of patch size as overlap
                
                # Apply tapering at edges
                if overlap > 0:
                    for axis in [0, 1]:  # Height and width dimensions
                        if patch_h > 2*overlap:  # Only apply if patch is big enough
                            ramp = torch.linspace(0, 1, overlap)
                            if axis == 0:
                                patch_weight[:, :overlap, :] *= ramp.view(-1, 1)
                                patch_weight[:, -overlap:, :] *= (1 - ramp).view(-1, 1)
                            else:
                                patch_weight[:, :, :overlap] *= ramp.view(1, -1)
                                patch_weight[:, :, -overlap:] *= (1 - ramp).view(1, -1)
                
                # Add weighted patch to output
                full_img[:, y:y+patch_h, x:x+patch_w] += img_patch * patch_weight
                full_mask[:, y:y+patch_h, x:x+patch_w] += mask_patch * patch_weight
                weight[:, y:y+patch_h, x:x+patch_w] += patch_weight
        
        # Average overlapping regions
        valid_mask = (weight > 0).float()
        full_img = full_img / (weight + 1e-8)
        full_mask = full_mask / (weight + 1e-8)
        
        # Set valid_mask to original size for reference
        original_shape = (max_h, max_w)
    
    if full_img is None:
        raise ValueError(f"No image data found for ID {img_id}")
        
    return full_img, full_mask, original_shape

def get_segmentation_distribution(model, img_id, dataset, num_samples=32, patch_size=None, overlap=100, temperature=1.0, enable_dropout=True):
    """Generate multiple segmentations using either full image or patch-based prediction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the full image and mask
    img, mask, original_shape = get_image_and_mask(dataset, img_id)
    img = img.unsqueeze(0).to(device)  # Add batch dimension [1, C, H, W]
    print(f"Full image shape for segmentation: {img.shape}")
    
    if enable_dropout:
        model.train()  # Enable dropout
    else:
        model.eval()
    
    # Get latent distribution parameters
    with torch.no_grad():
        mu, logvar = model.encode(img)
    print(f"Latent distribution shapes - mu: {mu.shape}, logvar: {logvar.shape}")
    
    # Initialize list to store segmentations
    segmentations = []
    
    # Generate multiple samples
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Sample from latent space with temperature
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        
        # Choose prediction method based on patch size
        if patch_size is not None:
            # Use patch-based prediction (original working method)
            seg = predict_with_patches(model, img, z, patch_size, overlap)
        else:
            # Use full image prediction (no patches)
            seg = predict_full_image(model, img, z)
            
        print(f"Sample {i} segmentation shape: {seg.shape}")
        segmentations.append(seg)
        
        # Free some memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Stack all segmentations along batch dimension
    segmentations = torch.cat(segmentations, dim=0)
    print(f"Final stacked segmentations shape: {segmentations.shape}")
    
    return segmentations, mask.unsqueeze(0).to(device), mu, logvar

def plot_reconstruction(model, img_id, dataset, num_samples=32, patch_size=None, overlap=100, temperature=1.0, enable_dropout=True):
    """Plot image, mask, and uncertainty analysis using either full image or patch-based prediction."""
    # Get multiple segmentations with either patch-based or full image mode
    segmentations, mask, mu, logvar = get_segmentation_distribution(
        model, img_id, dataset=dataset, num_samples=num_samples,
        patch_size=patch_size, overlap=overlap, 
        temperature=temperature, enable_dropout=enable_dropout
    )
    
    # Get the raw image for display
    raw_img, _, _ = get_image_and_mask(dataset, img_id)
    raw_img = raw_img.unsqueeze(0)  # Add batch dimension
    
    print(f"Segmentations shape: {segmentations.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image shape for display: {raw_img.shape}")
    
    # Calculate uncertainty metrics
    metrics = calculate_uncertainty_metrics(segmentations)
    
    # Create figure with subplots
    plt.rcParams['figure.dpi'] = 500  # Higher DPI for better quality
    fig = plt.figure(figsize=(20, 16))  # Increased figure size for more plots
    gs = gridspec.GridSpec(3, 3, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)  # Increased spacing between plots
    
    # Original image (take first batch) with improved visualization
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get image from tensor and properly prepare it for visualization
    img_display = raw_img[0].cpu().permute(1, 2, 0).numpy()  # Convert to [H, W, C] format
    img_display = np.clip(img_display, 0, 1)  # Ensure proper range
    
    # Display the properly scaled image
    ax1.imshow(img_display)
    ax1.set_title('Input Image', fontsize=12, pad=10)
    ax1.axis('off')
    
    # Ground truth mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask[0, 0].cpu(), cmap='gray')
    ax2.set_title('Ground Truth', fontsize=12, pad=10)
    ax2.axis('off')
    
    # Mean prediction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(metrics['mean'][0].cpu(), cmap='gray')
    ax3.set_title(f'Mean Prediction\n(T={temperature}, N={num_samples})', fontsize=12, pad=10)
    ax3.axis('off')
    
    # Standard deviation
    ax4 = fig.add_subplot(gs[1, 0])
    std_plot = ax4.imshow(metrics['std'][0].cpu(), cmap='hot')
    ax4.set_title('Std Deviation\n(Aleatory Uncertainty)', fontsize=12, pad=10)
    ax4.axis('off')
    plt.colorbar(std_plot, ax=ax4)
    
    # Entropy
    ax5 = fig.add_subplot(gs[1, 1])
    entropy_plot = ax5.imshow(metrics['entropy'][0].cpu(), cmap='hot')
    ax5.set_title('Entropy\n(Epistemic Uncertainty)', fontsize=12, pad=10)
    ax5.axis('off')
    plt.colorbar(entropy_plot, ax=ax5)
    
    # Mutual Information
    ax6 = fig.add_subplot(gs[1, 2])
    mi_plot = ax6.imshow(metrics['mutual_info'][0].cpu(), cmap='hot')
    ax6.set_title('Mutual Information\n(Total Uncertainty)', fontsize=12, pad=10)
    ax6.axis('off')
    plt.colorbar(mi_plot, ax=ax6)
    
    # Sample predictions
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(segmentations[i, 0].cpu(), cmap='gray')
        ax.set_title(f'Sample {i+1}', fontsize=12, pad=10)
        ax.axis('off')
    
    plt.suptitle('VAE-UNet Segmentation Analysis', fontsize=14, y=0.95)
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Set up logging - FIX THE TYPO in the format string
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = get_args()
    
    # Convert patch_size=0 to None for full image mode
    original_patch_size = args.patch_size
    if args.patch_size == 0:
        args.patch_size = None
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetResNet(n_channels=3, n_classes=1, latent_dim=32, use_attention=args.use_attention)
    
    # Load the trained weights
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    model_path = Path(f'./checkpoints/{args.model}')
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Delete existing patch directories to force regeneration
    patch_dir = Path(f'./data/patches/test/{args.lesion_type}')
    if patch_dir.exists():
        logging.info(f"Removing existing patch directory: {patch_dir}")
        shutil.rmtree(patch_dir)
    
    # Display processing parameters
    is_full_image = args.patch_size is None
    logging.info(f"Requested inference mode: {'Full Image' if is_full_image else 'Patch-based'}")
    logging.info(f"Temperature: {args.temperature}, Samples: {args.samples}")
    if not is_full_image:
        logging.info(f"Patch size: {args.patch_size}, Overlap: {args.overlap}")
    logging.info(f"Scale: {args.scale}")
    logging.info(f"Enable dropout: {args.enable_dropout}")
    
    try:
        # First try to load with requested mode
        logging.info("Loading test dataset...")
        test_dataset = IDRIDDataset(
            base_dir='./data',
            split='test',
            scale=args.scale,
            patch_size=args.patch_size,
            lesion_type=args.lesion_type,

            max_images=args.max_images,
            skip_border_check=(not is_full_image)  # Be more permissive with border checks for patch mode
        )
        
        # If patch mode was requested but no patches were found, fall back to full image mode
        if not is_full_image and len(test_dataset) == 0:
            logging.warning(f"No valid patches found with patch_size={args.patch_size}. Falling back to full image mode.")
            
            # Clean up failed patch directory
            if patch_dir.exists():
                shutil.rmtree(patch_dir)
                
            # Fall back to full image mode
            args.patch_size = None
            is_full_image = True
            
            # Reload dataset in full image mode
            test_dataset = IDRIDDataset(
                base_dir='./data',
                split='test',
                scale=args.scale,
                patch_size=None,  # Force full image mode
                lesion_type=args.lesion_type,
                max_images=args.max_images,
                skip_border_check=True  # Skip border checks for full images
            )
            
            logging.info("Switched to full image mode automatically")
        
        if len(test_dataset) == 0:
            logging.error("No test samples found even in full image mode. Check if there are valid images with the specified lesion type.")
            exit(1)
            
        logging.info(f"Found {len(test_dataset)} test items")
        
        # Process unique image IDs rather than all patches
        processed_ids = set()
        
        for i in range(len(test_dataset)):
            # Get sample without loading to device
            sample = test_dataset[i]
            img_id = sample['img_id']
            
            # Skip if we've already processed this image ID
            if img_id in processed_ids:
                continue
                
            processed_ids.add(img_id)
            logging.info(f"Processing image ID: {img_id}")
            
            try:
                # Create visualization using the new function signature
                # Use this if you've added the new plot_reconstruction function:
                fig = plot_reconstruction(
                    model=model,
                    img_id=img_id,  # Pass img_id instead of img and mask
                    dataset=test_dataset,
                    num_samples=args.samples,
                    patch_size=args.patch_size,  # Pass patch_size instead of patch_dataset
                    overlap=args.overlap,
                    temperature=args.temperature,
                    enable_dropout=args.enable_dropout
                )
                
                # Save with descriptive filename
                # Use original_patch_size in filename to reflect what was requested
                mode_suffix = "full" if is_full_image else f"p{original_patch_size}"
                model_name = os.path.basename(args.model).replace('.pth', '')
                
                # Add fallback indicator if mode was changed
                if not is_full_image or original_patch_size == 0:
                    fallback_indicator = ""
                else:
                    fallback_indicator = "_fallback"
                    
                out_filename = f"{img_id}_{args.lesion_type}_T{args.temperature}_N{args.samples}_{mode_suffix}{fallback_indicator}_{model_name}.png"
                out_path = output_dir / out_filename
                fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
                plt.close(fig)
                
            except Exception as e:
                logging.error(f"Error processing image {img_id}: {e}")
                import traceback
                traceback.print_exc()  # Add full traceback for better debugging
                continue
                
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"High-resolution visualizations saved to {output_dir}!")




