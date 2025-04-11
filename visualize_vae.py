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
from utils.vae_utils import generate_predictions, encode_images
import gc
import psutil

def track_memory(func):
    """Decorator to track memory usage."""
    def wrapper(*args, **kwargs):
        # Before function call
        gc.collect()
        torch.cuda.empty_cache()
        
        gpu_mem_before = torch.cuda.memory_allocated() / 1024**2
        ram_before = psutil.Process().memory_info().rss / 1024**2
        
        logging.info(f"Memory before {func.__name__}: GPU: {gpu_mem_before:.1f} MB, RAM: {ram_before:.1f} MB")
        
        # Call function
        result = func(*args, **kwargs)
        
        # After function call
        gc.collect()
        torch.cuda.empty_cache()
        
        gpu_mem_after = torch.cuda.memory_allocated() / 1024**2
        ram_after = psutil.Process().memory_info().rss / 1024**2
        
        logging.info(f"Memory after {func.__name__}: GPU: {gpu_mem_after:.1f} MB, RAM: {ram_after:.1f} MB")
        logging.info(f"Memory change: GPU: {gpu_mem_after-gpu_mem_before:.1f} MB, RAM: {ram_after-ram_before:.1f} MB")
        
        return result
    return wrapper

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
        output = model.final_conv(x)
        
        # Add interpolation to match input size
        output = F.interpolate(output, size=img.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply sigmoid after resizing
        output = torch.sigmoid(output)
        
        print(f"Full image prediction shape: {output.shape}")
        
    return output
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
        patch_dataset=patch_dataset, temperature=temperature, enable_dropout=enable_dropout, batch_size=args.batch_size
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
    parser.add_argument('--max_images', type=int, default=5, help='Maximum number of test images to process')
    parser.add_argument('--temperature-range', type=float, nargs='+',
                        default=None, help='Multiple temperatures to compare [0.5 1.0 2.0 3.0]')
    parser.add_argument('--ensemble', action='store_true',
                        help='Create ensemble prediction from multiple temperatures')
    parser.add_argument('--weighted-ensemble', action='store_true',
                        help='Use weighted temperature ensemble (weights favor T=1)')
    parser.add_argument('--samples-per-temp', type=int, default=5,
                        help='Number of samples per temperature for ensembling')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for patch processing (default: 4)')
    parser.set_defaults(use_attention=True, enable_dropout=False)
    parser.add_argument('--latent-injection', type=str, default='all', 
                        choices=['all', 'first', 'last', 'bottleneck', 'none'],
                        help='Latent space injection strategy: inject at all levels, only first, only last, only bottleneck, or none')
    return parser.parse_args()

def predict_with_patches(model, img, z, patch_size=512, overlap=None, batch_size=4):
    """Predict segmentation using patches with memory optimization and batch processing."""
    model.eval()
    device = img.device
    B, C, H, W = img.shape
    
    # Calculate adaptive overlap if not specified
    if overlap is None:
        # Make overlap proportional to patch size (20% of patch size, min 32px, max 128px)
        overlap = max(min(int(patch_size * 0.2), 128), 32)
    
    # Calculate number of patches needed
    stride = patch_size - overlap
    n_patches_h = math.ceil((H - overlap) / stride)
    n_patches_w = math.ceil((W - overlap) / stride)
    total_patches = n_patches_h * n_patches_w
    
    # Debug prints to verify calculations
    print(f"Input image shape: {img.shape}")
    print(f"Patch size: {patch_size}, Overlap: {overlap}, Stride: {stride}")
    print(f"Calculated patches: {n_patches_h} x {n_patches_w} = {total_patches}")
    print(f"Processing with batch size: {batch_size}")
    
    # Initialize output mask with zeros on CPU to save memory
    output_mask = torch.zeros((B, 1, H, W), dtype=torch.float32)
    weight_mask = torch.zeros((B, 1, H, W), dtype=torch.float32)
    
    # Create all patch coordinates first
    patch_info = []
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
                
            patch_info.append((start_h, end_h, start_w, end_w))
    
    # Process patches in batches
    with torch.no_grad():
        for batch_idx in range(0, len(patch_info), batch_size):
            batch_patches = []
            batch_coords = []
            
            # Collect batch of patches
            for idx in range(batch_idx, min(batch_idx + batch_size, len(patch_info))):
                start_h, end_h, start_w, end_w = patch_info[idx]
                patch = img[:, :, start_h:end_h, start_w:end_w]
                batch_patches.append(patch)
                batch_coords.append((start_h, end_h, start_w, end_w))
            
            # Stack patches into a batch
            if len(batch_patches) > 1:
                # Create padded tensors if patches have different sizes
                max_h = max(p.shape[2] for p in batch_patches)
                max_w = max(p.shape[3] for p in batch_patches)
                padded_patches = []
                
                for p in batch_patches:
                    if p.shape[2] < max_h or p.shape[3] < max_w:
                        pad_h = max_h - p.shape[2]
                        pad_w = max_w - p.shape[3]
                        padded = F.pad(p, (0, pad_w, 0, pad_h))
                        padded_patches.append(padded)
                    else:
                        padded_patches.append(p)
                        
                stacked_patches = torch.cat(padded_patches, dim=0)
            else:
                stacked_patches = batch_patches[0]
            
            try:
                # Process batch through encoder
                features_batch = model.encoder(stacked_patches)
                x_enc_batch = features_batch[-1]
                
                # Interpolate z to match encoder output size for batch
                if len(batch_patches) > 1:
                    z_batch = z.expand(len(batch_patches), -1, -1, -1)
                else:
                    z_batch = z
                    
                z_resized = F.interpolate(z_batch, size=x_enc_batch.shape[2:], 
                                         mode='bilinear', align_corners=True)
                
                # Initial projection
                x = model.z_initial(z_resized)
                
                # Decode with z injection
                for k, decoder_block in enumerate(model.decoder_blocks):
                    skip = features_batch[-(k+2)] if k < len(features_batch)-1 else None
                    x = decoder_block(x, skip, z_resized)
                
                # Final conv and sigmoid
                batch_preds = torch.sigmoid(model.final_conv(x))
                
                # Process each prediction in the batch
                for idx, ((start_h, end_h, start_w, end_w), patch) in enumerate(zip(batch_coords, batch_patches)):
                    if len(batch_patches) > 1:
                        patch_pred = batch_preds[idx:idx+1]
                    else:
                        patch_pred = batch_preds
                        
                    # Resize if needed
                    patch_h, patch_w = patch.shape[2:]
                    if patch_pred.shape[2:] != (patch_h, patch_w):
                        patch_pred = F.interpolate(patch_pred, size=(patch_h, patch_w), 
                                                mode='bilinear', align_corners=True)
                    
                    # Create weight mask for blending
                    weight = torch.ones_like(patch_pred)
                    
                    # Apply tapering for smooth blending
                    for axis in [2, 3]:
                        if patch_pred.shape[axis] > 2 * overlap:
                            ramp = torch.linspace(0, 1, overlap, device=device)
                            
                            # Apply ramp to edges
                            i, j = divmod(idx + batch_idx, n_patches_w)
                            
                            # Left/top edge
                            if (i > 0 and axis == 2) or (j > 0 and axis == 3):
                                if axis == 2:
                                    weight[:, :, :overlap, :] *= ramp.view(-1, 1)
                                else:
                                    weight[:, :, :, :overlap] *= ramp.view(-1)
                                    
                            # Right/bottom edge
                            if ((i < n_patches_h - 1 and axis == 2) or
                                (j < n_patches_w - 1 and axis == 3)):
                                if axis == 2:
                                    weight[:, :, -overlap:, :] *= (1 - ramp).view(-1, 1)
                                else:
                                    weight[:, :, :, -overlap:] *= (1 - ramp).view(-1)
                    
                    # Move to CPU immediately to save GPU memory
                    patch_pred_cpu = patch_pred.cpu()
                    weight_cpu = weight.cpu()
                    
                    # Add to output mask
                    output_mask[:, :, start_h:end_h, start_w:end_w] += patch_pred_cpu * weight_cpu
                    weight_mask[:, :, start_h:end_h, start_w:end_w] += weight_cpu
                
            except RuntimeError as e:
                # Handle out of memory errors
                torch.cuda.empty_cache()
                if 'out of memory' in str(e) and batch_size > 1:
                    logging.warning(f"Out of memory with batch_size={batch_size}. Falling back to single patch processing.")
                    # Process each patch individually
                    for idx, (start_h, end_h, start_w, end_w) in enumerate(batch_coords):
                        # Extract single patch
                        patch = img[:, :, start_h:end_h, start_w:end_w]
                        # Use single patch prediction with the same method
                        result = predict_single_patch(model, patch, z, device, overlap, 
                                                    (n_patches_h, n_patches_w), batch_idx + idx)
                        
                        if result is not None:
                            patch_pred, weight = result
                            # Move to CPU and add to output
                            output_mask[:, :, start_h:end_h, start_w:end_w] += patch_pred.cpu()
                            weight_mask[:, :, start_h:end_h, start_w:end_w] += weight.cpu()
                else:
                    logging.error(f"Error processing patch batch: {e}")
                    raise e
            
            # Clean up batch memory
            del stacked_patches, features_batch, x_enc_batch, z_resized
            if 'batch_preds' in locals():
                del batch_preds
            torch.cuda.empty_cache()
    
    # Average overlapping regions
    output_mask = output_mask / (weight_mask + 1e-8)
    
    # Move back to GPU for further processing
    result = output_mask.to(device)
    
    # Clean up
    del output_mask, weight_mask
    torch.cuda.empty_cache()
    
    return result

def predict_single_patch(model, patch, z, device, overlap, patch_grid, patch_idx):
    """Helper function to predict a single patch for memory fallback."""
    try:
        n_patches_h, n_patches_w = patch_grid
        i, j = divmod(patch_idx, n_patches_w)
        
        # Get encoder features for this patch
        features = model.encoder(patch)
        x_enc = features[-1]
        
        # Interpolate z to match encoder output size
        z_patch = F.interpolate(z, size=x_enc.shape[2:], mode='bilinear', align_corners=True)
        
        # Initial projection
        x = model.z_initial(z_patch)
        
        # Decode with z injection at each stage
        for k, decoder_block in enumerate(model.decoder_blocks):
            skip = features[-(k+2)] if k < len(features)-1 else None
            x = decoder_block(x, skip, z_patch)
        
        # Final conv and sigmoid
        patch_pred = torch.sigmoid(model.final_conv(x))
        
        # Resize if needed
        patch_h, patch_w = patch.shape[2:]
        if patch_pred.shape[2:] != (patch_h, patch_w):
            patch_pred = F.interpolate(patch_pred, size=(patch_h, patch_w), 
                                      mode='bilinear', align_corners=True)
                                      
        # Create weight mask for blending
        weight = torch.ones_like(patch_pred)
        
        # Apply tapering at edges for smooth blending
        for axis in [2, 3]:
            if patch_pred.shape[axis] > 2*overlap:
                ramp = torch.linspace(0, 1, overlap, device=device)
                
                # Left/top edge
                if (i > 0 and axis == 2) or (j > 0 and axis == 3):
                    if axis == 2:
                        weight[:, :, :overlap, :] *= ramp.view(-1, 1)
                    else:
                        weight[:, :, :, :overlap] *= ramp.view(-1)
                        
                # Right/bottom edge
                if ((i < n_patches_h - 1 and axis == 2) or
                    (j < n_patches_w - 1 and axis == 3)):
                    if axis == 2:
                        weight[:, :, -overlap:, :] *= (1 - ramp).view(-1, 1)
                    else:
                        weight[:, :, :, -overlap:] *= (1 - ramp).view(-1)
        
        return patch_pred, weight
        
    except RuntimeError:
        torch.cuda.empty_cache()
        logging.error(f"Failed to process patch {patch_idx} even in single mode")
        return None

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
        # Log the dataset patch size to check if correct
        logging.info(f"Stitching patches with dataset patch size: {dataset.patch_size}")
        
        # Find the size of the original image from the patch coordinates
        max_h = max(y + dataset.patch_size for y, _ in coords)
        max_w = max(x + dataset.patch_size for _, x in coords)
        
        # Create tensors to hold the full image and mask
        full_img = torch.zeros((3, max_h, max_w), dtype=torch.float32)
        full_mask = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        weight = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        
        # Calculate adaptive overlap for blending
        patch_size = dataset.patch_size
        overlap = max(min(int(patch_size * 0.2), 128), 32)  # 20% of patch size, bounded
        
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

def get_segmentation_distribution(model, img_id, dataset, num_samples=32, 
                                 patch_size=None, overlap=None, 
                                 temperature=1.0, enable_dropout=True,
                                 batch_size=4):
    """Generate multiple segmentations using either full image or patch-based prediction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the full image and mask
    img, mask, original_shape = get_image_and_mask(dataset, img_id)
    img = img.unsqueeze(0).to(device)  # Add batch dimension [1, C, H, W]
    print(f"Full image shape for segmentation: {img.shape}")
    
    # Log the actual patch size being used to check for overrides
    logging.info(f"Requested patch size: {patch_size}")
    if hasattr(dataset, 'patch_size'):
        logging.info(f"Dataset patch size: {dataset.patch_size}")
    
    # Set model mode based on dropout preference
    if enable_dropout:
        model.train()  # Enable dropout
    else:
        model.eval()
    
    # Get latent distribution parameters
    with torch.no_grad():
        mu, logvar = encode_images(model, img)
    print(f"Latent distribution shapes - mu: {mu.shape}, logvar: {logvar.shape}")
    
    # Initialize list to store segmentations
    B, H, W = 1, img.shape[2], img.shape[3]
    segmentations_cpu = torch.zeros((num_samples, 1, H, W), dtype=torch.float32)
    
    # Generate one sample at a time to save memory
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Sample from latent space
        with torch.no_grad():
            std = torch.exp(0.5 * logvar) * temperature
            eps = torch.randn_like(std)
            z = mu + eps * std
            z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
            
            # Choose prediction method based on patch size
            if patch_size is not None and patch_size > 0:
                # Calculate adaptive overlap if not specified
                if overlap is None:
                    # Make overlap proportional to patch size (20% of patch size)
                    overlap = max(min(int(patch_size * 0.2), 128), 32)
                    
                seg = predict_with_patches(model, img, z, patch_size, overlap, batch_size)
            else:
                seg = predict_full_image(model, img, z)
            
            # Move result to CPU immediately
            segmentations_cpu[i] = seg.cpu()
            
            # Free memory
            del seg, eps
            torch.cuda.empty_cache()
    
    print(f"Final stacked segmentations shape: {segmentations_cpu.shape}")
    
    # Return with mask on device for further processing
    return segmentations_cpu.to(device), mask.unsqueeze(0).to(device), mu, logvar

def plot_reconstruction(model, img_id, dataset, num_samples=32, patch_size=None, overlap=100, temperature=1.0, enable_dropout=True, batch_size=4):
    """Plot image, mask, and uncertainty analysis using either full image or patch-based prediction."""
    # Get multiple segmentations with either patch-based or full image mode
    segmentations, mask, mu, logvar = get_segmentation_distribution(
        model, img_id, dataset=dataset, num_samples=num_samples,
        patch_size=patch_size, overlap=overlap, 
        temperature=temperature, enable_dropout=enable_dropout,
        batch_size=batch_size
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

def visualize_temperature_sampling(model, image, mask=None, 
                                  temperatures=[0.5, 1.0, 2.0, 3.0],
                                  samples_per_temp=10,
                                  patch_size=512,   
                                  overlap=100,
                                  batch_size=4): 
    """Memory-optimized visualization processing one temperature at a time."""
    device = image.device
    
    # Create figure first
    plt.figure(figsize=(15, 8))
    
    # Plot original image (downsampled)
    with torch.no_grad():
        # Create a smaller version for display
        if image.shape[2] > 512 or image.shape[3] > 512:
            scale = 512 / max(image.shape[2], image.shape[3])
            display_size = (int(image.shape[2] * scale), int(image.shape[3] * scale))
            img_display = F.interpolate(image, size=display_size, mode='bilinear', align_corners=False)
        else:
            img_display = image
            
        plt.subplot(2, len(temperatures) + 1, 1)
        plt.imshow(img_display[0].cpu().permute(1, 2, 0).numpy())
        plt.title("Original Image")
        plt.axis('off')
        # Free immediately
        del img_display
    
    # Plot ground truth if available
    if mask is not None:
        if mask.shape[2] > 512 or mask.shape[3] > 512:
            scale = 512 / max(mask.shape[2], mask.shape[3])
            display_size = (int(mask.shape[2] * scale), int(mask.shape[3] * scale))
            mask_display = F.interpolate(mask, size=display_size, mode='nearest')
        else:
            mask_display = mask
            
        plt.subplot(2, len(temperatures) + 1, len(temperatures) + 2)
        plt.imshow(mask_display[0, 0].cpu().numpy(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        del mask_display
    
    # Get latent distribution once to reuse
    with torch.no_grad():
        mu, logvar = encode_images(model, image)
    
    # Process each temperature separately
    for temp_idx, temp in enumerate(temperatures):
        logging.info(f"Processing temperature {temp}")
        
        # Process on CPU to save memory
        B, C, H, W = image.shape
        mean_pred_cpu = torch.zeros((1, 1, H, W), dtype=torch.float32)
        sq_pred_cpu = torch.zeros((1, 1, H, W), dtype=torch.float32)
        
        # Process one sample at a time
        for s in range(samples_per_temp):
            logging.info(f"  Sample {s+1}/{samples_per_temp}")
            
            with torch.no_grad():
                # Sample with temperature
                std = torch.exp(0.5 * logvar) * temp
                eps = torch.randn_like(std)
                z = mu + eps * std
                z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
                
                # Use correct prediction method
                if patch_size is not None and patch_size > 0:
                    pred = predict_with_patches(model, image, z, patch_size, overlap, batch_size)
                else:
                    pred = predict_full_image(model, image, z)
                
                # Move to CPU immediately and accumulate
                pred_cpu = pred.cpu()
                mean_pred_cpu += pred_cpu
                sq_pred_cpu += (pred_cpu ** 2)
                
                # Free GPU memory
                del pred, eps, pred_cpu
                torch.cuda.empty_cache()
        
        # Calculate mean and std
        mean_pred = mean_pred_cpu / samples_per_temp
        var_pred = (sq_pred_cpu / samples_per_temp) - (mean_pred ** 2)
        std_pred = torch.sqrt(torch.clamp(var_pred, min=1e-8))
        
        # Create smaller versions for display
        if mean_pred.shape[2] > 512 or mean_pred.shape[3] > 512:
            scale = 512 / max(mean_pred.shape[2], mean_pred.shape[3])
            display_size = (int(mean_pred.shape[2] * scale), int(mean_pred.shape[3] * scale))
            mean_display = F.interpolate(mean_pred, size=display_size, mode='bilinear', align_corners=False)
            std_display = F.interpolate(std_pred, size=display_size, mode='bilinear', align_corners=False)
        else:
            mean_display = mean_pred
            std_display = std_pred
        
        # Plot
        plt.subplot(2, len(temperatures) + 1, temp_idx + 2)
        plt.imshow(mean_display[0, 0].numpy(), cmap='gray')
        plt.title(f"T={temp}\nMean")
        plt.axis('off')
        
        plt.subplot(2, len(temperatures) + 1, temp_idx + len(temperatures) + 3)
        plt.imshow(std_display[0, 0].numpy(), cmap='hot')
        plt.title(f"T={temp}\nUncertainty")
        plt.axis('off')
        
        # Free memory
        del mean_pred, var_pred, std_pred, mean_display, std_display
        del mean_pred_cpu, sq_pred_cpu
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()
    
    plt.tight_layout()
    return plt.gcf()

@track_memory
def generate_and_compare_ensemble(model, image, mask, temperatures=[0.5, 1.0, 2.0, 3.0], 
                                samples_per_temp=5, weighted=True, patch_size=512, overlap=100, batch_size=4):
    """Memory-optimized ensemble visualization that processes one temperature at a time."""
    device = image.device
    B, C, H, W = image.shape
    
    # Create smaller versions for display
    if image.shape[2] > 512 or image.shape[3] > 512:
        scale = 512 / max(image.shape[2], image.shape[3])
        display_size = (int(image.shape[2] * scale), int(image.shape[3] * scale))
        image_vis = F.interpolate(image.cpu(), size=display_size, mode='bilinear', align_corners=False)
        mask_vis = F.interpolate(mask.cpu(), size=display_size, mode='nearest')
    else:
        image_vis = image.cpu()
        mask_vis = mask.cpu()
    
    # Log dimensions for debugging
    logging.info(f"Input image shape: {image.shape}, Mask shape: {mask.shape}")
    
    # Create visualization figure early
    fig = plt.figure(figsize=(15, 10))
    
    # Plot original image and ground truth
    plt.subplot(2, len(temperatures) + 1, 1)
    plt.imshow(image_vis[0].permute(1, 2, 0).numpy())
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, len(temperatures) + 1, 2)
    plt.imshow(mask_vis[0, 0].numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # Free memory for visualizations we no longer need
    del image_vis, mask_vis
    
    # Get latent distribution once to reuse
    with torch.no_grad():
        mu, logvar = encode_images(model, image)
    
    # Temperature prediction scores and downsampled predictions
    dice_scores = []
    temp_pred_small = {}
    
    # Generate individual temperature predictions one at a time
    for temp_idx, temp in enumerate(temperatures):
        logging.info(f"Processing temperature {temp}")
        
        # Generate prediction for this temperature
        temp_pred_cpu = torch.zeros((1, 1, H, W), dtype=torch.float32)
        
        for s in range(samples_per_temp):
            logging.info(f"  Sample {s+1}/{samples_per_temp}")
            
            with torch.no_grad():
                # Sample latent vector
                std = torch.exp(0.5 * logvar) * temp
                eps = torch.randn_like(std)
                z = mu + eps * std
                z = z.unsqueeze(-1).unsqueeze(-1)
                
                # Generate prediction using appropriate method
                if patch_size is not None and patch_size > 0:
                    pred = predict_with_patches(model, image, z, patch_size, overlap, batch_size)
                else:
                    pred = predict_full_image(model, image, z)
                    
                # Accumulate on CPU
                temp_pred_cpu += pred.cpu()
                
                # Free memory
                del pred, eps, z
                torch.cuda.empty_cache()
        
        # Average samples
        temp_pred_cpu /= samples_per_temp
        
        # Move back to device for Dice calculation
        temp_pred = temp_pred_cpu.to(device)
        
        # Calculate dice score
        dice = (2.0 * ((temp_pred > 0.5) & (mask > 0.5)).sum().float()) / ((temp_pred > 0.5).sum() + (mask > 0.5).sum() + 1e-8)
        dice_scores.append(dice.item())
        logging.info(f"  Dice score: {dice.item():.4f}")
        
        # Create small version for display
        if H > 512 or W > 512:
            scale = 512 / max(H, W)
            display_size = (int(H * scale), int(W * scale))
            temp_pred_small[temp] = F.interpolate(temp_pred_cpu, size=display_size, mode='bilinear', align_corners=False)
        else:
            temp_pred_small[temp] = temp_pred_cpu
        
        # Free full-size prediction
        del temp_pred, temp_pred_cpu
        torch.cuda.empty_cache()
    
    # Generate ensemble prediction on CPU to save memory
    logging.info("Generating ensemble prediction")
    ensemble_pred_cpu = torch.zeros((1, 1, H, W), dtype=torch.float32)
    
    # Initialize weights
    if weighted:
        weights = [1.0 / (abs(temp - 1.0) + 0.5) for temp in temperatures]
    else:
        weights = [1.0] * len(temperatures)
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Process one temperature at a time
    for temp_idx, temp in enumerate(temperatures):
        logging.info(f"Adding temperature {temp} to ensemble with weight {weights[temp_idx]:.4f}")
        
        # Generate prediction for this temperature again
        temp_pred_cpu = torch.zeros((1, 1, H, W), dtype=torch.float32)
        
        for s in range(samples_per_temp):
            logging.info(f"  Sample {s+1}/{samples_per_temp}")
            
            with torch.no_grad():
                # Sample latent vector
                std = torch.exp(0.5 * logvar) * temp
                eps = torch.randn_like(std)
                z = mu + eps * std
                z = z.unsqueeze(-1).unsqueeze(-1)
                
                # Generate prediction
                if patch_size is not None and patch_size > 0:
                    pred = predict_with_patches(model, image, z, patch_size, overlap, batch_size)
                else:
                    pred = predict_full_image(model, image, z)
                    
                # Accumulate on CPU
                temp_pred_cpu += pred.cpu()
                
                # Free memory
                del pred, eps, z
                torch.cuda.empty_cache()
        
        # Average samples and add weighted contribution to ensemble
        temp_pred_cpu /= samples_per_temp
        ensemble_pred_cpu += temp_pred_cpu * weights[temp_idx]
        
        # Free memory
        del temp_pred_cpu
        torch.cuda.empty_cache()
    
    # Move ensemble to device for dice calculation
    ensemble_pred = ensemble_pred_cpu.to(device)
    
    # Calculate ensemble dice score
    ensemble_dice = (2.0 * ((ensemble_pred > 0.5) & (mask > 0.5)).sum().float()) / ((ensemble_pred > 0.5).sum() + (mask > 0.5).sum() + 1e-8)
    ensemble_dice = ensemble_dice.item()
    logging.info(f"Ensemble dice score: {ensemble_dice:.4f}")
    
    # Create small version for display
    if H > 512 or W > 512:
        scale = 512 / max(H, W)
        display_size = (int(H * scale), int(W * scale))
        ensemble_vis = F.interpolate(ensemble_pred_cpu, size=display_size, mode='bilinear', align_corners=False)
    else:
        ensemble_vis = ensemble_pred_cpu
    
    # Free full-size ensemble
    del ensemble_pred, ensemble_pred_cpu
    torch.cuda.empty_cache()
    
    # Plot ensemble prediction
    plt.subplot(2, len(temperatures) + 1, 3)
    plt.imshow(ensemble_vis[0, 0].numpy(), cmap='gray')
    plt.title(f"Ensemble\nDice: {ensemble_dice:.4f}")
    plt.axis('off')
    
    # Plot individual temperature predictions
    for i, temp in enumerate(temperatures):
        plt.subplot(2, len(temperatures) + 1, len(temperatures) + i + 2)
        plt.imshow(temp_pred_small[temp][0, 0].numpy(), cmap='gray')
        plt.title(f"T={temp}\nDice: {dice_scores[i]:.4f}")
        plt.axis('off')
    
    # Bar chart for dice scores
    plt.subplot(2, len(temperatures) + 1, len(temperatures) + 1)
    bars = plt.bar(['Ensemble'] + [f'T={t}' for t in temperatures], [ensemble_dice] + dice_scores)
    bars[0].set_color('green')  # Highlight ensemble
    plt.ylim(0, 1)
    plt.title("Dice Score Comparison")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Free remaining variables
    del mu, logvar, temp_pred_small, ensemble_vis
    torch.cuda.empty_cache()
    
    return fig

def generate_ensemble_prediction(model, image, temps=[0.5, 1.0, 2.0, 3.0], 
                               samples_per_temp=5, weighted=True,
                               patch_size=512,    # Add this parameter
                               overlap=100,
                               batch_size=4):
    """Memory-efficient ensemble prediction using patches directly."""
    device = image.device
    B, C, H, W = image.shape
 
    
    logging.info(f"Generating ensemble prediction with temperatures {temps}")
    
    # Initialize ensemble prediction tensor
    ensemble_pred = torch.zeros((1, 1, H, W), device=device)
    
    # Initialize weights
    temp_weights = []
    for temp in temps:
        if weighted:
            # Weight based on temperature - favoring middle range (around 1.0)
            weight = 1.0 / (abs(temp - 1.0) + 0.5)
        else:
            weight = 1.0
        temp_weights.append(weight)
    
    # Normalize weights
    temp_weights = torch.tensor(temp_weights, device=device)
    temp_weights = temp_weights / temp_weights.sum()
    
    # Process one temperature at a time to save memory
    for temp_idx, temp in enumerate(temps):
        logging.info(f"Processing temperature {temp} with weight {temp_weights[temp_idx].item():.4f}")
        
        # Get latent distribution parameters
        with torch.no_grad():
            mu, logvar = encode_images(model, image)
            
        # Average predictions for this temperature
        temp_pred = torch.zeros((1, 1, H, W), device=device)
        
        # Generate multiple samples for this temperature
        for s in range(samples_per_temp):
            logging.info(f"  Sample {s+1}/{samples_per_temp}")
            
            # Sample from latent space
            with torch.no_grad():
                std = torch.exp(0.5 * logvar) * temp
                eps = torch.randn_like(std)
                z = mu + eps * std
                z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
                
                # Generate prediction using patches
                pred = predict_with_patches(model, image, z, patch_size, overlap, batch_size)
                
                # Add to temperature prediction
                temp_pred += pred
                
                # Free memory
                del pred
                torch.cuda.empty_cache()
        
        # Average the predictions for this temperature
        temp_pred = temp_pred / samples_per_temp
        
        # Add weighted prediction to ensemble
        ensemble_pred += temp_pred * temp_weights[temp_idx]
        
        # Free memory
        del temp_pred, mu, logvar
        torch.cuda.empty_cache()
    
    return ensemble_pred

# Downsample large images for display to avoid memory issues
def downsample_for_display(image_tensor, max_size=512):
    """Downsample large tensor for display purposes with robust dimension handling"""
    if image_tensor is None:
        return None
    
    # Ensure tensor has 4 dimensions (N, C, H, W) for interpolation
    original_shape = image_tensor.shape
    
    # Fix tensor dimensions based on original shape
    if len(original_shape) == 2:  # H, W
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add N, C dimensions
    elif len(original_shape) == 3:
        # Could be either [1, H, W] or [C, H, W]
        image_tensor = image_tensor.unsqueeze(0)  # Add N dimension
    
    # Now check if resizing is needed
    if image_tensor.shape[-1] > max_size or image_tensor.shape[-2] > max_size:
        h, w = image_tensor.shape[-2], image_tensor.shape[-1]
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Downsample
        result = F.interpolate(
            image_tensor, 
            size=(new_h, new_w),
            mode='bilinear', 
            align_corners=False
        )
        
        # Restore original dimension structure
        if len(original_shape) == 2:
            result = result.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            result = result.squeeze(0)
        
        return result
    
    # Restore original dimension structure if no resizing was done
    if len(original_shape) != len(image_tensor.shape):
        if len(original_shape) == 2:
            image_tensor = image_tensor.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            image_tensor = image_tensor.squeeze(0)
    
    return image_tensor

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
    model = UNetResNet(n_channels=3, n_classes=1, latent_dim=32, use_attention=args.use_attention, latent_injection=args.latent_injection)  
    
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
    logging.info(f"Batch size: {args.batch_size}")
    
    # Base output directory
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model-specific subdirectory
    model_name = os.path.basename(args.model).replace('.pth', '')
    model_dir = base_output_dir / model_name
    
    # Create lesion type subdirectory
    lesion_dir = model_dir / args.lesion_type
    
    # Create patch size subdirectory
    mode_suffix = "full" if is_full_image else f"p{original_patch_size}"
    patch_dir = lesion_dir / mode_suffix
    
    # Create temperature subdirectory
    temp_dir = patch_dir / f"T{args.temperature}"
    
    # Create samples subdirectory
    samples_dir = temp_dir / f"N{args.samples}"
    
    # Ensure all directories exist
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # First try to load with requested mode
        logging.info("Loading test dataset...")
        test_dataset = IDRIDDataset(
            base_dir='./data',
            split='test',
            scale=args.scale,
            patch_size=args.patch_size,  # This is where the patch size is first passed
            lesion_type=args.lesion_type,
            max_images=args.max_images,
            skip_border_check=(not is_full_image)
        )
        
        # Check patch size just after dataset is loaded
        if hasattr(test_dataset, 'patch_size'):
            logging.info(f"Dataset's internal patch_size: {test_dataset.patch_size}")
        else:
            logging.info("Dataset has no internal patch_size attribute")
            
        # Make sure the dataset's patch size matches what we requested
        if not is_full_image and hasattr(test_dataset, 'patch_size') and test_dataset.patch_size != args.patch_size:
            logging.warning(f"Dataset patch size ({test_dataset.patch_size}) doesn't match requested patch size ({args.patch_size})")
            # Force the dataset to use our patch size - may need to reload the dataset
            logging.info("Forcing dataset to use the requested patch size")
            test_dataset.patch_size = args.patch_size
        
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
                    enable_dropout=args.enable_dropout,
                    batch_size=args.batch_size
                )
                
                # Temperature comparison if requested
                if args.temperature_range:
                    logging.info(f"Comparing temperatures: {args.temperature_range}")
                    
                    # Get image and mask
                    raw_img, mask, _ = get_image_and_mask(test_dataset, img_id)
                    image = raw_img.unsqueeze(0).to(device)
                    mask = mask.unsqueeze(0).to(device)
                    
                    # Generate temperature comparison
                    temp_fig = visualize_temperature_sampling(
                        model=model,
                        image=image,
                        mask=mask,
                        temperatures=args.temperature_range,
                        samples_per_temp=args.samples_per_temp,
                        patch_size=args.patch_size,
                        overlap=args.overlap,
                        batch_size=args.batch_size
                    )
                    
                    # Save temperature comparison
                    temp_path = samples_dir / f"{img_id}_temp_comparison.png"
                    temp_fig.savefig(temp_path, dpi=300, bbox_inches='tight')
                    plt.close(temp_fig)
                    logging.info(f"Saved temperature comparison to {temp_path}")
                
                    # Generate ensemble if requested
                    if args.ensemble:
                        ensemble_fig = generate_and_compare_ensemble(
                            model=model,
                            image=image,
                            mask=mask,
                            temperatures=args.temperature_range,
                            samples_per_temp=args.samples_per_temp,
                            weighted=args.weighted_ensemble,
                            patch_size=args.patch_size,
                            overlap=args.overlap,
                            batch_size=args.batch_size
                        )
                        
                        # Save ensemble comparison
                        ensemble_path = samples_dir / f"{img_id}_ensemble.png"
                        ensemble_fig.savefig(ensemble_path, dpi=300, bbox_inches='tight')
                        plt.close(ensemble_fig)
                        logging.info(f"Saved ensemble visualization to {ensemble_path}")
                
                # Create timestamp for unique filename
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save with timestamp in the filename in the hierarchical directory
                out_filename = f"{img_id}_{timestamp}.png"
                out_path = samples_dir / out_filename
                fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
                plt.close(fig)
                
                logging.info(f"Saved visualization to {out_path}")
                
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
    
    # Print the folder structure at the end
    print(f"\nHigh-resolution visualizations saved in the following structure:")
    print(f"{base_output_dir}/")
    print(f"└── {model_name}/")
    print(f"    └── {args.lesion_type}/")
    print(f"        └── {mode_suffix}/")
    print(f"            └── T{args.temperature}/")
    print(f"                └── N{args.samples}/")
    print(f"                    └── [image_id]_[timestamp].png")




