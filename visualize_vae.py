import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch.nn.functional as F
import logging
from pathlib import Path
from torchvision import transforms
from utils.data_loading import IDRIDDataset, load_image  # Import from data_loading
from unet.unet_resnet import UNetResNet
import math
import argparse

def load_image_for_prediction(filename, scale=1.0):
    """Load and preprocess image for the model."""
    img = Image.open(filename).convert('RGB')
    dataset = IDRIDDataset(base_dir='data', split='test')
    
    # Use IDRIDDataset's preprocess method which returns [C, H, W] numpy array
    img_array = dataset.preprocess(img, scale=scale, is_mask=False)  # Returns [C, H, W]
    
    # Convert to tensor and add batch dimension: [C, H, W] -> [1, C, H, W]
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension
    print(f"Image tensor shape after preprocessing: {img_tensor.shape}")
    return img_tensor

def load_mask_for_prediction(filename, scale=1.0):
    """Load mask image."""
    mask = Image.open(filename).convert('L')
    dataset = IDRIDDataset(base_dir='data', split='test')
    
    # Use IDRIDDataset's preprocess method which returns [H, W] numpy array for masks
    mask_array = dataset.preprocess(mask, scale=scale, is_mask=True)  # Returns [H, W]
    
    # Convert to tensor and add batch and channel dimensions: [H, W] -> [1, 1, H, W]
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    print(f"Mask tensor shape after preprocessing: {mask_tensor.shape}")
    return mask_tensor

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
                
                # Initial projection
                x = model.z_initial(z_patch)
                
                # Decode with z injection at each stage
                for k, decoder_block in enumerate(model.decoder_blocks):
                    skip = features[-(k+2)] if k < len(features)-1 else None
                    x = decoder_block(x, skip, z_patch)
                
                # Final conv and sigmoid
                patch_pred = torch.sigmoid(model.final_conv(x))
                print(f"Patch prediction shape before resize: {patch_pred.shape}")
                
                # Resize prediction back to original patch size
                patch_pred = F.interpolate(patch_pred, size=(patch_h, patch_w), mode='bilinear', align_corners=True)
                print(f"Patch prediction shape after resize: {patch_pred.shape}")
                
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

def calculate_uncertainty_metrics(segmentations):
    """Calculate various uncertainty metrics from multiple segmentation samples.
    
    Args:
        segmentations: Tensor of shape [num_samples, B, C, H, W]
    
    Returns:
        Dictionary containing different uncertainty metrics
    """
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

def get_segmentation_distribution(model, img, num_samples=32, patch_size=512, overlap=100, temperature=1.0, enable_dropout=True):
    """Generate multiple segmentations using patch-based prediction with consistent latent sampling.
    
    Args:
        model: VAE-UNet model
        img: Input image tensor
        num_samples: Number of samples to generate
        patch_size: Size of patches to process
        overlap: Overlap between patches
        temperature: Controls the variance of the sampling (higher = more diverse samples)
        enable_dropout: Whether to enable dropout during inference for epistemic uncertainty
    """
    if enable_dropout:
        model.train()  # Enable dropout
    else:
        model.eval()
        
    device = img.device
    
    # Ensure input has batch dimension
    if img.dim() == 3:
        img = img.unsqueeze(0)
    print(f"Input shape after dimension adjustment: {img.shape}")
    
    # Get latent distribution parameters
    with torch.no_grad():
        mu, logvar = model.encode(img)
    print(f"Latent distribution shapes - mu: {mu.shape}, logvar: {logvar.shape}")
    
    # Initialize list to store segmentations
    segmentations = []
    
    # Generate multiple samples
    for i in range(num_samples):
        # Sample from latent space with increased variance
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        print(f"Sample {i} latent shape: {z.shape}")
        
        # Generate segmentation using patches
        seg = predict_with_patches(model, img, z, patch_size, overlap)
        print(f"Sample {i} segmentation shape: {seg.shape}")
        segmentations.append(seg)
    
    # Stack all segmentations
    segmentations = torch.cat(segmentations, dim=0)
    print(f"Final stacked segmentations shape: {segmentations.shape}")
    
    return segmentations, mu, logvar

def plot_reconstruction(model, img, mask, num_samples=32, temperature=1.0, enable_dropout=True):
    """Plot the input image, ground truth mask, and uncertainty analysis from multiple sampled reconstructions."""
    # Ensure correct dimensions
    if img.dim() != 4:
        raise ValueError(f"Expected img to have 4 dimensions [B, C, H, W], got shape {img.shape}")
    if mask.dim() != 4:
        raise ValueError(f"Expected mask to have 4 dimensions [B, C, H, W], got shape {mask.shape}")
        
    print(f"Input shapes - img: {img.shape}, mask: {mask.shape}")
    
    # Get multiple segmentations
    segmentations, mu, logvar = get_segmentation_distribution(
        model, img, num_samples=num_samples, temperature=temperature, enable_dropout=enable_dropout
    )
    print(f"Segmentations shape after distribution: {segmentations.shape}")
    
    # Calculate uncertainty metrics
    metrics = calculate_uncertainty_metrics(segmentations)
    
    # Create figure with subplots
    plt.rcParams['figure.dpi'] = 300  # Higher DPI for better quality
    fig = plt.figure(figsize=(20, 16))  # Increased figure size for more plots
    gs = gridspec.GridSpec(3, 3, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)  # Increased spacing between plots
    
    # Original image (take first batch)
    ax1 = fig.add_subplot(gs[0, 0])
    img_display = img[0].cpu().clone()  # [C, H, W]
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = img_display * std + mean
    img_display = torch.clamp(img_display, 0, 1)  # Ensure values are in [0, 1]
    
    ax1.imshow(img_display.permute(1, 2, 0).numpy(), interpolation='lanczos')
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
    parser.add_argument('--lesion_type', type=str, required=True, default='EX', choices=['EX', 'HE', 'MA','OD'], help='Lesion type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling (default: 1.0)')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for prediction (default: 512)')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for resizing (default: 1.0)')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples for ensemble prediction (default: 1)')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args() 
    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetResNet(n_channels=3, n_classes=1, latent_dim=32)
    
    # Load the trained weights
    model_path = Path('./checkpoints/best_model.pth')
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Process a few test images
    test_dir = Path('./data/imgs/test')
    mask_dir = Path(f"./data/masks/test/{args.lesion_type}")
    
    # Set scale factor to reduce memory usage
    scale = 0.5  # Reduce image size by half
    
    # Get all test images
    image_files = sorted(test_dir.glob('*.jpg'))[:3]  # Process first 3 images
    
    for i, img_path in enumerate(image_files):
        # Load image and corresponding mask with scaling
        img = load_image_for_prediction(img_path, scale=scale).to(device)
        mask_path = mask_dir / f"{img_path.stem}_{args.lesion_type}.tif"
        mask = load_mask_for_prediction(mask_path, scale=scale).to(device)
        
        # Print memory usage before processing
        if torch.cuda.is_available():
            print(f"GPU Memory before processing image {i+1}:")
            print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        # Create visualization with high DPI
        fig = plot_reconstruction(model, img, mask, num_samples=args.samples, temperature=args.temperature)
        fig.savefig(f'vae_analysis_{i+1}.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        
        # Clear some memory
        torch.cuda.empty_cache()
    
    print("High-resolution visualizations saved as PNG files!")
