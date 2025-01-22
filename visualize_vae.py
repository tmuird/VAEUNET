import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import logging
from pathlib import Path
from torchvision import transforms
from utils.data_loading import IDRIDDataset, load_image  # Import from data_loading
from unet.unet_resnet import UNetResNet

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
    """Predict using patches with weighted blending, using the same latent vector."""
    model.eval()
    device = img.device
    
    # Get dimensions
    _, c, h, w = img.shape
    print(f"Input image shape: {img.shape}")
    
    # Initialize the full mask and weight map
    full_mask = torch.zeros((1, 1, h, w), device=device)
    weight_map = torch.zeros((1, 1, h, w), device=device)
    
    # Create a weight matrix for blending at full resolution
    y, x = torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size), indexing='ij')
    weight = (1 - x.abs()) * (1 - y.abs())
    weight = weight.to(device)[None, None, :, :]
    
    # Calculate steps with overlap
    stride = patch_size - overlap
    
    # Calculate number of patches needed
    n_patches_h = max(1, (h - patch_size + stride) // stride)
    n_patches_w = max(1, (w - patch_size + stride) // stride)
    
    print(f"Processing {n_patches_h}x{n_patches_w} patches")
    
    # Process each patch
    with torch.no_grad():
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                top = min(i * stride, h - patch_size)
                left = min(j * stride, w - patch_size)
                
                # Extract patch
                patch = img[:, :, top:top+patch_size, left:left+patch_size]
                print(f"Patch {i},{j} original size: {patch.shape[-2:]}") 
                
                valid_h, valid_w = patch.shape[-2:]
                
                if patch.shape[-2:] != (patch_size, patch_size):
                    # If patch is smaller than patch_size, pad it
                    temp_patch = torch.zeros((1, c, patch_size, patch_size), device=device)
                    temp_patch[:, :, :patch.shape[-2], :patch.shape[-1]] = patch
                    patch = temp_patch
                    print(f"Padded patch to: {patch.shape[-2:]}")
                
                # Get encoder features for this patch
                features = model.encoder(patch)
                x_enc = features[-1]
                
                # Use the same z but interpolate to match encoder output size
                z_patch = F.interpolate(z, size=x_enc.shape[2:], mode='bilinear', align_corners=True)
                
                # Initial projection
                x = model.z_initial(z_patch)
                
                # Decode with z injection at each stage
                for k, decoder_block in enumerate(model.decoder_blocks):
                    skip = features[-(k+2)] if k < len(features)-1 else None
                    x = decoder_block(x, skip, z_patch)
                
                # Final conv and sigmoid
                patch_pred = torch.sigmoid(model.final_conv(x))
                print(f"Patch prediction shape: {patch_pred.shape}")
                
                # Upscale prediction to full patch size using bicubic interpolation
                patch_pred = F.interpolate(patch_pred, size=(patch_size, patch_size), 
                                        mode='bicubic', align_corners=True)
                
                # Apply weight for blending
                patch_pred = patch_pred * weight
                
                # Add to full mask (only the valid part if padded)
                if valid_h != patch_size or valid_w != patch_size:
                    full_mask[:, :, top:top+valid_h, left:left+valid_w] += patch_pred[:, :, :valid_h, :valid_w]
                    weight_map[:, :, top:top+valid_h, left:left+valid_w] += weight[:, :, :valid_h, :valid_w]
                else:
                    full_mask[:, :, top:top+patch_size, left:left+patch_size] += patch_pred
                    weight_map[:, :, top:top+patch_size, left:left+patch_size] += weight
    
    # Average overlapping regions
    eps = 1e-7
    full_mask = full_mask / (weight_map + eps)
    print(f"Final mask shape: {full_mask.shape}")
    
    return full_mask

def get_segmentation_distribution(model, img, num_samples=4, patch_size=512, overlap=100):
    """Generate multiple segmentations using patch-based prediction with consistent latent sampling."""
    model.eval()
    device = img.device
    
    # Ensure input has shape [B, C, H, W]
    if img.dim() == 3:  # [C, H, W]
        img = img.unsqueeze(0)  # [1, C, H, W]
    elif img.dim() == 5:  # [1, 1, C, H, W]
        img = img.squeeze(1)  # [1, C, H, W]
        
    print(f"Input shape after dimension adjustment: {img.shape}")
    B, C, H, W = img.shape
    
    # Get encoder features once for efficiency
    with torch.no_grad():
        features = model.encoder(img)
        x_enc = features[-1]
        mu = model.mu_head(x_enc).squeeze(-1).squeeze(-1)
        logvar = model.logvar_head(x_enc).squeeze(-1).squeeze(-1)
        print(f"Latent distribution shapes - mu: {mu.shape}, logvar: {logvar.shape}")
    
    # Generate multiple samples
    segmentations = []
    for i in range(num_samples):
        # Sample from latent distribution
        z = model.reparameterize(mu, logvar)
        z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions back
        print(f"Sample {i} latent shape: {z.shape}")
        
        # Use the same z for all patches in this sample
        with torch.no_grad():
            seg = predict_with_patches(model, img, z, patch_size, overlap)
            print(f"Sample {i} segmentation shape: {seg.shape}")
            segmentations.append(seg)
    
    # Stack all segmentations: [num_samples, B, C, H, W]
    segmentations = torch.cat(segmentations, dim=0)
    print(f"Final stacked segmentations shape: {segmentations.shape}")
    return segmentations, mu, logvar

def plot_reconstruction(model, img, mask, num_samples=4):
    """Plot the input image, ground truth mask, and multiple sampled reconstructions."""
    # Ensure correct dimensions
    if img.dim() != 4:
        raise ValueError(f"Expected img to have 4 dimensions [B, C, H, W], got shape {img.shape}")
    if mask.dim() != 4:
        raise ValueError(f"Expected mask to have 4 dimensions [B, C, H, W], got shape {mask.shape}")
        
    print(f"Input shapes - img: {img.shape}, mask: {mask.shape}")
    
    # Get multiple segmentations
    segmentations, mu, logvar = get_segmentation_distribution(
        model, img, num_samples=num_samples
    )
    print(f"Segmentations shape after distribution: {segmentations.shape}")
    
    # Calculate mean and std of segmentations
    mean_seg = segmentations.mean(dim=0)  # [1, H, W]
    std_seg = segmentations.std(dim=0)    # [1, H, W]
    print(f"Mean seg shape: {mean_seg.shape}")
    print(f"Std seg shape: {std_seg.shape}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 3)
    
    # Original image (take first batch)
    ax1 = fig.add_subplot(gs[0, 0])
    img_display = img[0].cpu().clone()  # [C, H, W]
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = img_display * std + mean
    
    ax1.imshow(img_display.permute(1, 2, 0).numpy(), interpolation='lanczos')
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Ground truth mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask[0, 0].cpu(), cmap='gray')
    ax2.set_title('Ground Truth')
    ax2.axis('off')
    
    # Mean prediction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mean_seg[0].cpu(), cmap='gray')
    ax3.set_title('Mean Prediction')
    ax3.axis('off')
    
    # Uncertainty (std)
    ax4 = fig.add_subplot(gs[1, 0])
    uncertainty = ax4.imshow(std_seg[0].cpu(), cmap='hot')
    ax4.set_title('Uncertainty (Std)')
    ax4.axis('off')
    plt.colorbar(uncertainty, ax=ax4)
    
    # Sample 1
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(segmentations[0, 0].cpu(), cmap='gray')
    ax5.set_title('Sample 1')
    ax5.axis('off')
    
    # Sample 2
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(segmentations[1, 0].cpu(), cmap='gray')
    ax6.set_title('Sample 2')
    ax6.axis('off')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
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
    mask_dir = Path('./data/masks/test/EX')
    
    # Set scale factor to reduce memory usage
    scale = 0.5  # Reduce image size by half
    
    # Get all test images
    image_files = sorted(test_dir.glob('*.jpg'))[:3]  # Process first 3 images
    
    for i, img_path in enumerate(image_files):
        # Load image and corresponding mask with scaling
        img = load_image_for_prediction(img_path, scale=scale).to(device)
        mask_path = mask_dir / f"{img_path.stem}_EX.tif"
        mask = load_mask_for_prediction(mask_path, scale=scale).to(device)
        
        # Print memory usage before processing
        if torch.cuda.is_available():
            print(f"GPU Memory before processing image {i+1}:")
            print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        # Create visualization with high DPI
        fig = plot_reconstruction(model, img, mask)
        fig.savefig(f'vae_analysis_{i+1}.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        
        # Clear some memory
        torch.cuda.empty_cache()
    
    print("High-resolution visualizations saved as PNG files!")
