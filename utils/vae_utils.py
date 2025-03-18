import torch
import numpy as np
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm
import gc

def sample_from_latent(mu: torch.Tensor, logvar: torch.Tensor, 
                      temperature: float = 1.0) -> torch.Tensor:
    """Sample from latent distribution with temperature control."""
    std = torch.exp(0.5 * logvar) * temperature
    eps = torch.randn_like(std)
    return mu + eps * std

def encode_images(model, img):
    """Encode images using the model's encoder to get latent distribution parameters."""
    # This wrapper ensures consistent API whether model has encode method or not
    with torch.no_grad():
        if hasattr(model, 'encode'):
            return model.encode(img)
        else:
            # Manual encoding if encode method doesn't exist
            features = model.encoder(img)
            x_enc = features[-1]
            # Apply global average pooling
            x_pool = torch.mean(x_enc, dim=[2, 3])
            # Use latent projections to get mu and logvar
            mu = model.mu_projection(x_pool)
            logvar = model.logvar_projection(x_pool)
            return mu, logvar

def generate_predictions(model, images, temperature=1.0, num_samples=5):
    """Generate predictions by sampling from the VAE latent space.
    Args:
        model: UNet-VAE model
        images: Input images tensor [B, C, H, W]
        temperature: Sampling temperature (higher = more diverse)
        num_samples: Number of samples to generate
    Returns:
        Average prediction tensor [B, 1, H, W]
    """
    # Check if we have a list of images with different sizes (batch collation)
    device = next(model.parameters()).device
    
    # Handle case when images are passed as a list (mixed sizes)
    if isinstance(images, list):
        # Process each image separately and return a list of predictions
        results = []
        for img in images:
            # Ensure single image has batch dimension
            img_batch = img.unsqueeze(0) if img.dim() == 3 else img
            results.append(generate_predictions(model, img_batch, temperature, num_samples))
        return results
    
    # Continue with tensor batch processing
    B, C, H, W = images.shape
    
    # Enter evaluation mode but with dropout potentially on
    was_training = model.training
    model.eval()  # eval mode still needed for batch norm layers
    
    # Record original input size for consistent outputs
    original_size = (H, W)
    
    with torch.no_grad():
        # Get latent distribution parameters
        mu, logvar = encode_images(model, images)
        
        # Initialize prediction accumulator
        pred_sum = torch.zeros((B, 1, H, W), device=device)
        
        # Generate multiple predictions
        for i in range(num_samples):
            # Sample from latent space with temperature
            std = torch.exp(0.5 * logvar) * temperature
            eps = torch.randn_like(std)
            z = mu + eps * std
            z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
            
            # Pass through decoder
            features = model.encoder(images)
            x_enc = features[-1]
            
            # Interpolate z to match encoder output size
            z_resized = F.interpolate(z, size=x_enc.shape[2:], mode='bilinear', align_corners=True)
            
            # Initial projection
            x = model.z_initial(z_resized)
            
            # Decode with z injection at each stage
            for k, decoder_block in enumerate(model.decoder_blocks):
                skip = features[-(k+2)] if k < len(features)-1 else None
                x = decoder_block(x, skip, z_resized)
            
            # Final conv and sigmoid
            pred = torch.sigmoid(model.final_conv(x))
            
            # Ensure prediction has consistent size with input images
            # This is the critical fix to ensure all predictions have the same size
            if pred.shape[2:] != original_size:
                pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=True)
                
            # Add to running sum
            pred_sum += pred
            
            # Clean up
            del pred, features, x_enc, z_resized, x, eps
        
        # Restore model state
        if was_training:
            model.train()
        
        # Return average prediction
        return pred_sum / num_samples

def generate_ensemble_prediction(model, image, temps=[0.5, 1.0, 2.0, 3.0], 
                               samples_per_temp=5, weighted=True,
                               patch_size=512, overlap=100):
    """Memory-efficient ensemble prediction combining multiple temperatures."""
    device = image.device
    all_preds = []
    weights = []
    
    for temp in temps:
        logging.info(f"Generating ensemble component for temperature {temp}")
        # Generate predictions at this temperature with patching
        temp_pred = generate_predictions(
            model, 
            image, 
            temperature=temp, 
            num_samples=samples_per_temp,
            patch_size=patch_size,
            overlap=overlap
        )
        
        # Add prediction to ensemble
        all_preds.append(temp_pred)
        
        # Compute weight
        if weighted:
            weight = 1.0 / (abs(temp - 1.0) + 0.5)
        else:
            weight = 1.0
            
        weights.append(weight)
        
        # Free memory after each temperature
        torch.cuda.empty_cache()
    
    # Normalize weights
    weights = torch.tensor(weights, device=device)
    weights = weights / weights.sum()
    
    # Compute weighted ensemble
    ensemble = torch.zeros_like(all_preds[0])
    for pred, w in zip(all_preds, weights):
        ensemble += pred * w
        
    return ensemble

def resize_or_patch_image(model, image):
    """Resize image to the size the model expects while preserving aspect ratio."""
    # Get model's expected input size (default to 256 if not specified)
    target_size = 256
    if hasattr(model, 'patch_size'):
        target_size = model.patch_size
    
    orig_h, orig_w = image.shape[2], image.shape[3]
    logging.info(f"Original image dimensions: {orig_h}x{orig_w}, target: {target_size}")
    
    # Check if resize is needed
    if orig_h > target_size or orig_w > target_size:
        # Resize preserving aspect ratio
        scale = min(target_size / orig_h, target_size / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        resized = F.interpolate(
            image, 
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
        logging.info(f"Resized image to: {new_h}x{new_w} (preserved aspect ratio)")
        return resized, (orig_h, orig_w)
    
    return image, (orig_h, orig_w)