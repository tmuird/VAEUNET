import torch
import numpy as np
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Union, Optional

def sample_from_latent(mu: torch.Tensor, logvar: torch.Tensor, 
                      temperature: float = 1.0) -> torch.Tensor:
    """Sample from latent distribution with temperature control."""
    std = torch.exp(0.5 * logvar) * temperature
    eps = torch.randn_like(std)
    return mu + eps * std

def encode_images(model, images):
    """Extract mu and logvar from model in a consistent way."""
    # First try the (seg_output, mu, logvar) tuple pattern
    with torch.no_grad():
        outputs = model(images)
        
        if isinstance(outputs, tuple) and len(outputs) == 3:
            # Model returns (seg_output, mu, logvar)
            _, mu, logvar = outputs
            return mu, logvar
        
        # Try using encoder + mu_head/logvar_head pattern
        try:
            features = model.encoder(images)
            mu = model.mu_head(features)
            logvar = model.logvar_head(features)
            return mu, logvar
        except (AttributeError, TypeError):
            # Last attempt: maybe the model has a dedicated encode method
            try:
                return model.encode(images)
            except AttributeError:
                raise ValueError("Model doesn't provide a standard way to extract mu and logvar.")

# Update this function in utils/vae_utils.py
def generate_predictions(model, images, temperature=1.0, num_samples=1, 
                        return_all=False):
    """Generate multiple predictions using the correct VAE-UNet architecture."""
    device = images.device
    
    # Get latent distribution
    mu, logvar = encode_images(model, images)
    
    # Generate samples
    all_preds = []
    
    for _ in range(num_samples):
        # Sample latent vector
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Add spatial dimensions for proper z injection
        z_spatial = z.unsqueeze(-1).unsqueeze(-1)  # [B, latent_dim, 1, 1]
        
        with torch.no_grad():
            # Get encoder features
            features = model.encoder(images)
            x_enc = features[-1]
            
            # Interpolate z to match encoder output size
            z_full = F.interpolate(z_spatial, size=x_enc.shape[2:], mode='bilinear', align_corners=True)
            
            # Initial projection
            x = model.z_initial(z_full)
            
            # Decode with z injection at each stage
            for k, decoder_block in enumerate(model.decoder_blocks):
                skip = features[-(k+2)] if k < len(features)-1 else None
                x = decoder_block(x, skip, z_full)
            
            # Final conv and sigmoid
            pred = torch.sigmoid(model.final_conv(x))
            
            all_preds.append(pred)
    
    # Stack all predictions
    all_preds = torch.stack(all_preds)
    
    if return_all:
        return {
            'mean': torch.mean(all_preds, dim=0),
            'samples': all_preds,
            'std': torch.std(all_preds, dim=0)
        }
    else:
        return torch.mean(all_preds, dim=0)

def generate_ensemble_prediction(model, image, temps=[0.5, 1.0, 2.0, 3.0], 
                               samples_per_temp=5, weighted=True):
    """Generate ensemble prediction combining multiple temperatures."""
    device = image.device
    all_preds = []
    weights = []
    
    for temp in temps:
        # Generate predictions at this temperature
        result = generate_predictions(
            model, 
            image, 
            temperature=temp, 
            num_samples=samples_per_temp,
            return_all=True
        )
        
        # Add mean prediction to ensemble
        all_preds.append(result['mean'])
        
        # Higher weight for middle temperatures if weighted
        if weighted:
            # Weight based on temperature - favoring middle range (around 1.0)
            weight = 1.0 / (abs(temp - 1.0) + 0.5)
        else:
            weight = 1.0
            
        weights.append(weight)
    
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