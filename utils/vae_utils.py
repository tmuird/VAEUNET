import torch
import numpy as np
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm

def sample_from_latent(mu: torch.Tensor, logvar: torch.Tensor, 
                      temperature: float = 1.0) -> torch.Tensor:
    """Sample from latent distribution with temperature control."""
    std = torch.exp(0.5 * logvar) * temperature
    eps = torch.randn_like(std)
    return mu + eps * std

def encode_images(model, images):
    """Extract mean and log variance from input images."""
    model.eval()
    with torch.no_grad():
        # Get encoder features
        features = model.encoder(images)
        x_enc = features[-1]
        
        # Extract latent distribution parameters
        mu = model.mu_head(x_enc).squeeze(-1).squeeze(-1)  # Shape: [B, latent_dim]
        logvar = model.logvar_head(x_enc).squeeze(-1).squeeze(-1)  # Shape: [B, latent_dim]
        
    return mu, logvar

def generate_predictions(model, images, temperature=1.0, num_samples=3):
    """
    Generate predictions using the VAE model with a specific temperature.
    
    Args:
        model: The VAE-UNet model
        images: Input images
        temperature: Temperature for sampling from the latent space
                    Higher values = more diversity
        num_samples: Number of samples to generate and average
    
    Returns:
        Averaged prediction from multiple samples
    """
    model.eval()
    device = images.device

    with torch.no_grad():
        # Get encoder features and distribution parameters
        features = model.encoder(images)
        x_enc = features[-1]
        mu = model.mu_head(x_enc).squeeze(-1).squeeze(-1)
        logvar = model.logvar_head(x_enc).squeeze(-1).squeeze(-1)
        
        # Prepare to accumulate predictions
        predictions = []
        
        for _ in range(num_samples):
            # Sample from latent space with temperature
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * temperature
            z = mu + eps * std
            
            # Forward pass through decoder using sampled z
            z = z.unsqueeze(-1).unsqueeze(-1)
            
            # Spatial size for initial z projection
            initial_size = features[-1].shape[2:]
            z_spatial = F.interpolate(z, size=initial_size, mode='bilinear', align_corners=True)
            
            # Apply bottleneck processing based on injection strategy
            if hasattr(model, 'use_bottleneck') and model.use_bottleneck:
                x = model.z_initial(z_spatial)
            else:
                # For 'none' mode, use encoder features
                x = features[-1]
            
            # Decode with conditional z injection
            for i, decoder_block in enumerate(model.decoder_blocks):
                skip = features[-(i+2)] if i < len(features)-1 and model.use_skip else None
                x = decoder_block(x, skip, z_spatial)
            
            # Final conv
            pred = model.final_conv(x)
            predictions.append(pred)
            
        # Average the predictions
        if len(predictions) > 1:
            pred = torch.stack(predictions).mean(0)
        else:
            pred = predictions[0]
        
    return pred

def calculate_latent_stats(mu, logvar):
    """
    Calculate statistics about the latent space to monitor for posterior collapse.
    
    Args:
        mu: Mean vectors from the encoder [B, latent_dim]
        logvar: Log variance vectors from the encoder [B, latent_dim]
        
    Returns:
        Dictionary of metrics about the latent space
    """
    # Calculate metrics
    mean_mu = mu.mean(dim=0)
    var = torch.exp(logvar)
    mean_var = var.mean(dim=0)
    
    # Check for posterior collapse: 
    # If KL is near zero, the posterior is too close to the prior
    active_dims = ((mean_mu.abs() > 0.1) | (mean_var < 0.9) | (mean_var > 1.1)).sum().item()
    total_dims = mu.shape[1]
    activity_ratio = active_dims / total_dims
    
    # Calculate KL divergence per dimension
    kl_per_dim = 0.5 * (mean_mu.pow(2) + mean_var - logvar.mean(dim=0) - 1)
    total_kl = kl_per_dim.sum().item()
    
    return {
        'active_dims': active_dims,
        'total_dims': total_dims,
        'activity_ratio': activity_ratio,
        'total_kl': total_kl,
        'mean_mu_abs': mean_mu.abs().mean().item(),
        'mean_var': mean_var.mean().item(),
    }