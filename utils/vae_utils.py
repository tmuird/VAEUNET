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

def generate_predictions(model, images, temperature=1.0, num_samples=10):
    """
    Generate multiple predictions from the VAE model and average them.
    
    Args:
        model: The VAE-UNet model
        images: Input images tensor [B, C, H, W]
        temperature: Temperature for sampling (higher = more diversity)
        num_samples: Number of predictions to generate and average
        
    Returns:
        Average prediction after sigmoid [B, 1, H, W]
    """
    model.eval()
    device = images.device
    B, C, H, W = images.shape
    
    # Initialize tensor to accumulate predictions
    accumulated_preds = torch.zeros((B, 1, H, W), device=device)
    
    # Get latent distribution parameters
    with torch.no_grad():
        mu, logvar = encode_images(model, images)
        
        # Generate multiple samples
        for _ in range(num_samples):
            # Sample from latent space with temperature
            std = torch.exp(0.5 * logvar) * temperature
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # Add spatial dimensions for the decoder
            z = z.unsqueeze(-1).unsqueeze(-1)  # [B, latent_dim, 1, 1]
            
            # Get encoder features again
            features = model.encoder(images)
            
            # Initial size matches the smallest encoder feature
            initial_size = features[-1].shape[2:]
            z_full = F.interpolate(z, size=initial_size, mode='bilinear', align_corners=True)
            
            # Initial projection
            x = model.z_initial(z_full)
            
            # Decode with z injection at each stage
            for k, decoder_block in enumerate(model.decoder_blocks):
                skip = features[-(k+2)] if k < len(features)-1 else None
                x = decoder_block(x, skip, z_full)
            
            # Final conv and resize
            pred = model.final_conv(x)
            
            # Ensure output size matches input size
            if pred.shape[2:] != (H, W):
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
            
            # Apply sigmoid for probability
            pred = torch.sigmoid(pred)
            
            # Accumulate prediction
            accumulated_preds += pred
    
    # Average the predictions
    return accumulated_preds / num_samples

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

def kl_with_free_bits(mu, logvar, free_bits=1e-4):
    """
    Compute KL divergence with free bits to prevent posterior collapse.
    
    Args:
        mu: Mean vector
        logvar: Log variance vector
        free_bits: Minimum value for KL per dimension
    """
    # Standard KL calculation
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
    
    # Apply free bits: max(free_bits, kl_i) for each dimension
    if free_bits > 0:
        kl_per_dim = torch.max(kl_per_dim, torch.tensor(free_bits, device=kl_per_dim.device))
    
    # Sum across latent dimensions
    kl_loss = kl_per_dim.sum(dim=1).mean()
    return kl_loss

class KLAnnealer:
    """
    Anneals the KL weight over time to prevent posterior collapse.
    """
    def __init__(self, kl_start=0.0, kl_end=1.0, warmup_epochs=10, strategy='linear'):
        self.kl_start = kl_start
        self.kl_end = kl_end
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy
        
    def get_weight(self, epoch, batch=None, num_batches=None):
        if self.strategy == 'constant':
            return self.kl_end
            
        if batch is not None and num_batches is not None:
            # Batch-level annealing within each epoch
            progress = (epoch + batch / num_batches) / self.warmup_epochs
        else:
            # Epoch-level annealing
            progress = epoch / self.warmup_epochs
            
        progress = min(progress, 1.0)
        
        if self.strategy == 'linear':
            return self.kl_start + progress * (self.kl_end - self.kl_start)
        elif self.strategy == 'cyclical':
            # Cycle between min and max values
            cycle_progress = progress % 1.0
            return self.kl_start + cycle_progress * (self.kl_end - self.kl_start)
        else:
            return self.kl_end