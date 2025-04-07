import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(inputs, targets, smooth=1.0):
    """
    Compute the Dice loss between inputs and targets.
    
    Args:
        inputs: Model predictions, raw logits before sigmoid (B, C, H, W)
        targets: Ground truth masks (B, C, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss scalar
    """
    # Apply sigmoid to raw logits
    inputs = torch.sigmoid(inputs)
    
    # Flatten inputs and targets
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection and union
    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return 1.0 - dice

class DiceLoss(nn.Module):
    """
    Dice Loss class wrapper around the dice_loss function
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        return dice_loss(inputs, targets, self.smooth)

class CombinedLoss(nn.Module):
    """
    Combines BCE and Dice loss for segmentation tasks with customizable weights.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        # Binary Cross Entropy loss
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        
        # Dice Loss
        dice = dice_loss(inputs, targets)
        
        # Combined loss
        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss

class MAFocalLoss(nn.Module):
    """
    Focal Loss specifically tuned for microaneurysms, which addresses the extreme class imbalance.
    """
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive class (higher for rare MA)
        self.gamma = gamma  # Focus parameter (higher to focus more on hard examples)
        self.eps = 1e-6
        
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Focal loss formula
        p_t = targets * inputs_sigmoid + (1 - targets) * (1 - inputs_sigmoid)
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # Add alpha weighting - higher weight for positive pixels (MA)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Binary cross-entropy with added weights
        bce = -targets * torch.log(inputs_sigmoid + self.eps) - (1 - targets) * torch.log(1 - inputs_sigmoid + self.eps)
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()

class MASegmentationLoss(nn.Module):
    """
    Combined loss function specifically for microaneurysms segmentation.
    Combines Focal Loss with a DiceLoss, both weighted to handle small features.
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, focal_gamma=2.0, class_weight=0.9):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_loss = MAFocalLoss(alpha=class_weight, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        combined_loss = self.dice_weight * dice + self.focal_weight * focal
        return combined_loss

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
