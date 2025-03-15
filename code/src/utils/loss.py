import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.0):
    """Calculate Dice Loss"""
    if isinstance(pred, list):
        return dice_loss(pred[0], target, smooth)  # Only use main output for dice loss
        
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, aux_weight=0.4, kld_weight=0.1):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.kld_weight = kld_weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        if isinstance(pred, list):
            # Split predictions into segmentation outputs and VAE parameters
            *seg_outputs, mu, logvar = pred
            
            # Main segmentation loss
            main_loss = self._calculate_loss(seg_outputs[0], target)
            
            # Auxiliary segmentation losses (if any)
            aux_loss = 0
            if len(seg_outputs) > 1:
                for aux_pred in seg_outputs[1:-2]:  # Exclude mu and logvar
                    aux_loss += self._calculate_loss(aux_pred, target)
                aux_loss /= (len(seg_outputs) - 1)
            
            # KL divergence loss for VAE
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld_loss = kld_loss / mu.size(0)  # Normalize by batch size
            
            # Combine all losses
            total_loss = main_loss + self.aux_weight * aux_loss + self.kld_weight * kld_loss
            return total_loss
        
        return self._calculate_loss(pred, target)
    
    def _calculate_loss(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice = dice_loss(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice
