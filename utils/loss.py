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
    def __init__(self, bce_weight=0.5, dice_weight=0.5, aux_weight=0.4):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight  # Weight for auxiliary outputs
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        if isinstance(pred, list):
            # Main output loss
            main_loss = self._calculate_loss(pred[0], target)
            
            # Auxiliary outputs loss (if any)
            aux_loss = 0
            if len(pred) > 1:
                for aux_pred in pred[1:]:
                    aux_loss += self._calculate_loss(aux_pred, target)
                aux_loss /= (len(pred) - 1)  # Average auxiliary losses
                
            # Combine main and auxiliary losses
            return main_loss + self.aux_weight * aux_loss
        
        return self._calculate_loss(pred, target)
    
    def _calculate_loss(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice = dice_loss(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice
