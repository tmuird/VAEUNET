import torch
from torch import Tensor
from typing import Dict, Tuple
import logging
from torchmetrics.functional import dice

def dice_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    """Compute Dice score between input and target tensors"""
    # Add shape logging
    logging.debug(f'Dice calculation - Input shape: {input.shape}, Target shape: {target.shape}')

    # Ensure same shape
    if input.shape != target.shape:
        raise ValueError(f'Shape mismatch in dice_score: input {input.shape} vs target {target.shape}')

    # Apply thresholding to make predictions binary
    input = (input > 0.5).float()
    target = (target > 0.5).float()

    if input.dim() > 2 or target.dim() > 2:
        if reduce_batch_first:
            input = input.contiguous().view(input.shape[0], -1)
            target = target.contiguous().view(target.shape[0], -1)
        else:
            input = input.contiguous().view(-1)
            target = target.contiguous().view(-1)

    intersection = (input * target).sum()
    denominator = input.sum() + target.sum()

    if denominator.item() == 0:
        return torch.tensor(1.0).to(input.device)

    return (2.0 * intersection + epsilon) / (denominator + epsilon)

def multiclass_dice_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    """Average of Dice score for all classes"""
    return dice_score(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    """Dice loss (objective to minimize) between 0 and 1"""
    fn = multiclass_dice_score if multiclass else dice_score
    return 1 - fn(input, target, reduce_batch_first=True)

def iou_score(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Compute Intersection over Union score"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + epsilon) / (union + epsilon)

def precision_recall(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Compute Precision and Recall scores"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fp = pred.sum() - tp
    fn = target.sum() - tp
    
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    
    return precision, recall

def specificity(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Compute Specificity (True Negative Rate)"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = pred.sum() - (pred * target).sum()
    
    return (tn + epsilon) / (tn + fp + epsilon)

def accuracy(pred: Tensor, target: Tensor) -> Tensor:
    """Compute Accuracy score"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    
    return correct / total

def get_all_metrics(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Dict[str, float]:
    """Compute metrics following standard medical image segmentation practices"""
    # Log input statistics
    # logging.info(f'\nMetrics Input Statistics:')
    # logging.info(f'Predictions:\n    Shape: {pred.shape}\n    Min: {pred.min():.4f}, Max: {pred.max():.4f}\n    Mean: {pred.mean():.4f}')
    # logging.info(f'True Masks:\n    Shape: {target.shape}\n    Min: {target.min():.4f}, Max: {target.max():.4f}\n    Mean: {target.mean():.4f}')
    
    # Calculate standard metrics
    dice_val = dice_score(pred, target)
    iou_val = iou_score(pred, target)
    prec, recall = precision_recall(pred, target)
    spec = specificity(pred, target)
    acc = accuracy(pred, target)
    
    metrics = {
        'dice': dice_val.item(),
        'iou': iou_val.item(),
        'precision': prec.item(),
        'recall': recall.item(),
        'specificity': spec.item(),
        'accuracy': acc.item()
    }
    
    # Log computed metrics
    # logging.info(f'\nComputed Metrics:')
    # logging.info(f'    Dice: {metrics["dice"]:.4f}')
    # logging.info(f'    IoU: {metrics["iou"]:.4f}')
    
    return metrics

class MetricTracker:
    """Class to track metrics during training"""
    def __init__(self):
        # Standard metrics
        standard_metrics = ['loss', 'dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy']
        
        self.metrics = {
            'train': {metric: [] for metric in standard_metrics},
            'val': {metric: [] for metric in standard_metrics}
        }
        self.best_dice = 0.0

    def update(self, phase: str, metrics: Dict[str, float]):
        """Update metrics for given phase (train/val)"""
        for k, v in metrics.items():
            self.metrics[phase][k].append(v)

    def get_current(self, phase: str) -> Dict[str, float]:
        """Get most recent metrics for given phase"""
        return {k: v[-1] if v else 0.0 for k, v in self.metrics[phase].items()}

    def is_best_dice(self, current_dice: float) -> bool:
        """Check if current dice score is best (using standard Dice calculated on positive patches only)"""
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            return True
        return False

def focal_loss(input: Tensor, target: Tensor, alpha: float = 0.8, gamma: float = 2.0, epsilon: float = 1e-6):
    """
    Compute Focal Loss for binary segmentation
    
    Args:
        input: Model predictions after sigmoid [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]
        alpha: Weighting factor for rare class (exudates)
        gamma: Focusing parameter (reduces loss for well-classified examples)
        epsilon: Small constant to prevent log(0)
    """
    # Ensure input has been passed through sigmoid
    input = torch.sigmoid(input)
    
    # Flatten the tensors
    input = input.view(-1)
    target = target.view(-1)
    
    # Calculate BCE
    bce = -target * torch.log(input + epsilon) - (1 - target) * torch.log(1 - input + epsilon)
    
    # Apply focal term
    pt = torch.where(target == 1, input, 1 - input)
    focal_term = (1 - pt) ** gamma
    
    # Apply alpha weighting
    alpha_weight = torch.where(target == 1, alpha, 1 - alpha)
    
    # Combine all terms
    focal = alpha_weight * focal_term * bce
    
    return focal.mean()