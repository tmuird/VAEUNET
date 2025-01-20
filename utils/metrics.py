import torch
from torch import Tensor
from typing import Dict, Tuple
import logging

def dice_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    """Compute Dice score between input and target tensors"""
    # Handle deep supervision outputs
    if isinstance(input, list):
        return dice_score(input[0], target, reduce_batch_first, epsilon)

    # Apply sigmoid if not already applied
    if not isinstance(input, list) and input.requires_grad:
        input = torch.sigmoid(input)

    # Ensure same shape
    if input.shape != target.shape:
        raise ValueError(f'Shape mismatch in dice_score: input {input.shape} vs target {target.shape}')

    # Make input and target contiguous
    input = input.contiguous()
    target = target.contiguous()

    # Handle 4D tensors (batch, channel, height, width)
    if input.dim() == 4:
        intersection = (input * target).sum(dim=(2, 3))
        denominator = input.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()
    else:
        # For flattened or other dimensional tensors
        intersection = (input * target).sum()
        denominator = input.sum() + target.sum()
        dice = (2.0 * intersection + epsilon) / (denominator + epsilon)

    return dice

def multiclass_dice_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    """Average of Dice score for all classes"""
    return dice_score(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    """Dice loss (objective to minimize) between 0 and 1"""
    fn = multiclass_dice_score if multiclass else dice_score
    return 1 - fn(input, target, reduce_batch_first=True)

def get_confusion_matrix(pred: Tensor, target: Tensor, threshold: float = 0.5) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute confusion matrix elements (TP, FP, TN, FN)"""
    # For evaluation metrics, we threshold
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Sum over spatial dimensions (H,W) if they exist
    if pred.dim() > 2:
        tp = (pred_binary * target_binary).sum(dim=(-2,-1))
        fp = pred_binary.sum(dim=(-2,-1)) - tp
        fn = target_binary.sum(dim=(-2,-1)) - tp
        tn = pred_binary[...,0,0].numel() - tp - fp - fn  # Count batch size correctly
    else:
        tp = (pred_binary * target_binary).sum()
        fp = pred_binary.sum() - tp
        fn = target_binary.sum() - tp
        tn = pred_binary.numel() - tp - fp - fn
    
    return tp, fp, tn, fn

def get_all_metrics(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Dict[str, float]:
    """Compute all metrics at once efficiently"""
    # Handle deep supervision outputs
    if isinstance(pred, list):
        pred = pred[0]
    
    # Apply sigmoid for probability scores
    pred = torch.sigmoid(pred)
    
    # Calculate Dice using the same method as loss function
    dice = dice_score(pred, target, epsilon=epsilon)
    
    # Get confusion matrix for binary metrics
    tp, fp, tn, fn = get_confusion_matrix(pred, target)
    
    # Calculate metrics using confusion matrix elements and take mean over batch
    precision = ((tp + epsilon) / (tp + fp + epsilon)).mean()
    recall = ((tp + epsilon) / (tp + fn + epsilon)).mean()
    specificity = ((tn + epsilon) / (tn + fp + epsilon)).mean()
    f1 = (2 * (precision * recall) / (precision + recall + epsilon)).mean()
    accuracy = ((tp + tn) / (tp + tn + fp + fn)).mean()
    
    # Calculate IoU using binary predictions
    intersection_binary = tp
    union_binary = tp + fp + fn
    iou = ((intersection_binary + epsilon) / (union_binary + epsilon)).mean()
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item()
    }

def iou_score(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Compute Intersection over Union score"""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    return (intersection + epsilon) / (union + epsilon)

def precision_recall(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Compute Precision and Recall scores"""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    tp = (pred_binary * target_binary).sum()
    fp = pred_binary.sum() - tp
    fn = target_binary.sum() - tp
    
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    
    return precision, recall

def specificity(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Compute Specificity (True Negative Rate)"""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    tn = ((1 - pred_binary) * (1 - target_binary)).sum()
    fp = pred_binary.sum() - (pred_binary * target_binary).sum()
    
    return (tn + epsilon) / (tn + fp + epsilon)

def f1_score(precision: Tensor, recall: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Compute F1 Score from precision and recall"""
    return 2 * (precision * recall) / (precision + recall + epsilon)

def accuracy(pred: Tensor, target: Tensor) -> Tensor:
    """Compute Accuracy score"""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    correct = (pred_binary == target_binary).float().sum()
    total = torch.numel(pred)
    
    return correct / total

class MetricTracker:
    """Class to track metrics during training"""
    def __init__(self):
        self.metrics = {
            'train': {metric: [] for metric in ['loss', 'dice', 'iou', 'precision', 'recall', 'specificity', 'f1', 'accuracy']},
            'val': {metric: [] for metric in ['loss', 'dice', 'iou', 'precision', 'recall', 'specificity', 'f1', 'accuracy']}
        }
        self.best_dice = 0.0
    
    def update(self, phase: str, metrics: Dict[str, float]):
        """Update metrics for given phase (train/val)"""
        for metric, value in metrics.items():
            self.metrics[phase][metric].append(value)
    
    def get_current(self, phase: str) -> Dict[str, float]:
        """Get most recent metrics for given phase"""
        return {metric: values[-1] if values else 0.0 
                for metric, values in self.metrics[phase].items()}
    
    def is_best_dice(self, current_dice: float) -> bool:
        """Check if current dice score is best"""
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