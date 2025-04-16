import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

def calculate_uncertainty_metrics(segmentations):
    """Calculate various uncertainty metrics from multiple segmentation samples.
    
    Args:
        segmentations: Tensor of shape [N, B, C, H, W] where N is number of samples
    
    Returns:
        Dictionary of tensors with keys:
         - 'mean': mean prediction
         - 'std': pixelwise standard deviation
         - 'entropy': pixelwise entropy of the mean
         - 'mutual_info': pixelwise mutual information
         - 'coeff_var': pixelwise coefficient of variation
    """
    # N = number of samples
    mean_pred = segmentations.mean(dim=0)  # [B, C, H, W]
    std_dev = segmentations.std(dim=0)     # [B, C, H, W]

    epsilon = 1e-7
    entropy = -(mean_pred * torch.log(mean_pred + epsilon) + 
               (1 - mean_pred) * torch.log(1 - mean_pred + epsilon))

    sample_entropies = -(segmentations * torch.log(segmentations + epsilon) + 
                        (1 - segmentations)*torch.log(1 - segmentations + epsilon))
    mean_entropy = sample_entropies.mean(dim=0)  # [B, C, H, W]
    mutual_info = entropy - mean_entropy
    
    coeff_var = std_dev / (mean_pred + epsilon)
    
    return {
        'mean': mean_pred.squeeze(1),
        'std': std_dev.squeeze(1),
        'entropy': entropy.squeeze(1),
        'mutual_info': mutual_info.squeeze(1),
        'coeff_var': coeff_var.squeeze(1)
    }

def calculate_expected_calibration_error(pred_probs, ground_truth, num_bins=10):
    device = pred_probs.device
    pred_flat = pred_probs.flatten()
    gt_flat = ground_truth.flatten()

    bin_boundaries = torch.linspace(0, 1, num_bins+1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accs = torch.zeros(num_bins, device=device)
    bin_confs = torch.zeros(num_bins, device=device)
    bin_counts = torch.zeros(num_bins, device=device)

    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (pred_flat >= bin_lower) & (pred_flat < bin_upper)
        bin_counts[bin_idx] = in_bin.sum()
        if bin_counts[bin_idx] > 0:
            bin_accs[bin_idx] = gt_flat[in_bin].float().mean()
            bin_confs[bin_idx] = pred_flat[in_bin].mean()
    
    ece = (bin_counts * (bin_accs - bin_confs).abs()).sum() / bin_counts.sum()

    ece_value = ece.item()
    bin_accs_np = bin_accs.cpu().numpy()
    bin_confs_np = bin_confs.cpu().numpy()
    bin_counts_np = bin_counts.cpu().numpy()

    return ece_value, bin_accs_np, bin_confs_np, bin_counts_np

def brier_score(pred_probs, ground_truth):
    brier = F.mse_loss(pred_probs, ground_truth.float())
    return brier.item()

def plot_reliability_diagram(bin_accs, bin_confs, bin_counts, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    
    normalized_counts = bin_counts / bin_counts.max() if bin_counts.max()>0 else bin_counts
    bin_centers = np.linspace(0.05, 0.95, len(bin_accs))
    bar_width = 0.35

    bars1 = ax.bar(bin_centers - bar_width/2, bin_accs, bar_width, alpha=0.7, color='blue', label='Accuracy')
    bars2 = ax.bar(bin_centers + bar_width/2, bin_confs, bar_width, alpha=0.7, color='green', label='Confidence')
    
    gap = np.abs(bin_accs - bin_confs)
    for i, (center, acc, conf) in enumerate(zip(bin_centers, bin_accs, bin_confs)):
        if gap[i] > 0.05:
            y_min, y_max = min(acc, conf), max(acc, conf)
            ax.plot([center, center],[y_min, y_max], color='red', linestyle='-', lw=2, alpha=0.7)
    
    ax2 = ax.twinx()
    ax2.bar(bin_centers, normalized_counts, width=bar_width*1.8, alpha=0.15, color='gray')
    ax2.set_ylabel('Relative Frequency', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0,1.1)
    
    ax.plot([0,1],[0,1],'k--', label='Perfect Calibration')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid(True, alpha=0.3)
    
    return ax

def calculate_sparsification_metrics(pred_probs, uncertainties, ground_truth, num_points=20):
    device = pred_probs.device
    batch_size = pred_probs.shape[0]

    epsilon = 1e-7
    pixel_errors = -(ground_truth*torch.log(pred_probs+epsilon) + 
                     (1 - ground_truth)*torch.log(1-pred_probs+epsilon))
    
    pixel_errors_flat = pixel_errors.reshape(batch_size,-1).detach().cpu().numpy()
    uncertainties_flat = uncertainties.reshape(batch_size,-1).detach().cpu().numpy()

    fraction_removed = np.linspace(0, 0.99, num_points)
    errors_random = np.zeros(num_points)
    errors_uncertainty = np.zeros(num_points)

    for b in range(batch_size):
        batch_errors = pixel_errors_flat[b]
        batch_uncerts = uncertainties_flat[b]
        if np.all(batch_errors==batch_errors[0]) or np.isnan(batch_errors).any():
            continue
        initial_error = batch_errors.mean()
        if initial_error<=0 or np.isnan(initial_error):
            continue
        
        num_pixels = batch_errors.shape[0]
        all_indices = np.arange(num_pixels)
        uncertainty_order = np.argsort(batch_uncerts)[::-1]

        for i, frac in enumerate(fraction_removed):
            if frac>=1.0:
                errors_random[i]+=1.0
                errors_uncertainty[i]+=1.0
                continue
            pixels_to_remove = int(num_pixels*frac)
            pixels_to_keep = num_pixels-pixels_to_remove
            if pixels_to_keep<=0:
                errors_random[i]+=1.0
                errors_uncertainty[i]+=1.0
                continue
            
            np.random.seed(i+b*1000)
            random_indices = np.random.choice(all_indices, pixels_to_keep, replace=False)
            random_error = batch_errors[random_indices].mean()
            errors_random[i]+=random_error/initial_error

            uncertainty_indices = uncertainty_order[pixels_to_remove:]
            uncertainty_error = batch_errors[uncertainty_indices].mean()
            errors_uncertainty[i]+=uncertainty_error/initial_error

    valid_batches = batch_size
    errors_random/=valid_batches
    errors_uncertainty/=valid_batches

    if errors_random[0]>0:
        errors_random=errors_random/errors_random[0]
    if errors_uncertainty[0]>0:
        errors_uncertainty=errors_uncertainty/errors_uncertainty[0]
    
    for i in range(1,num_points):
        if errors_uncertainty[i]>errors_uncertainty[i-1]:
            errors_uncertainty[i]=errors_uncertainty[i-1]
    
    errors_random=np.nan_to_num(errors_random,nan=1.0,posinf=1.0,neginf=0.0)
    errors_uncertainty=np.nan_to_num(errors_uncertainty,nan=1.0,posinf=1.0,neginf=0.0)

    return fraction_removed, errors_random, errors_uncertainty

def plot_sparsification_curve(fraction_removed, errors_random, errors_uncertainty, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    
    se = np.trapz(errors_random - errors_uncertainty, fraction_removed)
    color='green' if se>0 else 'red'
    ax.fill_between(fraction_removed, errors_random, errors_uncertainty,alpha=0.2, color=color)
    
    ax.plot(fraction_removed, errors_random, 'b--', label='Random')
    ax.plot(fraction_removed, errors_uncertainty, 'r-', label='By Uncertainty')
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)
    
    half_idx = np.argmin(np.abs(errors_uncertainty-0.5))
    if half_idx< len(fraction_removed):
        half_frac = fraction_removed[half_idx]
        ax.plot([half_frac],[0.5],'ro',markersize=8,alpha=0.7)
        ax.annotate(f'{half_frac:.2f}',xy=(half_frac,0.5),xytext=(half_frac+0.05,0.55),
                    arrowprops=dict(arrowstyle="->",color='black',alpha=0.6))
    
    ax.set_xlabel('Fraction of Pixels Removed',fontsize=12)
    ax.set_ylabel('Normalized Error',fontsize=12)
    ax.set_title(f'Sparsification Curve (SE={se:.4f})',fontsize=14)
    ax.legend(loc='best')
    ax.grid(True,alpha=0.3)
    ax.set_xlim(0,max(fraction_removed))
    ax.set_ylim(0,1.1)
    return ax,se
def calculate_uncertainty_error_auc(mean_pred, gt_mask, uncertainty_map):
    pred_binary = (mean_pred>0.5).float().view(-1).cpu().numpy()
    gt_flat = gt_mask.view(-1).cpu().numpy()
    uncertainty = uncertainty_map.view(-1).cpu().numpy()

    errors = (pred_binary!=gt_flat).astype(np.int32)
    try:
        auroc = roc_auc_score(errors, uncertainty)
        auprc = average_precision_score(errors, uncertainty)
    except ValueError:
        auroc=float('nan')
        auprc=float('nan')
    
    return auroc,auprc

# def calculate_uncertainty_error_auc(mean_pred, gt_mask, uncertainty_map, threshold=0.5, handle_imbalance=True):
#     """Calculate how well uncertainty predicts errors in segmentation.
    
#     Args:
#         mean_pred: Mean prediction probabilities [B, H, W]
#         gt_mask: Ground truth binary mask [B, H, W]
#         uncertainty_map: Uncertainty measure [B, H, W]
#         threshold: Threshold for binarizing predictions (default: 0.5)
#         handle_imbalance: Whether to use class-balancing techniques for severe imbalance
        
#     Returns:
#         Tuple of (AUROC, AUPRC) between uncertainty and prediction errors
#     """
#     pred_binary = (mean_pred > threshold).float().view(-1).cpu().numpy()
#     gt_flat = gt_mask.view(-1).cpu().numpy()
#     uncertainty = uncertainty_map.view(-1).cpu().numpy()

#     # Get binary error mask (1 where prediction != ground truth)
#     errors = (pred_binary != gt_flat).astype(np.int32)
    
#     # Get proportion of errors to understand class imbalance
#     error_rate = np.mean(errors)
#     logging.info(f"Error rate: {error_rate:.4f} (fraction of pixels where prediction != ground truth)")
    
#     # Check if we have both error and non-error cases (need both for ROC/PR curve)
#     if np.sum(errors) == 0 or np.sum(errors) == len(errors):
#         # All predictions correct or all predictions wrong - ROC/PR not defined
#         return float('nan'), float('nan')
    
#     # Check if uncertainty has variation - needed for meaningful scoring
#     if np.std(uncertainty) < 1e-10:
#         # No variation in uncertainty scores
#         return float('nan'), float('nan')
    
#     try:
#         # Check for NaN or infinity values that could cause problems
#         if np.isnan(uncertainty).any() or np.isinf(uncertainty).any():
#             return float('nan'), float('nan')
        
#         # For severe class imbalance (typical in medical imaging segmentation):
#         if handle_imbalance and (error_rate < 0.05 or error_rate > 0.95):
#             # Option 1: Use stratified sampling to balance the classes
#             pos_indices = np.where(errors == 1)[0]
#             neg_indices = np.where(errors == 0)[0]
            
#             # Limit to the smaller of the two class sizes
#             min_samples = min(len(pos_indices), len(neg_indices))
#             max_samples = 100000  # Cap at 100k samples to prevent memory issues
#             min_samples = min(min_samples, max_samples)
            
#             if min_samples > 100:  # Ensure we have enough samples for meaningful metrics
#                 # If we have at least some of both classes
#                 np.random.seed(42)  # For reproducibility
#                 sampled_pos = np.random.choice(pos_indices, min_samples, replace=False)
#                 sampled_neg = np.random.choice(neg_indices, min_samples, replace=False)
                
#                 # Combine the balanced samples
#                 balanced_indices = np.concatenate([sampled_pos, sampled_neg])
#                 balanced_errors = errors[balanced_indices]
#                 balanced_uncertainty = uncertainty[balanced_indices]
                
#                 # Calculate metrics on the balanced sample
#                 auroc_balanced = roc_auc_score(balanced_errors, balanced_uncertainty)
#                 auprc_balanced = average_precision_score(balanced_errors, balanced_uncertainty)
                
#                 logging.info(f"Balanced sampling: AUROC={auroc_balanced:.4f}, AUPRC={auprc_balanced:.4f}")
                
#                 # Return balanced metrics for severe imbalance cases
#                 return auroc_balanced, auprc_balanced
        
#         # Standard calculation for more balanced cases
#         auroc = roc_auc_score(errors, uncertainty)
#         auprc = average_precision_score(errors, uncertainty)
        
#         # Add informative log statement
#         logging.info(f"AUROC and AUPRC measure how well uncertainty predicts errors, not segmentation performance")
        
#         # Verify the output is valid
#         if np.isnan(auroc) or np.isnan(auprc):
#             return float('nan'), float('nan')
            
#     except Exception as e:
#         # Catch any remaining calculation errors
#         logging.error(f"Error calculating AUROC/AUPRC: {e}")
#         return float('nan'), float('nan')
    
#     return auroc, auprc

def calculate_segmentation_metrics(predictions, ground_truth, threshold=0.5):
    """Calculate AUROC and AUPRC for the segmentation task itself (not uncertainty).
    
    Args:
        predictions: Predicted probabilities [B, H, W]
        ground_truth: Ground truth binary mask [B, H, W]
        threshold: Threshold for binarizing predictions (default: 0.5)
        
    Returns:
        Dictionary with segmentation metrics
    """
    # Flatten predictions and ground truth
    pred_flat = predictions.view(-1).cpu().numpy()
    gt_flat = ground_truth.view(-1).cpu().numpy()
    
    # Calculate class balance
    pos_rate = np.mean(gt_flat)
    
    logging.info(f"Positive class rate in ground truth: {pos_rate:.4f}")
    
    try:
        # Calculate AUROC and AUPRC for segmentation (predicting positive class)
        seg_auroc = roc_auc_score(gt_flat, pred_flat)
        seg_auprc = average_precision_score(gt_flat, pred_flat)
        
        # Binary predictions for other metrics
        pred_binary = (pred_flat > threshold).astype(np.int32)
        
        # True/false positives/negatives
        tp = np.sum((pred_binary == 1) & (gt_flat == 1))
        fp = np.sum((pred_binary == 1) & (gt_flat == 0))
        tn = np.sum((pred_binary == 0) & (gt_flat == 0))
        fn = np.sum((pred_binary == 0) & (gt_flat == 1))
        
        # Calculate precision, recall, specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'seg_auroc': seg_auroc,
            'seg_auprc': seg_auprc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }
    except Exception as e:
        logging.error(f"Error calculating segmentation metrics: {e}")
        return {
            'seg_auroc': float('nan'),
            'seg_auprc': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'specificity': float('nan')
        }

def calculate_negative_log_likelihood(predictions, targets, epsilon=1e-9):
    """Calculates the pixel-wise Negative Log-Likelihood (NLL).

    Args:
        predictions (torch.Tensor): Predicted probabilities [H, W].
        targets (torch.Tensor): Ground truth labels (0 or 1) [H, W].
        epsilon (float): Small value to prevent log(0).

    Returns:
        float: Mean NLL over all pixels.
    """
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    nll = -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
    return nll.mean().item()

def calculate_uncertainty_error_dice(uncertainty_map, predictions_binary, targets, uncertainty_threshold=0.2):
    """Calculates the Dice score between high uncertainty regions and error regions.

    Args:
        uncertainty_map (torch.Tensor): Uncertainty values (e.g., std dev) [H, W].
        predictions_binary (torch.Tensor): Binarized predictions (0 or 1) [H, W].
        targets (torch.Tensor): Ground truth labels (0 or 1) [H, W].
        uncertainty_threshold (float): Threshold to define high uncertainty regions.

    Returns:
        float: Dice score between high uncertainty and error masks.
    """
    high_uncertainty_mask = (uncertainty_map > uncertainty_threshold).float()
    error_mask = (predictions_binary != targets).float()

    intersection = (high_uncertainty_mask * error_mask).sum()
    denominator = high_uncertainty_mask.sum() + error_mask.sum()

    if denominator == 0:
        # If both masks are empty, perfect overlap (Dice=1) or undefined?
        # Let's return 1 if intersection is also 0, else 0.
        return 1.0 if intersection == 0 else 0.0
        
    dice = (2.0 * intersection) / (denominator + 1e-8)
    return dice.item()

