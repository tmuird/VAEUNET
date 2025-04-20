import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
import logging
from tqdm import tqdm
import gc
def calculate_segmentation_metrics_chunked(processed_ids, temp_pixel_data_dir, threshold=0.5, chunk_size=100000):
    """Calculate segmentation metrics incrementally over chunks to avoid memory issues."""

    # incremental calculation
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    total_elements = 0
    
    all_scores = []
    all_labels = []
    
    logging.info("Calculating segmentation metrics in chunks...")
    
    for img_id in tqdm(processed_ids, desc="Processing metrics by image"):
        try:
            pred_path = temp_pixel_data_dir / f"{img_id}_pred_flat.npy"
            gt_path = temp_pixel_data_dir / f"{img_id}_gt_flat.npy"
            
            if pred_path.exists() and gt_path.exists():
                # Process this image data
                pred = np.load(pred_path)
                gt = np.load(gt_path)
                
                # For AUROC/AUPRC store only a random subset to manage memory
                if len(pred) > 10000:
                    indices = np.random.choice(len(pred), 10000, replace=False)
                    all_scores.append(pred[indices])
                    all_labels.append(gt[indices])
                else:
                    all_scores.append(pred)
                    all_labels.append(gt)
                
                # Process predictions in chunks to calculate TP, FP, TN, FN
                for i in range(0, len(pred), chunk_size):
                    end_idx = min(i + chunk_size, len(pred))
                    pred_chunk = pred[i:end_idx]
                    gt_chunk = gt[i:end_idx]
                    
                    # Binarize predictions
                    pred_binary = (pred_chunk > threshold).astype(np.int32)
                    gt_binary = gt_chunk.astype(np.int32)
                    
                    # Calculate confusion matrix elements
                    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
                    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
                    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
                    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
                    
                    # Accumulate counts
                    total_tp += tp
                    total_fp += fp
                    total_tn += tn
                    total_fn += fn
                    total_elements += len(pred_chunk)
            
        except Exception as e:
            logging.warning(f"Error processing file for {img_id}: {e}")
    
    #final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    accuracy = (total_tp + total_tn) / total_elements if total_elements > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Concatenate stored scores
    try:
        scores_concat = np.concatenate(all_scores)
        labels_concat = np.concatenate(all_labels)
        
        #  ROC curve and AUC
        fpr, tpr, _ = roc_curve(labels_concat, scores_concat)
        roc_auc = auc(fpr, tpr)
        
        # PR curve and AUC
        precision_curve, recall_curve, _ = precision_recall_curve(labels_concat, scores_concat)
        pr_auc = auc(recall_curve, precision_curve)
        
        del scores_concat, labels_concat
        gc.collect()
        
    except Exception as e:
        logging.error(f"Error calculating ROC/PR metrics: {e}")
        roc_auc = float('nan')
        pr_auc = float('nan')
    
    return {
        'seg_auroc': roc_auc,
        'seg_auprc': pr_auc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
        'f1_score': f1_score
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



def calculate_segmentation_metrics(predictions, ground_truth, threshold=0.5):
    """Calculate AUROC and AUPRC for the segmentation task itself (not uncertainty)."""
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
    """Calculates the pixel-wise Negative Log-Likelihood (NLL) """
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    nll = -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
    return nll.mean().item()

def calculate_uncertainty_error_dice(uncertainty_map, predictions_binary, targets, uncertainty_threshold=0.2):
    """Calculates the Dice score between high uncertainty regions and error regions."""
    high_uncertainty_mask = (uncertainty_map > uncertainty_threshold).float()
    error_mask = (predictions_binary != targets).float()

    intersection = (high_uncertainty_mask * error_mask).sum()
    denominator = high_uncertainty_mask.sum() + error_mask.sum()

    if denominator == 0:

        return 1.0 if intersection == 0 else 0.0
        
    dice = (2.0 * intersection) / (denominator + 1e-8)
    return dice.item()

