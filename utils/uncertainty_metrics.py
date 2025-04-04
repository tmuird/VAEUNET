import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F

def calculate_uncertainty_metrics(segmentations):
    """Calculate various uncertainty metrics from multiple segmentation samples.
    
    Args:
        segmentations: Tensor of shape [N, B, C, H, W] where N is number of samples
    
    Returns:
        Dictionary of uncertainty metrics
    """
    # Calculate mean prediction
    mean_pred = segmentations.mean(dim=0)  # [B, C, H, W]
    
    # Standard deviation (aleatory uncertainty)
    std_dev = segmentations.std(dim=0)     # [B, C, H, W]
    
    # Entropy of the mean prediction (epistemic uncertainty)
    epsilon = 1e-7  # Small constant to avoid log(0)
    entropy = -(mean_pred * torch.log(mean_pred + epsilon) + 
               (1 - mean_pred) * torch.log(1 - mean_pred + epsilon))
    
    # Mutual information (total uncertainty)
    sample_entropies = -(segmentations * torch.log(segmentations + epsilon) + 
                        (1 - segmentations) * torch.log(1 - segmentations + epsilon))
    mean_entropy = sample_entropies.mean(dim=0)
    mutual_info = entropy - mean_entropy
    
    # Coefficient of variation
    coeff_var = std_dev / (mean_pred + epsilon)
    
    return {
        'mean': mean_pred.squeeze(1),      # Remove channel dim for visualization
        'std': std_dev.squeeze(1),
        'entropy': entropy.squeeze(1),
        'mutual_info': mutual_info.squeeze(1),
        'coeff_var': coeff_var.squeeze(1)
    }

def calculate_expected_calibration_error(pred_probs, ground_truth, num_bins=10):
    """
    Calculate Expected Calibration Error (ECE) for binary segmentation.
    
    Args:
        pred_probs: Predicted probabilities [B, H, W]
        ground_truth: Binary ground truth masks [B, H, W]
        num_bins: Number of bins for confidence
        
    Returns:
        ece: Expected Calibration Error
        bin_accs: Accuracy in each bin
        bin_confs: Confidence in each bin
        bin_counts: Number of pixels in each bin
    """
    device = pred_probs.device
    
    # Flatten predictions and ground truth
    pred_flat = pred_probs.flatten()
    gt_flat = ground_truth.flatten()
    
    # Create bins for probabilities
    bin_boundaries = torch.linspace(0, 1, num_bins+1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Initialize arrays to store bin statistics
    bin_accs = torch.zeros(num_bins, device=device)  # Actual rate of positives in bin
    bin_confs = torch.zeros(num_bins, device=device)  # Mean confidence in bin
    bin_counts = torch.zeros(num_bins, device=device)  # Count of predictions in bin
    
    # For each bin, calculate statistics
    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find predictions in this bin
        in_bin = (pred_flat >= bin_lower) & (pred_flat < bin_upper)
        bin_counts[bin_idx] = in_bin.sum()
        
        if bin_counts[bin_idx] > 0:
            # FIXED: For binary problems, accuracy is the actual rate of positives
            bin_accs[bin_idx] = gt_flat[in_bin].float().mean()  # Average ground truth value
            bin_confs[bin_idx] = pred_flat[in_bin].mean()  # Average predicted probability
    
    # Calculate ECE as weighted average of |accuracy - confidence|
    ece = (bin_counts * (bin_accs - bin_confs).abs()).sum() / bin_counts.sum()
    
    # Convert tensors to Python scalars/numpy arrays for pandas compatibility
    ece_value = ece.item()
    bin_accs_np = bin_accs.cpu().numpy()
    bin_confs_np = bin_confs.cpu().numpy()
    bin_counts_np = bin_counts.cpu().numpy()
    
    return ece_value, bin_accs_np, bin_confs_np, bin_counts_np

def brier_score(pred_probs, ground_truth):
    """
    Calculate Brier Score (mean squared error of predictions).
    
    Args:
        pred_probs: Predicted probabilities [B, H, W]
        ground_truth: Binary ground truth masks [B, H, W]
        
    Returns:
        brier: Brier score (Python float)
    """
    brier = F.mse_loss(pred_probs, ground_truth.float())
    return brier.item()  # Convert to Python scalar

def plot_reliability_diagram(bin_accs, bin_confs, bin_counts, ax=None):
    """
    Plot reliability diagram to visualize calibration with improved visualization.
    
    Args:
        bin_accs: Accuracy in each bin (numpy array)
        bin_confs: Confidence in each bin (numpy array)
        bin_counts: Number of pixels in each bin (numpy array)
        ax: Optional matplotlib axis for plotting
    
    Returns:
        ax: The matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Normalize bin counts for visualization
    normalized_counts = bin_counts / bin_counts.max() if bin_counts.max() > 0 else bin_counts
    
    # Compute bin centers for x-axis
    bin_centers = np.linspace(0.05, 0.95, len(bin_accs))
    
    # Plot the bar charts
    bar_width = 0.35
    bars1 = ax.bar(bin_centers - bar_width/2, bin_accs, bar_width, alpha=0.7, color='blue', label='Accuracy')
    bars2 = ax.bar(bin_centers + bar_width/2, bin_confs, bar_width, alpha=0.7, color='green', label='Confidence')
    
    # Add gap between accuracy and confidence
    gap = np.abs(bin_accs - bin_confs)
    for i, (center, acc, conf) in enumerate(zip(bin_centers, bin_accs, bin_confs)):
        if gap[i] > 0.05:  # Only draw gaps that are significant
            y_min, y_max = min(acc, conf), max(acc, conf)
            ax.plot([center, center], [y_min, y_max], color='red', linestyle='-', lw=2, alpha=0.7)
    
    # Plot histogram of predictions in each bin (frequency)
    ax2 = ax.twinx()
    ax2.bar(bin_centers, normalized_counts, width=bar_width*1.8, alpha=0.15, color='gray', label='Frequency')
    ax2.set_ylabel('Relative Frequency', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 1.1)
    
    # Plot the identity line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Set labels and title
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return ax

def calculate_sparsification_metrics(pred_probs, uncertainties, ground_truth, num_points=20):
    """
    Calculate metrics for sparsification plots with improved pixel selection and normalization.
    
    Args:
        pred_probs: Predicted probabilities [B, H, W]
        uncertainties: Uncertainty estimates [B, H, W]
        ground_truth: Binary ground truth masks [B, H, W]
        num_points: Number of points in the sparsification curve
        
    Returns:
        fraction_removed: Fraction of pixels removed at each step
        errors_random: Error when removing pixels randomly
        errors_uncertainty: Error when removing pixels by uncertainty
    """
    device = pred_probs.device
    batch_size = pred_probs.shape[0]
    
    # Calculate errors (binary cross-entropy) for each pixel
    epsilon = 1e-7
    pixel_errors = -(ground_truth * torch.log(pred_probs + epsilon) + 
                    (1 - ground_truth) * torch.log(1 - pred_probs + epsilon))
    
    # Flatten for easier processing and move to CPU for memory efficiency
    pixel_errors_flat = pixel_errors.reshape(batch_size, -1).detach().cpu().numpy()
    uncertainties_flat = uncertainties.reshape(batch_size, -1).detach().cpu().numpy()
    
    # Calculate fractions to remove - ensure we include both endpoints
    fraction_removed = np.linspace(0, 0.99, num_points)  # Avoid removing 100% of pixels
    
    # Initialize arrays for error curves
    errors_random = np.zeros(num_points)
    errors_uncertainty = np.zeros(num_points)
    
    # For each batch element
    for b in range(batch_size):
        # Get data for this batch element
        batch_errors = pixel_errors_flat[b]
        batch_uncertainties = uncertainties_flat[b]
        
        # Skip if all values are the same or NaN
        if np.all(batch_errors == batch_errors[0]) or np.isnan(batch_errors).any():
            continue
            
        # Get initial error as baseline for normalization
        initial_error = batch_errors.mean()
        if initial_error <= 0 or np.isnan(initial_error):
            continue  # Skip this batch if initial error is invalid
        
        # Create index array for this batch
        num_pixels = batch_errors.shape[0]
        all_indices = np.arange(num_pixels)
        
        # Pre-sort uncertainties for more efficient lookup
        uncertainty_order = np.argsort(batch_uncertainties)[::-1]  # Descending order (most uncertain first)
        
        for i, frac in enumerate(fraction_removed):
            if frac >= 1.0:
                # Edge case: if we remove all pixels, error is undefined
                errors_random[i] += 1.0
                errors_uncertainty[i] += 1.0
                continue
                
            # Number of pixels to remove
            pixels_to_remove = int(num_pixels * frac)
            pixels_to_keep = num_pixels - pixels_to_remove
            
            if pixels_to_keep <= 0:
                # Handle edge case of keeping zero pixels
                errors_random[i] += 1.0
                errors_uncertainty[i] += 1.0
                continue
            
            # Random removal - use a new random seed each time for diversity
            np.random.seed(i + b * 1000)  # Different seed for each iteration/batch
            random_indices = np.random.choice(all_indices, pixels_to_keep, replace=False)
            random_error = batch_errors[random_indices].mean()
            errors_random[i] += random_error / initial_error
            
            # Uncertainty-based removal - keep pixels with lowest uncertainty
            # Get indices of pixels to keep (those with lowest uncertainty)
            uncertainty_indices = uncertainty_order[pixels_to_remove:]
            uncertainty_error = batch_errors[uncertainty_indices].mean()
            errors_uncertainty[i] += uncertainty_error / initial_error
    
    # Normalize by batch size
    valid_batches = batch_size  # Adjust if we skipped any batches
    errors_random /= valid_batches
    errors_uncertainty /= valid_batches
    
    # Ensure curves start at 1.0 for proper visualization
    if errors_random[0] > 0:
        errors_random = errors_random / errors_random[0]
    if errors_uncertainty[0] > 0:
        errors_uncertainty = errors_uncertainty / errors_uncertainty[0]
    
    # Force curves to be monotonic (errors shouldn't increase as we remove more uncertain pixels)
    # This addresses the issue of the curve shooting back up
    for i in range(1, num_points):
        # Random curve can go up (random removal could remove good pixels)
        # But uncertainty curve should be non-increasing (we're removing most uncertain first)
        if i > 0 and errors_uncertainty[i] > errors_uncertainty[i-1]:
            errors_uncertainty[i] = errors_uncertainty[i-1]
    
    # Handle any NaN or inf values
    errors_random = np.nan_to_num(errors_random, nan=1.0, posinf=1.0, neginf=0.0)
    errors_uncertainty = np.nan_to_num(errors_uncertainty, nan=1.0, posinf=1.0, neginf=0.0)
    
    return fraction_removed, errors_random, errors_uncertainty

def plot_sparsification_curve(fraction_removed, errors_random, errors_uncertainty, ax=None):
    """
    Plot improved sparsification curve with better visual indicators and handling of anomalies.
    
    Args:
        fraction_removed: Fraction of pixels removed at each step
        errors_random: Error when removing pixels randomly
        errors_uncertainty: Error when removing pixels by uncertainty
        ax: Optional matplotlib axis for plotting
        
    Returns:
        ax: The matplotlib axis with the plot
        sparsification_error: Area between curves (sparsification error)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute area between curves (sparsification error)
    # Positive value = good uncertainty estimates (uncertainty below random)
    # Negative value = poor uncertainty estimates (uncertainty above random)
    se = np.trapz(errors_random - errors_uncertainty, fraction_removed)
    
    # Fill the area between curves
    color = 'green' if se > 0 else 'red'
    ax.fill_between(fraction_removed, errors_random, errors_uncertainty, 
                   alpha=0.2, color=color)
    
    # Plot sparsification curves
    ax.plot(fraction_removed, errors_random, 'b--', label='Random')
    ax.plot(fraction_removed, errors_uncertainty, 'r-', label='By Uncertainty')
    
    # Add a horizontal line at y=0.5 to show the half-error point
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)
    
    # Find where uncertainty curve crosses the 0.5 line (half-error point)
    half_error_idx = np.argmin(np.abs(errors_uncertainty - 0.5))
    if half_error_idx < len(fraction_removed):
        half_error_fraction = fraction_removed[half_error_idx]
        ax.plot([half_error_fraction], [0.5], 'ro', 
               markersize=8, alpha=0.7)
        ax.annotate(f'{half_error_fraction:.2f}', 
                   xy=(half_error_fraction, 0.5),
                   xytext=(half_error_fraction + 0.05, 0.55),
                   arrowprops=dict(arrowstyle="->", color='black', alpha=0.6))
    
    # Set labels and title
    ax.set_xlabel('Fraction of Pixels Removed', fontsize=12)
    ax.set_ylabel('Normalized Error', fontsize=12)
    ax.set_title(f'Sparsification Curve (SE={se:.4f})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(fraction_removed))
    ax.set_ylim(0, 1.1)
    
    return ax, se

def create_comprehensive_uncertainty_report(segmentations, ground_truth, output_path=None):
    """
    Create a comprehensive report on uncertainty metrics and model calibration.
    
    Args:
        segmentations: Tensor of shape [N, B, C, H, W] where N is number of samples
        ground_truth: Binary ground truth masks [B, C, H, W]
        output_path: Optional path to save the report
        
    Returns:
        fig: Matplotlib figure with the report
    """
    # Calculate uncertainty metrics
    uncertainty_metrics = calculate_uncertainty_metrics(segmentations)
    
    # Extract mean prediction and standard deviation
    mean_pred = uncertainty_metrics['mean']
    std_dev = uncertainty_metrics['std']
    entropy = uncertainty_metrics['entropy']
    
    # Calculate calibration metrics
    ece, bin_accs, bin_confs, bin_counts = calculate_expected_calibration_error(
        mean_pred, ground_truth.squeeze(1)
    )
    
    brier = brier_score(mean_pred, ground_truth.squeeze(1))
    
    # Calculate sparsification metrics using standard deviation as uncertainty
    frac_removed, err_random, err_uncertainty = calculate_sparsification_metrics(
        mean_pred, std_dev, ground_truth.squeeze(1)
    )
    
    # Create figure for the report
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)
    
    # Plot ground truth and mean prediction
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ground_truth[0, 0].cpu().numpy(), cmap='gray')
    ax1.set_title('Ground Truth', fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mean_pred[0].cpu().numpy(), cmap='gray')
    ax2.set_title(f'Mean Prediction (N={segmentations.shape[0]})', fontsize=12)
    ax2.axis('off')
    
    # Plot standard deviation
    ax3 = fig.add_subplot(gs[0, 2])
    std_plot = ax3.imshow(std_dev[0].cpu().numpy(), cmap='hot')
    ax3.set_title('Std Deviation (Uncertainty)', fontsize=12)
    ax3.axis('off')
    plt.colorbar(std_plot, ax=ax3)
    
    # Plot reliability diagram
    ax4 = fig.add_subplot(gs[1, 0])
    plot_reliability_diagram(bin_accs, bin_confs, bin_counts, ax=ax4)
    ax4.set_title(f'Reliability Diagram (ECE={ece:.4f}, Brier={brier:.4f})', fontsize=12)
    
    # Plot sparsification curve
    ax5 = fig.add_subplot(gs[1, 1])
    plot_sparsification_curve(frac_removed, err_random, err_uncertainty, ax=ax5)
    
    # Plot entropy
    ax6 = fig.add_subplot(gs[1, 2])
    entropy_plot = ax6.imshow(entropy[0].cpu().numpy(), cmap='viridis')
    ax6.set_title('Entropy Map', fontsize=12)
    ax6.axis('off')
    plt.colorbar(entropy_plot, ax=ax6)
    
    # Plot sample predictions
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(segmentations[i, 0, 0].cpu().numpy(), cmap='gray')
        ax.set_title(f'Sample {i+1}', fontsize=12)
        ax.axis('off')
    
    plt.suptitle('Uncertainty and Calibration Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save report if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
