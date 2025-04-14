import os
import torch
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.gridspec as gridspec
from textwrap import wrap
import wandb  # Import wandb
from PIL import Image  # Import Image for visualization

from utils.data_loading import IDRIDDataset, load_image  # Add load_image import
from unet.unet_resnet import UNetResNet
from utils.uncertainty_metrics import (
    calculate_expected_calibration_error, 
    brier_score,
    calculate_sparsification_metrics,
    plot_reliability_diagram,
    plot_sparsification_curve,
    calculate_uncertainty_error_auc
)
from utils.tensor_utils import ensure_dict_python_scalars, fix_dataframe_tensors
from visualize_vae import get_segmentation_distribution, track_memory

##############################################################################
# Additional helper functions for global AUROC / AUPRC plotting,
# plus storing & plotting data from multiple images / models
##############################################################################
def plot_global_roc_pr(
    all_errors, 
    all_uncertainties,
    output_dir,
    model_label="Model",
    prefix="",
    log_wandb=False  # Add wandb logging flag
):
    """
    Plots the global AUROC and AUPRC curves for pixel-level uncertainty vs. errors.
    Saves plots into output_dir with optional prefix appended to filenames.

    Args:
        all_errors: 1D numpy array of shape (N_pixels,) with 0/1 for correct/incorrect
        all_uncertainties: 1D numpy array of shape (N_pixels,) with continuous uncertainty
        output_dir: Path to output directory
        model_label: e.g. 'High-KL Model'
        prefix: optional string to prepend to filenames, e.g. "global_" => "global_roc.png"
        log_wandb: If True, log plots to W&B
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute ROC
    fpr, tpr, _ = roc_curve(all_errors, all_uncertainties)
    roc_auc = auc(fpr, tpr)

    # Compute PR
    precision, recall, _ = precision_recall_curve(all_errors, all_uncertainties)
    prc_auc = auc(recall, precision)

    # Plot ROC
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'{model_label} (AUC={roc_auc:.4f})', lw=2)
    plt.plot([0,1],[0,1],'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Global ROC Curve ({model_label})')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    out_roc = output_dir / f"{prefix}roc_curve.png"
    plt.savefig(out_roc, dpi=300, bbox_inches='tight')
    plt.close()
    if log_wandb:
        try:
            wandb.log({f"plots/{prefix}roc_curve": wandb.Image(str(out_roc))})
        except Exception as e:
            logging.warning(f"Could not log ROC curve to W&B: {e}")

    # Plot PR
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f'{model_label} (AUC={prc_auc:.4f})', lw=2)
    baseline = np.sum(all_errors) / (len(all_errors)+1e-9)  # fraction of positives
    plt.plot([0,1],[baseline, baseline],'k--', label=f'Chance={baseline:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Global Precision-Recall Curve ({model_label})')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    out_pr = output_dir / f"{prefix}pr_curve.png"
    plt.savefig(out_pr, dpi=300, bbox_inches='tight')
    plt.close()
    if log_wandb:
        try:
            wandb.log({f"plots/{prefix}pr_curve": wandb.Image(str(out_pr))})
        except Exception as e:
            logging.warning(f"Could not log PR curve to W&B: {e}")

    logging.info(f"Global ROC/AUPR plots saved: {out_roc}, {out_pr}")
    logging.info(f"Global AUROC={roc_auc:.4f}, AUPRC={prc_auc:.4f}")
    if log_wandb:
        try:
            wandb.summary[f"{prefix}global_auroc"] = roc_auc
            wandb.summary[f"{prefix}global_auprc"] = prc_auc
        except Exception as e:
            logging.warning(f"Could not log global AUCs to W&B summary: {e}")


def store_sparsification_data(save_path, model_spars_data):
    """
    Saves the per-image sparsification data to a numpy file (or pickled file).
    model_spars_data: list of dicts, each with:
      {
        'img_id': str,
        'frac_removed': np.array,
        'err_random': np.array,
        'err_uncertainty': np.array
      }
    """
    np.save(save_path, model_spars_data, allow_pickle=True)
    logging.info(f"Stored sparsification data to {save_path}")

def store_uncertainty_data(save_path, model_uncert_data):
    """
    Saves the correct vs. incorrect pixel uncertainty data to disk.
    model_uncert_data: list of dicts, each with:
      {
        'img_id': str,
        'uncertainties_correct': np.array,
        'uncertainties_incorrect': np.array
      }
    """
    np.save(save_path, model_uncert_data, allow_pickle=True)
    logging.info(f"Stored uncertainty data to {save_path}")

##############################################################################


def get_args():
    parser = argparse.ArgumentParser(description='Analyze model performance, calibration and uncertainty')
    parser.add_argument('--model', '-m', default='best_model.pth', metavar='FILE', help='Model file')
    parser.add_argument('--lesion_type', type=str, required=True, default='EX', 
                      choices=['EX', 'HE', 'MA','OD'], help='Lesion type')
    parser.add_argument('--attention', dest='use_attention', action='store_true', help='Enable attention mechanism (default)')
    parser.add_argument('--no-attention', dest='use_attention', action='store_false', help='Disable attention mechanism')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--samples', type=int, default=30, help='Number of samples for ensemble prediction')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size (0 for full image)')
    parser.add_argument('--overlap', type=int, default=100, help='Overlap between patches')
    parser.add_argument('--output_dir', type=str, default='./analysis', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for resizing (default: 1.0)')
    
    # Keep temp_values for temperature analysis
    parser.add_argument('--temp_values', type=float, nargs='+', 
                       default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
                       help='Temperature values to analyze for temperature scaling')
    
    # Additional flags to control new features
    parser.add_argument('--plot_roc_pr', action='store_true',
                        help='If set, plot global pixel-level ROC and PR curves based on uncertainties')
    parser.add_argument('--store_data', action='store_true',
                        help='If set, store per-image sparsification and uncertainty arrays to disk for later comparisons')
    parser.add_argument('--model_label', type=str, default='Model',
                        help='Label to use when plotting ROC/PR (e.g. "High-KL")')
    
    # W&B Arguments
    parser.add_argument('--wandb_project', type=str, default='VAEUNET-Analysis', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team/user name)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (defaults to auto-generated)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging')

    parser.set_defaults(use_attention=True, enable_dropout=False)
    args = parser.parse_args()
    
    # Convert patch_size=0 to None for full image mode
    if args.patch_size == 0:
        args.patch_size = None
        
    return args


@track_memory
def analyze_model(model, dataset, args, wandb_run=None):  # Add wandb_run parameter
    """Unified analysis function that handles both uncertainty and calibration analysis,
       also storing per-image data for later comparisons, and optionally plotting
       global pixel-level ROC/PR curves."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.lesion_type}_T{args.temperature}_N{args.samples}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data storage
    metrics_data = []
    all_predictions = []
    all_ground_truths = []
    
    # Additional data structures for advanced comparisons
    model_sparsification_data = []  # for storing (frac_removed, err_random, err_uncertainty) per image
    model_uncert_data = []          # for storing correct vs. incorrect pixel uncertainties
    global_errors = []              # global arrays to plot ROC/PR
    global_uncertainties = []

    processed_ids = set()
    log_wandb = wandb_run is not None  # Check if W&B is active
    
    for i in tqdm(range(len(dataset)), desc="Analyzing images"):
        sample = dataset[i]
        img_id = sample['img_id']
        
        if img_id in processed_ids:
            continue
        processed_ids.add(img_id)
        logging.info(f"Processing image {img_id}")
        
        try:
            # Generate segmentations 
            segmentations, mask, mu, logvar = get_segmentation_distribution(
                model, img_id, dataset=dataset, num_samples=args.samples,
                patch_size=args.patch_size, overlap=args.overlap, 
                temperature=args.temperature, enable_dropout=False,
                batch_size=args.batch_size
            )
            
            # Move to CPU
            segmentations_cpu = segmentations.cpu()
            mask_cpu = mask.cpu()
            
            # Mean, std
            mean_pred = segmentations_cpu.mean(dim=0) # Shape [1, H, W]
            std_dev = segmentations_cpu.std(dim=0)   # Shape [1, H, W]

            # --- Start: Restore Metrics Calculation ---
            # Flatten predictions and ground truth for metrics
            pred_flat = mean_pred[0].flatten().numpy()
            gt_flat = mask_cpu[0,0].flatten().numpy()
            gt_flat = np.round(gt_flat).astype(int) # Ensure GT is 0 or 1

            # Calibration metrics
            ece, bin_accs, bin_confs, bin_counts = calculate_expected_calibration_error(
                mean_pred[0], mask_cpu[0, 0] # Use mean_pred[0] which is [H, W]
            )
            brier = brier_score(mean_pred[0], mask_cpu[0, 0])
            pred_binary = (mean_pred[0] > 0.5).float()
            dice_tensor = (2.0*(pred_binary*mask_cpu[0,0]).sum())/(pred_binary.sum()+mask_cpu[0,0].sum()+1e-8)
            dice = float(dice_tensor.item())

            # Store global predictions ALWAYS for calibration/temp sweep
            all_predictions.append(pred_flat)
            all_ground_truths.append(gt_flat)

            # Uncertainty metrics
            se = 0.0
            uncertain_pixel_percent = 0.0
            auroc = 0.0
            auprc = 0.0
            frac_removed, err_random, err_uncertainty = calculate_sparsification_metrics(
                mean_pred, std_dev, mask_cpu[:,0], num_points=20
            )
            if err_random[0]>0:
                norm_random = err_random/err_random[0]
                norm_uncertainty = err_uncertainty/err_random[0]
            else:
                norm_random = err_random
                norm_uncertainty = err_uncertainty
            se = float(np.trapz(norm_random - norm_uncertainty, frac_removed))

            uncertainty_threshold = 0.2 # Example threshold
            uncertain_percent_tensor = (std_dev[0]>uncertainty_threshold).float().mean()*100
            uncertain_pixel_percent = float(uncertain_percent_tensor.item())

            # Store per-image sparsification data if requested
            model_sparsification_data.append({
                'img_id': img_id,
                'frac_removed': frac_removed,
                'err_random': err_random,
                'err_uncertainty': err_uncertainty
            })

            # Prepare data for correct/incorrect uncertainty boxplot/storage
            pred_binary_np = pred_binary.numpy()
            gt_np = mask_cpu[0,0].numpy()
            correct_mask = (pred_binary_np == gt_np)
            incorrect_mask = ~correct_mask
            unc_map = std_dev[0].numpy()

            model_uncert_data.append({
                'img_id': img_id,
                'uncertainties_correct': unc_map[correct_mask],
                'uncertainties_incorrect': unc_map[incorrect_mask]
            })

            # Prepare data for global ROC/PR plots
            errors = (pred_binary_np != gt_np).astype(np.int32).flatten()
            uncertainties_flat = unc_map.flatten()
            global_errors.append(errors)
            global_uncertainties.append(uncertainties_flat)

            # Calculate pixel-level AUROC/AUPRC for uncertainty vs error
            auroc, auprc = calculate_uncertainty_error_auc(mean_pred[0], mask_cpu[0,0], std_dev[0])

            # Other calibration metrics
            max_calibration_error = float(np.max(np.abs(bin_accs - bin_confs)))
            mean_abs_calib_error = float(np.mean(np.abs(bin_accs - bin_confs)))

            # Assemble metrics dictionary
            metrics_dict = {
                'img_id': str(img_id),
                'dice': dice,
                'ece': ece,
                'brier': brier,
                'sparsification_error': se,
                'uncertain_pixel_percent': uncertain_pixel_percent,
                'max_calibration_error': max_calibration_error,
                'mean_abs_calib_error': mean_abs_calib_error,
                'error_auroc': auroc,
                'error_auprc': auprc
            }
            metrics_dict = ensure_dict_python_scalars(metrics_dict)
            metrics_data.append(metrics_dict)
            # --- End: Restore Metrics Calculation ---

            # --- Start: Add Visualization Logging (as before) ---
            if log_wandb:
                try:
                    # 1. Load Original Image (scaled appropriately)
                    img_file = dataset.images_dir / f"{img_id}.jpg"
                    img_pil = load_image(img_file).convert('RGB')
                    w, h = img_pil.size
                    newW, newH = int(dataset.scale * w), int(dataset.scale * h)
                    img_pil_scaled = img_pil.resize((newW, newH), resample=Image.BICUBIC)
                    
                    if dataset.is_full_image:
                        img_array_vis = np.array(img_pil_scaled).astype(np.float32) / 255.0
                        img_tensor_vis = torch.from_numpy(img_array_vis.transpose(2, 0, 1))
                        dummy_mask_tensor = torch.zeros_like(img_tensor_vis[0:1]) 
                        img_cropped_tensor, _ = dataset.crop_to_fundus(img_tensor_vis, dummy_mask_tensor, img_array_vis.transpose(2, 0, 1))
                        img_vis = img_cropped_tensor.permute(1, 2, 0).numpy()
                        del img_tensor_vis, dummy_mask_tensor, img_cropped_tensor
                    else:
                        img_vis = np.array(img_pil_scaled)

                    if img_vis.max() <= 1.0:
                        img_vis = (img_vis * 255).clip(0, 255).astype(np.uint8)
                    else:
                        img_vis = img_vis.clip(0, 255).astype(np.uint8)

                    gt_vis = (mask_cpu[0, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
                    pred_vis_prob = mean_pred[0].numpy()
                    pred_vis_prob = (pred_vis_prob - pred_vis_prob.min()) / (pred_vis_prob.max() - pred_vis_prob.min() + 1e-8)
                    pred_vis = (pred_vis_prob * 255).clip(0, 255).astype(np.uint8)
                    uncert_vis_raw = std_dev[0].numpy()
                    uncert_vis_raw = (uncert_vis_raw - uncert_vis_raw.min()) / (uncert_vis_raw.max() - uncert_vis_raw.min() + 1e-8)
                    uncert_vis = (uncert_vis_raw * 255).clip(0, 255).astype(np.uint8)

                    wandb.log({
                        f"visualizations/{img_id}/original_image": wandb.Image(img_vis),
                        f"visualizations/{img_id}/ground_truth": wandb.Image(gt_vis),
                        f"visualizations/{img_id}/mean_prediction": wandb.Image(pred_vis),
                        f"visualizations/{img_id}/uncertainty_map": wandb.Image(uncert_vis, caption=f"Mean Std: {std_dev[0].mean().item():.4f}"),
                        "image_index": i
                    })
                    
                    del img_vis, gt_vis, pred_vis, uncert_vis, img_pil, img_pil_scaled
                    if 'img_array_vis' in locals(): del img_array_vis

                except Exception as e:
                    logging.warning(f"Could not log visualizations to W&B for {img_id}: {e}")
                    import traceback
                    traceback.print_exc()
            # --- End: Add Visualization Logging ---

            # Log per-image metrics to W&B if active
            if log_wandb:
                try:
                    if 'metrics_dict' in locals(): 
                        wandb_metrics = {f"metrics/{k}": v for k, v in metrics_dict.items() if k != 'img_id'}
                        wandb_metrics['image_index'] = i
                        wandb.log(wandb_metrics)
                    else:
                         logging.warning(f"metrics_dict not found for image {img_id}, skipping W&B metric logging.")
                except Exception as e:
                    logging.warning(f"Could not log per-image metrics to W&B for {img_id}: {e}")
            
            # --- Start: Restore Individual Plot Generation ---
            # Generate individual image reports (conditional on analysis mode)
            img_output_dir = output_dir / "individual_reports"
            img_output_dir.mkdir(exist_ok=True)
            
            # Generate individual calibration report
            try: # Added try-except block
                # Create calibration curve using scikit-learn
                prob_true, prob_pred = calibration_curve(
                    gt_flat, pred_flat, n_bins=10, strategy='uniform'
                )
                
                # Create individual calibration plots
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Plot calibration curve
                ax.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                       label=f'Calibration Curve (ECE={ece:.4f})')
                
                # Plot the diagonal (perfect calibration)
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                
                # Add histogram of prediction confidences
                ax2 = ax.twinx()
                ax2.hist(pred_flat, bins=20, alpha=0.3, density=True, 
                        color='gray', label='Prediction Distribution')
                ax2.set_ylabel('Density')
                
                # Set labels and title
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title(f'Calibration Curve for Image {img_id}')
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                calib_plot_path = img_output_dir / f"{img_id}_calibration_curve.png"
                plt.savefig(calib_plot_path, dpi=200)
                plt.close(fig)
                if log_wandb:
                    try:
                        wandb.log({f"individual_plots/{img_id}_calibration": wandb.Image(str(calib_plot_path)), "image_index": i})
                    except Exception as e:
                        logging.warning(f"Could not log calibration plot to W&B for {img_id}: {e}")
            except Exception as plot_err:
                logging.error(f"Error generating calibration plot for {img_id}: {plot_err}")

            # Generate combined uncertainty metrics visualization
            try: # Added try-except block
                # Create figure with reliability and sparsification plots
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Enhanced reliability plot
                plot_reliability_diagram(bin_accs, bin_confs, bin_counts, ax=axes[0])
                axes[0].set_title(f'Reliability Diagram (ECE={ece:.4f}, MCE={max_calibration_error:.4f})')
                
                # Enhanced sparsification plot
                axes[1], se_value = plot_sparsification_curve(frac_removed, err_random, err_uncertainty, ax=axes[1])
                axes[1].set_title(f'Sparsification Curve (Sparsification Error={se:.4f})')

                plt.suptitle(f'Image {img_id} (Dice={dice:.4f}, Brier={brier:.4f})')
                plt.tight_layout()
                uncert_plot_path = img_output_dir / f"{img_id}_uncertainty_metrics.png"
                plt.savefig(uncert_plot_path, dpi=200)
                plt.close(fig)
                if log_wandb:
                    try:
                        wandb.log({f"individual_plots/{img_id}_uncertainty": wandb.Image(str(uncert_plot_path)), "image_index": i})
                    except Exception as e:
                        logging.warning(f"Could not log uncertainty plot to W&B for {img_id}: {e}")
            except Exception as plot_err:
                logging.error(f"Error generating uncertainty metrics plot for {img_id}: {plot_err}")

                # Generate a more detailed analysis figure
            try: # Added try-except block
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Show image, mask, and prediction
                img_file = dataset.images_dir / f"{img_id}.jpg"
                img_pil = load_image(img_file).convert('RGB')
                w, h = img_pil.size
                newW, newH = int(dataset.scale * w), int(dataset.scale * h)
                img_pil_scaled = img_pil.resize((newW, newH), resample=Image.BICUBIC)
                if dataset.is_full_image:
                    img_array_vis = np.array(img_pil_scaled).astype(np.float32) / 255.0
                    img_tensor_vis = torch.from_numpy(img_array_vis.transpose(2, 0, 1))
                    dummy_mask_tensor = torch.zeros_like(img_tensor_vis[0:1])
                    img_cropped_tensor, _ = dataset.crop_to_fundus(img_tensor_vis, dummy_mask_tensor, img_array_vis.transpose(2, 0, 1))
                    display_img = img_cropped_tensor.permute(1, 2, 0).numpy()
                else:
                    display_img = np.array(img_pil_scaled)
                if display_img.max() <= 1.0: display_img = (display_img * 255).clip(0, 255).astype(np.uint8)
                else: display_img = display_img.clip(0, 255).astype(np.uint8)

                display_mask = (mask_cpu[0, 0].numpy() * 255).clip(0, 255).astype(np.uint8) # Ground truth mask

                axes[0, 0].imshow(display_img)
                axes[0, 0].set_title('Original Image (Scaled/Cropped)')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(display_mask, cmap='gray')
                axes[0, 1].set_title('Ground Truth Mask')
                axes[0, 1].axis('off')

                # Show uncertainty map
                im = axes[1, 0].imshow(std_dev[0].numpy(), cmap='hot')
                plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
                axes[1, 0].set_title(f'Uncertainty Map (Mean Std: {std_dev[0].mean().item():.4f})')
                axes[1, 0].axis('off')
                
                # Show areas of high uncertainty/error
                uncertainty_threshold = 0.2  # Define threshold for high uncertainty
                pred_binary_np = pred_binary.numpy() # Already calculated
                mask_np = mask_cpu[0, 0].numpy() # Already calculated
                uncertain_mask = (std_dev[0].numpy() > uncertainty_threshold).astype(np.float32)
                error_mask = (pred_binary_np != mask_np).astype(np.float32)
                
                # Display composite mask - red where both high uncertainty and errors
                rgb_overlay = np.zeros((*uncertain_mask.shape, 3))
                rgb_overlay[..., 0] = uncertain_mask  # Red channel = high uncertainty
                rgb_overlay[..., 1] = error_mask      # Green channel = errors
                
                axes[1, 1].imshow(rgb_overlay)
                axes[1, 1].set_title('Red: High Uncertainty, Green: Errors, Yellow: Both')
                axes[1, 1].axis('off')
                
                plt.suptitle(f'Detailed Analysis for Image {img_id}', fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
                detailed_plot_path = img_output_dir / f"{img_id}_detailed_analysis.png"
                plt.savefig(detailed_plot_path, dpi=200)
                plt.close(fig)
                if log_wandb:
                    try:
                        wandb.log({f"individual_plots/{img_id}_detailed": wandb.Image(str(detailed_plot_path)), "image_index": i})
                    except Exception as e:
                        logging.warning(f"Could not log detailed plot to W&B for {img_id}: {e}")
            except Exception as plot_err:
                logging.error(f"Error generating detailed analysis plot for {img_id}: {plot_err}")
                import traceback
                traceback.print_exc() # Print detailed traceback for plotting errors
            # --- End: Restore Individual Plot Generation ---

            # Free up memory
            del segmentations_cpu, mask_cpu, mean_pred, std_dev
            if 'metrics_dict' in locals(): del metrics_dict 
            if 'calib_plot_path' in locals(): del calib_plot_path
            if 'uncert_plot_path' in locals(): del uncert_plot_path
            if 'detailed_plot_path' in locals(): del detailed_plot_path
            if 'pred_flat' in locals(): del pred_flat
            if 'gt_flat' in locals(): del gt_flat
            if 'pred_binary' in locals(): del pred_binary
            if 'pred_binary_np' in locals(): del pred_binary_np
            if 'gt_np' in locals(): del gt_np
            if 'unc_map' in locals(): del unc_map
            if 'frac_removed' in locals(): del frac_removed, err_random, err_uncertainty
            if 'display_img' in locals(): del display_img 
            if 'display_mask' in locals(): del display_mask

            torch.cuda.empty_cache() 

        except Exception as e:
            logging.error(f"Error processing image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if args.max_images and len(processed_ids)>=args.max_images:
            break

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = fix_dataframe_tensors(metrics_df)
    for col in metrics_df.columns:
        if col != 'img_id':
            try:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
            except Exception as e:
                logging.warning(f"Could not convert {col} to numeric: {e}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_dir / "analysis_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved metrics data to {metrics_csv_path}")
    
    if log_wandb:
        try:
            wandb.log({"analysis_summary_table": wandb.Table(dataframe=metrics_df)})
        except Exception as e:
            logging.warning(f"Could not log metrics DataFrame to W&B: {e}")

    if args.store_data:
        store_sparsification_data(output_dir / "sparsification_data.npy", model_sparsification_data)
        store_uncertainty_data(output_dir / "uncertainty_data.npy", model_uncert_data)

    create_uncertainty_visualizations(metrics_df, output_dir, log_wandb=log_wandb)
    create_calibration_visualizations(all_predictions, all_ground_truths, output_dir, log_wandb=log_wandb)
    # perform_temperature_analysis(all_predictions, all_ground_truths, output_dir, args.temp_values, log_wandb=log_wandb)
    
    if args.plot_roc_pr:
        if len(global_errors)>0:
            all_err = np.concatenate(global_errors)
            all_unc = np.concatenate(global_uncertainties)
            plot_global_roc_pr(all_err, all_unc, output_dir, model_label=args.model_label, prefix="global_", log_wandb=log_wandb)
    
    if log_wandb:
        try:
            summary_stats = {
                "summary/avg_dice": metrics_df['dice'].mean(),
                "summary/std_dice": metrics_df['dice'].std(),
                "summary/avg_ece": metrics_df['ece'].mean(),
                "summary/std_ece": metrics_df['ece'].std(),
                "summary/avg_brier": metrics_df['brier'].mean(),
                "summary/std_brier": metrics_df['brier'].std(),
                "summary/avg_sparsification_error": metrics_df['sparsification_error'].mean(),
                "summary/std_sparsification_error": metrics_df['sparsification_error'].std(),
                "summary/avg_uncertain_pixel_percent": metrics_df['uncertain_pixel_percent'].mean(),
                "summary/std_uncertain_pixel_percent": metrics_df['uncertain_pixel_percent'].std(),
            }
            wandb.summary.update(summary_stats)
        except Exception as e:
            logging.warning(f"Could not log summary statistics to W&B: {e}")
    
    logging.info("\nSummary Statistics:")
    logging.info(f"Number of images analyzed: {len(metrics_df)}")
    logging.info(f"Average Dice Score: {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
    logging.info(f"Average ECE: {metrics_df['ece'].mean():.4f} ± {metrics_df['ece'].std():.4f}")
    logging.info(f"Average Brier Score: {metrics_df['brier'].mean():.4f} ± {metrics_df['brier'].std():.4f}")
    logging.info(f"Average Sparsification Error: {metrics_df['sparsification_error'].mean():.4f} ± {metrics_df['sparsification_error'].std():.4f}")
    logging.info(f"Average Uncertain Pixel %: {metrics_df['uncertain_pixel_percent'].mean():.2f}% ± {metrics_df['uncertain_pixel_percent'].std():.2f}%")
    
    return metrics_df, model_sparsification_data, model_uncert_data


def create_uncertainty_visualizations(metrics_df, output_dir, log_wandb=False):  
    """Create visualizations for uncertainty analysis."""
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    numeric_cols = [col for col in metrics_df.columns if col != 'img_id']
    for col in numeric_cols:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
        else:
            logging.warning(f"Column '{col}' not found in metrics_df for numeric conversion.")
            
    logging.info(f"DataFrame columns for uncertainty viz: {metrics_df.columns.tolist()}")
    logging.info(f"DataFrame dtypes for uncertainty viz: {metrics_df.dtypes}")
    
    try:
        if 'dice' in metrics_df.columns and 'ece' in metrics_df.columns:
            sns.scatterplot(x='dice', y='ece', data=metrics_df, ax=axes[0, 0], s=80, alpha=0.7)
            axes[0, 0].set_title('Segmentation Accuracy vs. Calibration Error', fontsize=14)
            axes[0, 0].set_xlabel('Dice Score (higher is better)', fontsize=12)
            axes[0, 0].set_ylabel('ECE (lower is better)', fontsize=12)
            
            corr = metrics_df['dice'].corr(metrics_df['ece'])
            axes[0, 0].annotate(f'Correlation: {corr:.3f}', 
                             xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            axes[0, 0].annotate(
                "This plot shows the relationship between segmentation\n"
                "accuracy (Dice) and calibration quality (ECE).\n"
                "Lower ECE means better calibrated probabilities.",
                xy=(0.5, 0.05), xycoords='axes fraction', ha='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
        else:
            axes[0, 0].text(0.5, 0.5, "Dice or ECE data missing", ha='center', va='center')
            axes[0, 0].set_title('Segmentation Accuracy vs. Calibration Error (Data Missing)')
    except Exception as e:
        logging.error(f"Error creating Dice vs ECE plot: {e}")
    
    try:
        if 'dice' in metrics_df.columns and 'sparsification_error' in metrics_df.columns:
            colors = ['green' if se > 0 else 'red' for se in metrics_df['sparsification_error']]
            scatter = sns.scatterplot(x='dice', y='sparsification_error', data=metrics_df, 
                                     ax=axes[0, 1], s=80, alpha=0.7, hue=colors, palette={'green':'green', 'red':'red'}, legend=False) 
            axes[0, 1].set_title('Segmentation Accuracy vs. Uncertainty Quality', fontsize=14)
            axes[0, 1].set_xlabel('Dice Score (higher is better)', fontsize=12)
            axes[0, 1].set_ylabel('Sparsification Error (higher is better)', fontsize=12)
            
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                      label='Good uncertainty (SE > 0)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
                      label='Poor uncertainty (SE <= 0)')
            ]
            axes[0, 1].legend(handles=legend_elements, loc='lower right') 
            
            corr = metrics_df['dice'].corr(metrics_df['sparsification_error'])
            axes[0, 1].annotate(f'Correlation: {corr:.3f}', 
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            axes[0, 1].annotate(
                "This plot shows how uncertainty quality (SE) relates\n"
                "to segmentation accuracy (Dice).\n"
                "Positive SE (green): Uncertainty is meaningful\n"
                "Non-positive SE (red): Uncertainty is poorly estimated",
                xy=(0.5, 0.05), xycoords='axes fraction', ha='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
        else:
            axes[0, 1].text(0.5, 0.5, "Dice or Sparsification Error data missing", ha='center', va='center')
            axes[0, 1].set_title('Segmentation Accuracy vs. Uncertainty Quality (Data Missing)')
    except Exception as e:
        logging.error(f"Error creating Dice vs Sparsification Error plot: {e}")
        logging.exception("Detailed traceback:")
    
    try:
        if 'ece' in metrics_df.columns:
            sns.histplot(x='ece', data=metrics_df, kde=True, ax=axes[1, 0], color='teal')
            axes[1, 0].axvline(metrics_df['ece'].mean(), color='r', linestyle='--', 
                             label=f'Mean: {metrics_df["ece"].mean():.3f}')
            
            axes[1, 0].axvspan(0, 0.01, alpha=0.2, color='green', label='Excellent (<0.01)')
            axes[1, 0].axvspan(0.01, 0.05, alpha=0.2, color='yellowgreen', label='Good (<0.05)')
            axes[1, 0].axvspan(0.05, 0.15, alpha=0.2, color='orange', label='Fair (<0.15)')
            axes[1, 0].axvspan(0.15, 1, alpha=0.2, color='red', label='Poor (>0.15)')
            
            axes[1, 0].set_title('Distribution of Expected Calibration Error', fontsize=14)
            axes[1, 0].set_xlabel('ECE (lower is better)', fontsize=12)
            axes[1, 0].legend(loc='upper right', fontsize=9)
            
            axes[1, 0].annotate(
                "ECE measures how well confidence values\n"
                "match actual frequencies.\n"
                "Lower values indicate better calibration.",
                xy=(0.7, 0.5), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
        else:
            axes[1, 0].text(0.5, 0.5, "ECE data missing", ha='center', va='center')
            axes[1, 0].set_title('Distribution of Expected Calibration Error (Data Missing)')
    except Exception as e:
        logging.error(f"Error creating ECE histogram: {e}")
    
    try:
        if 'uncertain_pixel_percent' in metrics_df.columns:
            sns.histplot(x='uncertain_pixel_percent', data=metrics_df, kde=True, ax=axes[1, 1], color='purple')
            axes[1, 1].axvline(metrics_df['uncertain_pixel_percent'].mean(), color='r', linestyle='--',
                              label=f'Mean: {metrics_df["uncertain_pixel_percent"].mean():.1f}%')
            axes[1, 1].set_title('Distribution of Uncertain Pixel Percentage', fontsize=14)
            axes[1, 1].set_xlabel('Uncertain Pixel %', fontsize=12)
            axes[1, 1].legend(loc='upper right')
            
            axes[1, 1].annotate(
                "Shows what percentage of pixels have high uncertainty.\n"
                "Ideally correlates with difficult regions and errors.\n"
                "Too high: model is underconfident\n"
                "Too low: model might be overconfident",
                xy=(0.3, 0.7), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
        else:
            axes[1, 1].text(0.5, 0.5, "Uncertain Pixel % data missing", ha='center', va='center')
            axes[1, 1].set_title('Distribution of Uncertain Pixel Percentage (Data Missing)')
    except Exception as e:
        logging.error(f"Error creating Uncertain Pixel histogram: {e}")
    
    plt.suptitle(f'Uncertainty Analysis Summary - {len(metrics_df)} Images', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    uncertainty_summary_path = output_dir / "uncertainty_summary.png"
    plt.savefig(uncertainty_summary_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if log_wandb:
        try:
            wandb.log({"plots/uncertainty_summary": wandb.Image(str(uncertainty_summary_path))})
        except Exception as e:
            logging.warning(f"Could not log uncertainty summary plot to W&B: {e}")
    
    try:
        existing_numeric_cols = [col for col in numeric_cols if col in metrics_df.columns]
        if len(existing_numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            numeric_df = metrics_df[existing_numeric_cols]
            corr_matrix = numeric_df.corr()
            
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, 
                       annot=True, fmt='.2f', center=0, square=True, linewidths=.5)
            
            plt.title('Correlation Matrix of Uncertainty Metrics', fontsize=15)
            
            plt.figtext(0.5, 0.01, 
                      "This heatmap shows how different metrics relate to each other.\n"
                      "Values close to 1 or -1 indicate strong correlation.\n"
                      "Look for strong correlations between uncertainty and performance metrics.",
                      ha="center", fontsize=12, 
                      bbox={"facecolor":"lightgoldenrodyellow", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            corr_matrix_path = output_dir / "correlation_matrix.png"
            plt.savefig(corr_matrix_path, dpi=300)
            plt.close()
            if log_wandb:
                try:
                    wandb.log({"plots/correlation_matrix": wandb.Image(str(corr_matrix_path))})
                except Exception as e:
                    logging.warning(f"Could not log correlation matrix plot to W&B: {e}")
        else:
            logging.warning("Not enough numeric columns to create correlation heatmap.")
    except Exception as e:
        logging.error(f"Error creating correlation heatmap: {e}")
    
    try:
        selected_cols = ['dice', 'ece', 'brier', 'sparsification_error', 'uncertain_pixel_percent']
        existing_cols = [col for col in selected_cols if col in metrics_df.columns]
        if len(existing_cols) >= 2:  
            g = sns.pairplot(
                metrics_df[existing_cols],
                diag_kind="kde",
                plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k', 'linewidth': 0.5},
                corner=True,  
            )
            g.fig.suptitle('Pairwise Relationships Between Uncertainty Metrics', y=1.02, fontsize=16)
            
            plt.figtext(0.5, 0.01, 
                      "This matrix shows how each pair of metrics relates to each other.\n"
                      "Look for patterns and correlations that can help interpret uncertainty.\n"
                      "Diagonal plots show the distribution of each individual metric.",
                      ha="center", fontsize=12, 
                      bbox={"facecolor":"lightgoldenrodyellow", "alpha":0.5, "pad":5})
            
            plt.subplots_adjust(top=0.95, bottom=0.1)
            pairplot_path = output_dir / "metrics_pairplot.png"
            plt.savefig(pairplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            if log_wandb:
                try:
                    wandb.log({"plots/metrics_pairplot": wandb.Image(str(pairplot_path))})
                except Exception as e:
                    logging.warning(f"Could not log pairplot to W&B: {e}")
        else:
            logging.warning(f"Not enough numeric columns for pairplot. Found: {existing_cols}")
    except Exception as e:
        logging.error(f"Error creating pairplot: {e}")
    
    try:
        required_calib_cols = ['max_calibration_error', 'mean_abs_calib_error', 'ece', 'dice']
        if all(col in metrics_df.columns for col in required_calib_cols):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(
                metrics_df['max_calibration_error'], 
                metrics_df['mean_abs_calib_error'],
                c=metrics_df['ece'], 
                s=metrics_df['dice'] * 200,  
                alpha=0.7,
                cmap='viridis',
                edgecolors='k',
                linewidths=0.5
            )
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('Expected Calibration Error (ECE)', fontsize=12)
            
            handles, labels = [], []
            for dice_val in [0.25, 0.5, 0.75, 1.0]: 
                handles.append(plt.scatter([], [], s=dice_val*200, color='gray', alpha=0.7, edgecolors='k'))
                labels.append(f'Dice = {dice_val:.2f}')
            ax.legend(handles, labels, title="Dice Score", loc="upper left")
            
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='MCE = MACE')
            
            ax.text(0.25, 0.9, "Consistent Calibration Errors", transform=ax.transAxes, 
                   ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
            ax.text(0.75, 0.1, "Outlier-dominated Errors", transform=ax.transAxes, 
                   ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_title('Calibration Error Analysis', fontsize=15)
            ax.set_xlabel('Maximum Calibration Error (MCE)\nHighest error in any confidence bin', fontsize=12)
            ax.set_ylabel('Mean Absolute Calibration Error (MACE)\nAverage error across all bins', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            explanation = (
                "This plot helps understand the nature of calibration errors:\n\n"
                "• Points near diagonal: Errors are consistent across confidence levels\n"
                "• Points below diagonal: Errors concentrated in specific confidence bins\n"
                "• Larger points: Better segmentation performance (higher Dice score)\n"
                "• Darker points: Better overall calibration (lower ECE)\n\n"
                "Ideal models would have small points in the bottom-left corner."
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            ax.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            calib_analysis_path = output_dir / "calibration_analysis.png"
            plt.savefig(calib_analysis_path, dpi=300)
            plt.close(fig)
            if log_wandb:
                try:
                    wandb.log({"plots/calibration_analysis_chart": wandb.Image(str(calib_analysis_path))})
                except Exception as e:
                    logging.warning(f"Could not log calibration analysis chart to W&B: {e}")
        else:
            logging.warning(f"Missing required columns for calibration analysis chart. Needed: {required_calib_cols}")
    except Exception as e:
        logging.error(f"Error creating calibration analysis chart: {e}")


def create_calibration_visualizations(all_predictions, all_ground_truths, output_dir, log_wandb=False):  
    """Create visualizations specifically for calibration analysis."""
    if not all_predictions or not all_ground_truths:
        logging.warning("No prediction data available for calibration analysis")
        return
        
    all_pred_flat = np.concatenate(all_predictions)
    all_gt_flat = np.concatenate(all_ground_truths)
    
    all_gt_flat = np.round(all_gt_flat).astype(int)
    
    global_prob_true, global_prob_pred = calibration_curve(
        all_gt_flat, all_pred_flat, n_bins=10, strategy='uniform'
    )
    
    global_ece = np.sum(
        np.abs(global_prob_true - global_prob_pred) * 
        np.histogram(all_pred_flat, bins=10)[0] / len(all_pred_flat)
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(global_prob_pred, global_prob_true, marker='o', linewidth=2,
           label=f'Calibration Curve (ECE={global_ece:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    ax2 = ax.twinx()
    ax2.hist(all_pred_flat, bins=20, alpha=0.3, density=True,
            color='gray', label='Prediction Distribution')
    ax2.set_ylabel('Density')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Global Calibration Curve (All Images)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    global_calib_path = output_dir / "global_calibration_curve.png"
    plt.savefig(global_calib_path, dpi=200)
    plt.close(fig)
    if log_wandb:
        try:
            wandb.log({"plots/global_calibration_curve": wandb.Image(str(global_calib_path))})
        except Exception as e:
            logging.warning(f"Could not log global calibration curve to W&B: {e}")


def perform_temperature_analysis(all_predictions, all_ground_truths, output_dir, temperatures, log_wandb=False):
    """Analyze the effect of temperature scaling on calibration."""
    if not all_predictions or not all_ground_truths:
        logging.warning("No prediction data available for temperature analysis")
        return

    all_pred_flat = np.concatenate(all_predictions)
    all_gt_flat = np.concatenate(all_ground_truths)
    all_gt_flat = np.round(all_gt_flat).astype(int)

    results = []
    plt.figure(figsize=(12, 8))
    
    ax_hist = plt.gca().twinx()
    ax_hist.hist(all_pred_flat, bins=30, alpha=0.2, density=True, color='gray', label='Original Pred Dist')
    ax_hist.set_ylabel('Density')
    ax_hist.legend(loc='center right')

    for temp in temperatures:
        eps = 1e-9
        pred_temp = np.clip(all_pred_flat, eps, 1 - eps)
        
        logits = np.log(pred_temp / (1 - pred_temp))
        scaled_logits = logits / temp
        calibrated_pred = 1 / (1 + np.exp(-scaled_logits))
        
        prob_true, prob_pred = calibration_curve(all_gt_flat, calibrated_pred, n_bins=10, strategy='uniform')
        ece = np.sum(np.abs(prob_true - prob_pred) * np.histogram(calibrated_pred, bins=10)[0] / len(calibrated_pred))
        
        results.append({'temperature': temp, 'ece': ece})
        
        plt.plot(prob_pred, prob_true, marker='o', linestyle='--', linewidth=1.5, alpha=0.8,
                 label=f'T={temp:.2f} (ECE={ece:.4f})')

    plt.plot([0, 1], [0, 1], 'k-', label='Perfect Calibration')
    
    plt.xlabel('Mean Predicted Probability (Calibrated)')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves with Temperature Scaling')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    temp_scaling_plot_path = output_dir / "temperature_scaling_calibration.png"
    plt.savefig(temp_scaling_plot_path, dpi=200)
    plt.close()
    if log_wandb:
        try:
            wandb.log({"plots/temperature_scaling_calibration": wandb.Image(str(temp_scaling_plot_path))})
        except Exception as e:
            logging.warning(f"Could not log temperature scaling plot to W&B: {e}")

    temp_df = pd.DataFrame(results)
    best_temp_idx = temp_df['ece'].idxmin()
    best_temp = temp_df.loc[best_temp_idx, 'temperature']
    best_ece = temp_df.loc[best_temp_idx, 'ece']

    plt.figure(figsize=(8, 6))
    plt.plot(temp_df['temperature'], temp_df['ece'], marker='o')
    plt.scatter([best_temp], [best_ece], color='red', s=100, zorder=5, label=f'Best T={best_temp:.2f} (ECE={best_ece:.4f})')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('ECE vs. Temperature Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ece_vs_temp_path = output_dir / "ece_vs_temperature.png"
    plt.savefig(ece_vs_temp_path, dpi=200)
    plt.close()
    if log_wandb:
        try:
            wandb.log({"plots/ece_vs_temperature": wandb.Image(str(ece_vs_temp_path))})
            wandb.summary['best_temperature'] = best_temp
            wandb.summary['best_temperature_ece'] = best_ece
        except Exception as e:
            logging.warning(f"Could not log ECE vs Temp plot/summary to W&B: {e}")

    logging.info(f"Temperature scaling analysis complete. Best temperature: {best_temp:.2f} with ECE: {best_ece:.4f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = get_args()
    
    wandb_run = None
    if not args.no_wandb:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                job_type="analysis"
            )
            logging.info(f"W&B run initialized: {wandb_run.url}")
        except Exception as e:
            logging.error(f"Failed to initialize W&B: {e}. Running without W&B logging.")
            wandb_run = None
    else:
        logging.info("W&B logging disabled by --no_wandb flag.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetResNet(
        n_channels=3,
        n_classes=1,
        latent_dim=32,
        use_attention=args.use_attention,
        latent_injection=args.latent_injection
    )
    
    logging.info(f'Loading model {args.model}')
    model_path = Path(f'./checkpoints/{args.model}')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logging.info(f'Loading test dataset with patch_size={args.patch_size}')
    test_dataset = IDRIDDataset(
        base_dir='./data',
        split='test',
        scale=args.scale,
        patch_size=args.patch_size,
        lesion_type=args.lesion_type,
        max_images=args.max_images,
        skip_border_check=(args.patch_size is None)
    )
    
    logging.info(f'Found {len(test_dataset)} test items')

    metrics_df, model_spars_data, model_uncert_data = analyze_model(model, test_dataset, args, wandb_run=wandb_run)

    if wandb_run:
        wandb.finish()
        logging.info("W&B run finished.")
