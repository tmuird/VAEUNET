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
import wandb
from PIL import Image
import matplotlib.cm as cm
import math
import gc
import shutil
import psutil
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import traceback
import functools

from utils.data_loading import IDRIDDataset, load_image
from unet.unet_resnet import UNetResNet, AttentionGate
from utils.uncertainty_metrics import (
    calculate_expected_calibration_error, 
    calculate_sparsification_metrics,
    plot_reliability_diagram,
    plot_sparsification_curve,
    calculate_uncertainty_error_auc,
    calculate_uncertainty_error_dice,
    calculate_segmentation_metrics

)
from utils.tensor_utils import ensure_dict_python_scalars, fix_dataframe_tensors
from visualize_vae import get_segmentation_distribution_from_image, track_memory, get_image_and_mask, predict_with_patches, predict_full_image


def log_memory_usage(stage_name=""):
    """Logs current RAM and GPU memory usage."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    gpu_mem_mb = 0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    logging.info(f"Memory usage {stage_name}: RAM={ram_mb:.2f} MB, GPU={gpu_mem_mb:.2f} MB")


# --- Global dictionary to store attention maps from hooks ---
captured_attention_maps = defaultdict(list)
current_sample_index = 0  # Global variable to track sample index (use with caution)


# Modify hook function to accept module_name
def attention_hook_fn(module, input, output, module_name):
    """Hook function to capture attention map output."""
    global captured_attention_maps
    global current_sample_index
    # Store the output tensor on CPU to save GPU memory
    # Key by module name and sample index
    # Use module_name passed via functools.partial
    captured_attention_maps[f"sample_{current_sample_index}_{module_name}"].append(output.detach().cpu())


def plot_global_roc_pr(
    processed_ids,
    temp_pixel_data_dir,
    output_dir,
    model_label="Model",
    prefix="",
    log_wandb=False,
):
    """
    Plots the global AUROC and AUPRC curves by loading data incrementally.
    Saves plots into output_dir with optional prefix appended to filenames.

    Args:
        processed_ids: Set of image IDs that were processed.
        temp_pixel_data_dir: Path to the directory containing temporary .npy files.
        output_dir: Path to output directory
        model_label: e.g. 'High-KL Model'
        prefix: optional string to prepend to filenames, e.g. "global_" => "global_roc.png"
        log_wandb: If True, log plots to W&B
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_memory_usage("[Before Global ROC/PR Load]")

    all_errors_loaded = []
    all_uncertainties_loaded = []
    loaded_count = 0

    logging.info("Loading errors and uncertainties for global ROC/PR...")
    for img_id in tqdm(processed_ids, desc="Loading ROC/PR data"):
        errors_path = temp_pixel_data_dir / f"{img_id}_errors.npy"
        uncertainties_path = temp_pixel_data_dir / f"{img_id}_uncertainties.npy"
        if errors_path.exists() and uncertainties_path.exists():
            try:
                all_errors_loaded.append(np.load(errors_path))
                all_uncertainties_loaded.append(np.load(uncertainties_path))
                loaded_count += 1
            except Exception as e:
                logging.warning(f"Could not load ROC/PR data for {img_id}: {e}")
        else:
            logging.warning(f"ROC/PR files not found for {img_id}: {errors_path.exists()=}, {uncertainties_path.exists()=}")

    if loaded_count == 0:
        logging.warning("No valid error/uncertainty data loaded to plot global ROC/PR.")
        return
    logging.info(f"Loaded ROC/PR data for {loaded_count} images.")
    log_memory_usage("[After Global ROC/PR Load]")

    try:
        logging.info("Concatenating global errors and uncertainties...")
        all_errors = np.concatenate(all_errors_loaded)
        all_uncertainties = np.concatenate(all_uncertainties_loaded)
        total_pixels = len(all_errors)
        logging.info(f"Concatenated shapes: errors={all_errors.shape}, uncertainties={all_uncertainties.shape}")
        log_memory_usage("[After Global ROC/PR Concat]")
        
        # Calculate global baseline using all loaded errors
        global_baseline = np.sum(all_errors) / (total_pixels + 1e-9)
        logging.info(f"Calculated global baseline (positive rate): {global_baseline:.6f}")

        # Free original lists
        del all_errors_loaded, all_uncertainties_loaded
        gc.collect()
        log_memory_usage("[After Global ROC/PR List Cleanup]")

        # Compute ROC using all data
        fpr, tpr, _ = roc_curve(all_errors, all_uncertainties)
        roc_auc = auc(fpr, tpr)

        # Compute PR using all data
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
        plt.plot([0,1],[global_baseline, global_baseline],'k--', label=f'Chance={global_baseline:.3f}')
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

        # Final cleanup
        del fpr, tpr, precision, recall
        if 'all_errors' in locals(): del all_errors
        if 'all_uncertainties' in locals(): del all_uncertainties
        gc.collect()
        log_memory_usage("[After Global ROC/PR Plotting]")

    except ValueError as e:
         logging.error(f"Error concatenating global arrays for ROC/PR: {e}. Check shapes of temporary .npy files in {temp_pixel_data_dir}. Skipping plot.")
    except Exception as e:
         logging.error(f"Unexpected error during global ROC/PR generation: {e}. Skipping plot.")
    finally:
        if 'all_errors_loaded' in locals(): del all_errors_loaded
        if 'all_uncertainties_loaded' in locals(): del all_uncertainties_loaded
        if 'all_errors' in locals(): del all_errors
        if 'all_uncertainties' in locals(): del all_uncertainties
        gc.collect()
        log_memory_usage("[End of Global ROC/PR]")

def create_calibration_visualizations(processed_ids, temp_pixel_data_dir, output_dir, log_wandb=False):
    """Creates global calibration plot by loading data incrementally."""
    log_memory_usage("[Before Calibration Load]")
    all_predictions_loaded = []
    all_ground_truths_loaded = []
    loaded_count = 0

    logging.info("Loading predictions and ground truths for calibration...")
    for img_id in tqdm(processed_ids, desc="Loading calibration data"):
        pred_flat_path = temp_pixel_data_dir / f"{img_id}_pred_flat.npy"
        gt_flat_path = temp_pixel_data_dir / f"{img_id}_gt_flat.npy"
        if pred_flat_path.exists() and gt_flat_path.exists():
            try:
                all_predictions_loaded.append(np.load(pred_flat_path))
                all_ground_truths_loaded.append(np.load(gt_flat_path))
                loaded_count += 1
            except Exception as e:
                logging.warning(f"Could not load calibration data for {img_id}: {e}")
        else:
            logging.warning(f"Calibration files not found for {img_id}: {pred_flat_path.exists()=}, {gt_flat_path.exists()=}")

    if loaded_count == 0:
        logging.warning("No valid prediction/GT data loaded for calibration analysis.")
        return
    logging.info(f"Loaded calibration data for {loaded_count} images.")
    log_memory_usage("[After Calibration Load]")

    try:
        all_pred_flat = np.concatenate(all_predictions_loaded)
        all_gt_flat = np.concatenate(all_ground_truths_loaded)
        log_memory_usage("[After Calibration Concat]")
        
        del all_predictions_loaded, all_ground_truths_loaded
        gc.collect()
        log_memory_usage("[After Calibration List Cleanup]")
        
        all_gt_flat = np.round(all_gt_flat).astype(int)
        
        global_prob_true, global_prob_pred = calibration_curve(
            all_gt_flat, all_pred_flat, n_bins=10, strategy='uniform'
        )
        
        hist_counts, bin_edges = np.histogram(all_pred_flat, bins=10, range=(0,1))
        total_count = len(all_pred_flat)
        bin_weights = hist_counts / total_count
        global_ece = np.sum(np.abs(global_prob_true - global_prob_pred) * bin_weights)
        
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
        logging.info(f"Global calibration curve saved to {global_calib_path}")
        if log_wandb:
            try:
                wandb.log({"plots/global_calibration_curve": wandb.Image(str(global_calib_path))})
                wandb.summary["global_ece"] = global_ece
            except Exception as e:
                logging.warning(f"Could not log global calibration curve to W&B: {e}")

    except ValueError as e:
        logging.error(f"ValueError during calibration calculation: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during calibration visualization: {e}")
    finally:
        if 'all_pred_flat' in locals(): del all_pred_flat
        if 'all_gt_flat' in locals(): del all_gt_flat
        gc.collect()
        log_memory_usage("[End of Calibration]")


def perform_temperature_analysis(processed_ids, temp_pixel_data_dir, output_dir, temperatures, log_wandb=False):
    """Performs temperature scaling analysis by loading data incrementally."""
    log_memory_usage("[Before Temp Analysis Load]")
    all_predictions_loaded = []
    all_ground_truths_loaded = []
    loaded_count = 0

    logging.info("Loading predictions and ground truths for temperature analysis...")
    for img_id in tqdm(processed_ids, desc="Loading temp analysis data"):
        pred_flat_path = temp_pixel_data_dir / f"{img_id}_pred_flat.npy"
        gt_flat_path = temp_pixel_data_dir / f"{img_id}_gt_flat.npy"
        if pred_flat_path.exists() and gt_flat_path.exists():
            try:
                all_predictions_loaded.append(np.load(pred_flat_path))
                all_ground_truths_loaded.append(np.load(gt_flat_path))
                loaded_count += 1
            except Exception as e:
                logging.warning(f"Could not load temp analysis data for {img_id}: {e}")
        else:
            logging.warning(f"Temp analysis files not found for {img_id}: {pred_flat_path.exists()=}, {gt_flat_path.exists()=}")

    if loaded_count == 0:
        logging.warning("No valid prediction/GT data loaded for temperature analysis.")
        return
    logging.info(f"Loaded temp analysis data for {loaded_count} images.")
    log_memory_usage("[After Temp Analysis Load]")

    try:
        all_pred_flat = np.concatenate(all_predictions_loaded)
        all_gt_flat = np.concatenate(all_ground_truths_loaded)
        log_memory_usage("[After Temp Analysis Concat]")
        
        del all_predictions_loaded, all_ground_truths_loaded
        gc.collect()
        log_memory_usage("[After Temp Analysis List Cleanup]")

        all_gt_flat = np.round(all_gt_flat).astype(int)
        
        results = []
        eps = 1e-7
        pred_clipped = np.clip(all_pred_flat, eps, 1 - eps)
        logits = np.log(pred_clipped / (1 - pred_clipped))
        
        plt.figure(figsize=(12, 8))
        ax_hist = plt.gca().twinx()
        ax_hist.hist(all_pred_flat, bins=30, alpha=0.2, density=True, color='gray', label='Original Pred Dist')
        ax_hist.set_ylabel('Density')
        ax_hist.legend(loc='center right')

        for temp in temperatures:
            scaled_logits = logits / temp
            calibrated_pred = 1 / (1 + np.exp(-scaled_logits))
            
            prob_true, prob_pred = calibration_curve(all_gt_flat, calibrated_pred, n_bins=10, strategy='uniform')
            hist_counts, _ = np.histogram(calibrated_pred, bins=10, range=(0,1))
            total_count = len(calibrated_pred)
            bin_weights = hist_counts / total_count
            ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights)

            results.append({'temperature': temp, 'ece': ece})
            
            plt.plot(prob_pred, prob_true, marker='o', linestyle='--', linewidth=1.5, alpha=0.8,
                     label=f'T={temp:.2f} (ECE={ece:.4f})')

        temp_df = pd.DataFrame(results)
        valid_ece = temp_df[np.isfinite(temp_df['ece'])]
        if not valid_ece.empty:
            best_temp_idx = valid_ece['ece'].idxmin()
            best_temp = valid_ece.loc[best_temp_idx, 'temperature']
            best_ece = valid_ece.loc[best_temp_idx, 'ece']
            logging.info(f"Temperature scaling analysis complete. Best temperature: {best_temp:.2f} with ECE: {best_ece:.4f}")

            plt.figure(figsize=(8, 6))
            plt.plot(valid_ece['temperature'], valid_ece['ece'], marker='o')
            plt.scatter([best_temp], [best_ece], color='red', s=100, zorder=5, label=f'Best T={best_temp:.2f} (ECE={best_ece:.4f})')
            plt.xlabel('Temperature (T)')
            plt.ylabel('Expected Calibration Error (ECE)')
            plt.title('ECE vs. Temperature Scaling (Finite Values)')
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
        else:
            logging.warning("No valid (finite) ECE values found during temperature scaling.")
            if log_wandb:
                 wandb.summary['best_temperature'] = None
                 wandb.summary['best_temperature_ece'] = None

    except ValueError as e:
        logging.error(f"ValueError during temperature analysis: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during temperature analysis: {e}")
    finally:
        if 'all_pred_flat' in locals(): del all_pred_flat
        if 'all_gt_flat' in locals(): del all_gt_flat
        if 'logits' in locals(): del logits
        gc.collect()
        log_memory_usage("[End of Temp Analysis]")


def plot_global_sparsification_curve(processed_ids, temp_data_dir, output_dir, model_label="Model", log_wandb=False):
    """Plots the average sparsification curve across all images by loading data from temp files."""
    if not processed_ids:
        logging.warning("No processed image IDs available to plot global sparsification curve.")
        return

    all_frac_removed = []
    all_err_random = []
    all_err_uncertainty = []
    
    logging.info("Loading sparsification data for global curve...")
    loaded_count = 0
    for img_id in tqdm(processed_ids, desc="Loading sparsification data"):
        spars_path = temp_data_dir / f"{img_id}_sparsification.npz"
        if spars_path.exists():
            try:
                data = np.load(spars_path)
                if 'frac_removed' in data and 'err_random' in data and 'err_uncertainty' in data:
                    all_frac_removed.append(data['frac_removed'])
                    all_err_random.append(data['err_random'])
                    all_err_uncertainty.append(data['err_uncertainty'])
                    loaded_count += 1
                else:
                    logging.warning(f"Missing keys in sparsification file: {spars_path}")
                data.close()
            except Exception as e:
                logging.warning(f"Could not load sparsification data for {img_id}: {e}")
        else:
            logging.warning(f"Sparsification file not found: {spars_path}")

    if loaded_count == 0:
        logging.warning("No valid sparsification data loaded to plot global curve.")
        return
    logging.info(f"Loaded sparsification data for {loaded_count} images.")

    if not all(len(fr) == len(all_frac_removed[0]) for fr in all_frac_removed):
         logging.warning("Inconsistent lengths found in frac_removed arrays. Cannot average.")
         return 
    frac_removed = all_frac_removed[0]

    try:
        stacked_err_random = np.stack(all_err_random)
        stacked_err_uncertainty = np.stack(all_err_uncertainty)
    except ValueError as e:
        logging.error(f"Error stacking sparsification arrays (likely inconsistent shapes): {e}")
        logging.error(f"Shapes err_random: {[arr.shape for arr in all_err_random]}")
        logging.error(f"Shapes err_uncertainty: {[arr.shape for arr in all_err_uncertainty]}")
        return

    avg_err_random = np.mean(stacked_err_random, axis=0)
    avg_err_uncertainty = np.mean(stacked_err_uncertainty, axis=0)
    
    if avg_err_random[0] > 0:
        norm_avg_random = avg_err_random / avg_err_random[0]
        norm_avg_uncertainty = avg_err_uncertainty / avg_err_random[0]
    else:
        norm_avg_random = avg_err_random
        norm_avg_uncertainty = avg_err_uncertainty
        
    avg_se = np.trapz(norm_avg_random - norm_avg_uncertainty, frac_removed)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax, _ = plot_sparsification_curve(frac_removed, avg_err_random, avg_err_uncertainty, ax=ax)
    ax.set_title(f'Global Sparsification Curve ({model_label})\nAverage SE = {avg_se:.4f}')
    
    plt.tight_layout()
    global_spars_path = output_dir / "global_sparsification_curve.png"
    plt.savefig(global_spars_path, dpi=300)
    plt.close(fig)
    logging.info(f"Global sparsification curve saved to {global_spars_path}")
    
    if log_wandb:
        try:
            wandb.log({"plots/global_sparsification_curve": wandb.Image(str(global_spars_path))})
            wandb.summary["global_sparsification_error"] = avg_se
        except Exception as e:
            logging.warning(f"Could not log global sparsification curve to W&B: {e}")
    
    del all_frac_removed, all_err_random, all_err_uncertainty
    del stacked_err_random, stacked_err_uncertainty
    gc.collect()


def plot_global_uncertainty_distribution(processed_ids, temp_data_dir, output_dir, model_label="Model", log_wandb=False):
    """Plots boxplots comparing uncertainty for correct vs incorrect pixels globally by loading data from temp files."""
    if not processed_ids:
        logging.warning("No processed image IDs available to plot global uncertainty distribution.")
        return

    all_unc_correct_list = []
    all_unc_incorrect_list = []

    logging.info("Loading uncertainty distribution data for global plot...")
    loaded_count = 0
    for img_id in tqdm(processed_ids, desc="Loading uncertainty data"):
        uncert_path = temp_data_dir / f"{img_id}_uncertainty_dist.npz"
        if uncert_path.exists():
            try:
                data = np.load(uncert_path)
                if 'uncertainties_correct' in data and 'uncertainties_incorrect' in data:
                    all_unc_correct_list.append(data['uncertainties_correct'])
                    all_unc_incorrect_list.append(data['uncertainties_incorrect'])
                    loaded_count += 1
                else:
                    logging.warning(f"Missing keys in uncertainty distribution file: {uncert_path}")
                data.close()
            except Exception as e:
                logging.warning(f"Could not load uncertainty distribution data for {img_id}: {e}")
        else:
             logging.warning(f"Uncertainty distribution file not found: {uncert_path}")


    if loaded_count == 0:
        logging.warning("No valid uncertainty distribution data loaded to plot global distribution.")
        return
    logging.info(f"Loaded uncertainty distribution data for {loaded_count} images.")

    try:
        all_unc_correct = np.concatenate([arr for arr in all_unc_correct_list if arr.size > 0])
        all_unc_incorrect = np.concatenate([arr for arr in all_unc_incorrect_list if arr.size > 0])
    except ValueError as e:
        logging.error(f"Error concatenating uncertainty arrays: {e}")
        return
    finally:
        del all_unc_correct_list, all_unc_incorrect_list
        gc.collect()


    all_unc_correct = all_unc_correct[np.isfinite(all_unc_correct)]
    all_unc_incorrect = all_unc_incorrect[np.isfinite(all_unc_incorrect)]

    if all_unc_correct.size == 0 and all_unc_incorrect.size == 0:
        logging.warning("No valid uncertainty values found for global distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_to_plot = []
    labels = []
    if all_unc_correct.size > 0:
        data_to_plot.append(all_unc_correct)
        labels.append(f'Correct ({len(all_unc_correct)})')
    if all_unc_incorrect.size > 0:
        data_to_plot.append(all_unc_incorrect)
        labels.append(f'Incorrect ({len(all_unc_incorrect)})')

    if not data_to_plot:
        logging.warning("No data to plot for global uncertainty distribution.")
        plt.close(fig)
        return

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Uncertainty (Std Dev)')
    ax.set_title(f'Global Uncertainty Distribution ({model_label})')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    medians = [np.median(d) for d in data_to_plot]
    for i, median in enumerate(medians):
        ax.text(i + 1, median, f'{median:.3f}', 
                horizontalalignment='center', verticalalignment='bottom', 
                fontweight='bold', color='black')

    plt.tight_layout()
    global_unc_dist_path = output_dir / "global_uncertainty_distribution.png"
    plt.savefig(global_unc_dist_path, dpi=300)
    plt.close(fig)
    logging.info(f"Global uncertainty distribution plot saved to {global_unc_dist_path}")

    if log_wandb:
        try:
            wandb.log({"plots/global_uncertainty_distribution": wandb.Image(str(global_unc_dist_path))})
            if len(medians) == 2:
                 wandb.summary["median_unc_correct"] = medians[0]
                 wandb.summary["median_unc_incorrect"] = medians[1]
            elif len(medians) == 1 and 'Correct' in labels[0]:
                 wandb.summary["median_unc_correct"] = medians[0]
            elif len(medians) == 1 and 'Incorrect' in labels[0]:
                 wandb.summary["median_unc_incorrect"] = medians[0]
        except Exception as e:
            logging.warning(f"Could not log global uncertainty distribution to W&B: {e}")
            
    del all_unc_correct, all_unc_incorrect, data_to_plot
    gc.collect()


def create_uncertainty_visualizations(metrics_df, output_dir, log_wandb=False):  
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
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
            sns.scatterplot(x='dice', y='ece', data=metrics_df, ax=axes[0], s=80, alpha=0.7)
            axes[0].set_title('Segmentation Accuracy vs. Calibration Error', fontsize=14)
            axes[0].set_xlabel('Dice Score (higher is better)', fontsize=12)
            axes[0].set_ylabel('ECE (lower is better)', fontsize=12)
            
            corr = metrics_df['dice'].corr(metrics_df['ece'])
            axes[0].annotate(f'Correlation: {corr:.3f}', 
                             xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        else:
            axes[0].text(0.5, 0.5, "Dice or ECE data missing", ha='center', va='center')
            axes[0].set_title('Segmentation Accuracy vs. Calibration Error (Data Missing)')
    except Exception as e:
        logging.error(f"Error creating Dice vs ECE plot: {e}")
        axes[0].text(0.5, 0.5, "Error plotting Dice vs ECE", ha='center', va='center')
    
    try:
        if 'dice' in metrics_df.columns and 'sparsification_error' in metrics_df.columns:
            colors = ['green' if se > 0 else 'red' for se in metrics_df['sparsification_error']]
            scatter = sns.scatterplot(x='dice', y='sparsification_error', data=metrics_df, 
                                     ax=axes[1], s=80, alpha=0.7, hue=colors, palette={'green':'green', 'red':'red'}, legend=False) 
            axes[1].set_title('Segmentation Accuracy vs. Uncertainty Quality', fontsize=14)
            axes[1].set_xlabel('Dice Score (higher is better)', fontsize=12)
            axes[1].set_ylabel('Sparsification Error (higher is better)', fontsize=12)
            
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                      label='Good uncertainty (SE > 0)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
                      label='Poor uncertainty (SE <= 0)')
            ]
            axes[1].legend(handles=legend_elements, loc='lower right') 
            
            corr = metrics_df['dice'].corr(metrics_df['sparsification_error'])
            axes[1].annotate(f'Correlation: {corr:.3f}', 
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        else:
            axes[1].text(0.5, 0.5, "Dice or Sparsification Error data missing", ha='center', va='center')
            axes[1].set_title('Segmentation Accuracy vs. Uncertainty Quality (Data Missing)')
    except Exception as e:
        logging.error(f"Error creating Dice vs Sparsification Error plot: {e}")
        axes[1].text(0.5, 0.5, "Error plotting Dice vs SE", ha='center', va='center')
    
    try:
        if 'ece' in metrics_df.columns:
            sns.histplot(x='ece', data=metrics_df, kde=True, ax=axes[2], color='teal')
            axes[2].axvline(metrics_df['ece'].mean(), color='r', linestyle='--', 
                             label=f'Mean: {metrics_df["ece"].mean():.3f}')
            
            axes[2].axvspan(0, 0.01, alpha=0.2, color='green', label='Excellent (<0.01)')
            axes[2].axvspan(0.01, 0.05, alpha=0.2, color='yellowgreen', label='Good (<0.05)')
            axes[2].axvspan(0.05, 0.15, alpha=0.2, color='orange', label='Fair (<0.15)')
            axes[2].axvspan(0.15, 1, alpha=0.2, color='red', label='Poor (>0.15)')
            
            axes[2].set_title('Distribution of Expected Calibration Error', fontsize=14)
            axes[2].set_xlabel('ECE (lower is better)', fontsize=12)
            axes[2].legend(loc='upper right', fontsize=9)
        else:
            axes[2].text(0.5, 0.5, "ECE data missing", ha='center', va='center')
            axes[2].set_title('Distribution of Expected Calibration Error (Data Missing)')
    except Exception as e:
        logging.error(f"Error creating ECE histogram: {e}")
        axes[2].text(0.5, 0.5, "Error plotting ECE", ha='center', va='center')

    try:
        if 'uncertainty_error_dice' in metrics_df.columns:
            sns.histplot(x='uncertainty_error_dice', data=metrics_df, kde=True, ax=axes[3], color='indigo')
            axes[3].axvline(metrics_df['uncertainty_error_dice'].mean(), color='r', linestyle='--',
                              label=f'Mean: {metrics_df["uncertainty_error_dice"].mean():.3f}')
            axes[3].set_title('Distribution of Uncertainty-Error Dice', fontsize=14)
            axes[3].set_xlabel('U-E Dice (higher indicates better overlap)', fontsize=12)
            axes[3].set_xlim(0, 1)
            axes[3].legend(loc='upper right')
        else:
            axes[3].text(0.5, 0.5, "U-E Dice data missing", ha='center', va='center')
            axes[3].set_title('Distribution of Uncertainty-Error Dice (Data Missing)')
    except Exception as e:
        logging.error(f"Error creating U-E Dice histogram: {e}")
        axes[3].text(0.5, 0.5, "Error plotting U-E Dice", ha='center', va='center')

    plt.suptitle(f'Uncertainty Analysis Summary - {len(metrics_df)} Images', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    uncertainty_summary_path = output_dir / "uncertainty_summary.png"
    plt.savefig(uncertainty_summary_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if log_wandb:
        try:
            wandb.log({"plots/uncertainty_summary": wandb.Image(str(uncertainty_summary_path))})
        except Exception as e:
            logging.warning(f"Could not log uncertainty summary plot to W&B: {e}")


@track_memory
def analyze_model(model, dataset, args, wandb_run=None):
    # Declare global variables at the beginning of the function
    global captured_attention_maps
    global current_sample_index
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()    
    output_dir = Path(args.output_dir) / f"{args.lesion_type}_T{args.temperature}_N{args.samples}"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_pixel_data_dir = output_dir / "temp_pixel_data"
    temp_pixel_data_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_data = []
    processed_ids = set()
    log_wandb = wandb_run is not None
    
    # --- Check if model uses attention ---
    model_uses_attention = isinstance(model, UNetResNet) and model.use_attention
    if model_uses_attention:
        logging.info("Model uses attention, will attempt to capture attention maps.")
        if args.patch_size is not None and args.patch_size > 0:
            logging.warning("Attention map visualization is currently best supported for full-image analysis (patch_size=None). Stitching patch attention maps is not implemented.")

    for i in tqdm(range(len(dataset)), desc="Analyzing images"):
        sample = dataset[i]
        img_id = sample['img_id']
        
        if img_id in processed_ids:
            continue
        
        logging.info(f"Processing image {img_id}")
        
        # --- Register hooks before processing the image ---
        hook_handles = []
        captured_attention_maps.clear()  # Clear maps for the new image
        
        if model_uses_attention:
            logging.debug(f"Registering attention hooks for image {img_id}")
            attention_gate_count = 0
            try:
                # Use functools.partial to pass module name to the hook
                for name, module in model.named_modules():
                    if isinstance(module, AttentionGate):
                        # Create a partial function with the module name baked in
                        hook_with_name = functools.partial(attention_hook_fn, module_name=name)
                        handle = module.psi.register_forward_hook(hook_with_name)
                        hook_handles.append(handle)
                        attention_gate_count += 1
                logging.debug(f"Registered {attention_gate_count} attention hooks.")
            except Exception as e:
                logging.error(f"Failed to register attention hooks: {e}")
                for handle in hook_handles:
                    handle.remove()
                hook_handles = []

        try:
            # --- Call the main function to get segmentations ---
            all_segmentations = []
            all_masks = []
            
            current_sample_index = 0
            
            img_full, mask_full, original_shape_from_file = get_image_and_mask(dataset, img_id) # Rename original_shape
            img_full_dev = img_full.unsqueeze(0).to(device)
            mask_full_dev = mask_full.unsqueeze(0).to(device)
            
            # Get the actual shape of the tensor fed into the model
            input_tensor_shape = img_full_dev.shape[2:] # H, W
            
            # Determine if sampling should be applied based on model configuration
            should_sample = getattr(model, 'latent_injection', 'all') not in ['none', 'inject_no_bottleneck']
            if not should_sample:
                logging.info(f"Latent injection mode is '{model.latent_injection}'. Using deterministic mu (temperature ignored).")
            
            # Get latent distribution parameters
            with torch.no_grad():
                mu, logvar = model.encode(img_full_dev)

            segmentations_list_cpu = []
            for s_idx in range(args.samples):
                current_sample_index = s_idx
                logging.debug(f"Generating sample {s_idx} for image {img_id}")
                with torch.no_grad():
                    # Sample from latent space or use mu directly
                    if should_sample:
                        std = torch.exp(0.5 * logvar) * args.temperature
                        eps = torch.randn_like(std)
                        z = mu + eps * std
                    else:
                        # Use deterministic mu when sampling is disabled for this mode
                        z = mu
                    z = z.unsqueeze(-1).unsqueeze(-1)

                    if args.patch_size is not None and args.patch_size > 0:
                        seg = predict_with_patches(model, img_full_dev, z, patch_size=args.patch_size, overlap=args.overlap, batch_size=args.batch_size)
                    else:
                        seg = predict_full_image(model, img_full_dev, z)
                    
                    segmentations_list_cpu.append(seg.cpu())
                    del seg, z
                    if should_sample:
                        del eps, std
                    torch.cuda.empty_cache()

            segmentations = torch.cat(segmentations_list_cpu, dim=0)
            mask = mask_full_dev
            del segmentations_list_cpu, img_full_dev
            
            # --- Process captured attention maps ---
            if hook_handles and captured_attention_maps:
                logging.debug(f"Processing {len(captured_attention_maps)} captured attention map entries.")
                processed_attention_maps = defaultdict(list)
                
                for key, maps_list in captured_attention_maps.items():
                    parts = key.split('_')
                    sample_idx = int(parts[1])
                    # The rest of the key is the module name
                    module_name = '_'.join(parts[2:]) # Reconstruct module name
                    if maps_list:
                        # Store maps associated with the module name
                        processed_attention_maps[module_name].append(maps_list[0]) 
                
                averaged_attention_maps = {}
                for module_name, maps_list in processed_attention_maps.items():
                    if maps_list:
                        stacked_maps = torch.stack(maps_list, dim=0)
                        avg_map = stacked_maps.mean(dim=0)
                        averaged_attention_maps[module_name] = avg_map # Use module_name as key
                        logging.debug(f"Averaged attention map for module {module_name}: shape {avg_map.shape}")

                if log_wandb and averaged_attention_maps:
                    log_dict = {}
                    # Use the shape of the input tensor for resizing
                    img_h, img_w = input_tensor_shape 
                    map_idx = 0
                    # Sort maps by name to potentially get them in order (e.g., decoder_0, decoder_1...)
                    sorted_module_names = sorted(averaged_attention_maps.keys())

                    for module_name in sorted_module_names:
                        avg_map = averaged_attention_maps[module_name]
                        try:
                            # Fix: Ensure the attention map has the right number of dimensions and channels
                            # First, squeeze any singleton dimensions except batch
                            if avg_map.dim() > 2:
                                # Keep only first channel if multi-channel
                                if avg_map.shape[0] > 1:
                                    avg_map = avg_map[0:1]
                            
                            # Make sure it's in the format [1, C, H, W] for interpolation
                            if avg_map.dim() == 2:  # [H, W]
                                avg_map = avg_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                            elif avg_map.dim() == 3:  # [C, H, W] or [1, H, W]
                                avg_map = avg_map.unsqueeze(0)  # Add batch dim
                            
                            # Now use proper interpolation with input tensor in expected format
                            # Resize to the actual input tensor dimensions
                            logging.debug(f"Resizing attention map from {avg_map.shape} to {(img_h, img_w)}")
                            resized_map = F.interpolate(avg_map, size=(img_h, img_w), 
                                                      mode='bilinear', align_corners=False)
                            
                            # Squeeze back to [H, W] for visualization
                            resized_map = resized_map.squeeze()
                            
                            # Normalize for visualization
                            if resized_map.dim() >= 2:
                                # Get min/max for proper normalization
                                min_val = resized_map.min()
                                max_val = resized_map.max()
                                if max_val > min_val:
                                    resized_map = (resized_map - min_val) / (max_val - min_val)
                                
                                # Convert to numpy for visualization
                                resized_map_np = resized_map.numpy()
                                colormap = cm.get_cmap('viridis')
                                colored_map = (colormap(resized_map_np)[:, :, :3] * 255).astype(np.uint8)
                                # Use module_name in the log key for clarity
                                log_dict[f"attention_maps/{img_id}/{module_name}"] = wandb.Image(colored_map)
                                map_idx += 1 # Keep map_idx if needed, but module_name is more descriptive
                        except Exception as viz_e:
                            logging.warning(f"Could not visualize attention map for module {module_name}: {viz_e}")
                            import traceback
                            traceback.print_exc()
                    
                    if log_dict:
                        try:
                            wandb.log(log_dict)
                            logging.info(f"Logged {len(log_dict)} averaged attention maps for {img_id}.")
                        except Exception as log_e:
                            logging.warning(f"Could not log attention maps to W&B for {img_id}: {log_e}")
                
                del processed_attention_maps, averaged_attention_maps
                if 'log_dict' in locals(): del log_dict

            segmentations_cpu = segmentations.cpu()
            mask_cpu = mask.cpu()
            
            mean_pred = segmentations_cpu.mean(dim=0)
            std_dev = segmentations_cpu.std(dim=0)

   
            pred_flat = mean_pred[0].flatten().numpy()
            gt_flat = mask_cpu[0,0].flatten().numpy()
            gt_flat = np.round(gt_flat).astype(int)

            pred_binary = (mean_pred[0] > 0.5).float()
            ue_dice = calculate_uncertainty_error_dice(std_dev[0], pred_binary, mask_cpu[0, 0])

            ece, bin_accs, bin_confs, bin_counts = calculate_expected_calibration_error(
                mean_pred[0], mask_cpu[0, 0]
            )
            dice_tensor = (2.0*(pred_binary*mask_cpu[0,0]).sum())/(pred_binary.sum()+mask_cpu[0,0].sum()+1e-8)
            dice = float(dice_tensor.item())

            np.save(temp_pixel_data_dir / f"{img_id}_pred_flat.npy", pred_flat)
            np.save(temp_pixel_data_dir / f"{img_id}_gt_flat.npy", gt_flat)

            se = 0.0
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

            spars_save_path = temp_pixel_data_dir / f"{img_id}_sparsification.npz"
            np.savez(spars_save_path, 
                     frac_removed=frac_removed, 
                     err_random=err_random, 
                     err_uncertainty=err_uncertainty)

            pred_binary_np = pred_binary.numpy()
            gt_np = mask_cpu[0,0].numpy()
            correct_mask = (pred_binary_np == gt_np)
            incorrect_mask = ~correct_mask
            unc_map = std_dev[0].numpy()
            uncertainties_correct = unc_map[correct_mask]
            uncertainties_incorrect = unc_map[incorrect_mask]

            uncert_dist_save_path = temp_pixel_data_dir / f"{img_id}_uncertainty_dist.npz"
            np.savez(uncert_dist_save_path,
                     uncertainties_correct=uncertainties_correct,
                     uncertainties_incorrect=uncertainties_incorrect)

            errors = (pred_binary_np != gt_np).astype(np.int32).flatten()
            uncertainties_flat = unc_map.flatten()
            np.save(temp_pixel_data_dir / f"{img_id}_errors.npy", errors)
            np.save(temp_pixel_data_dir / f"{img_id}_uncertainties.npy", uncertainties_flat)

            auroc, auprc = calculate_uncertainty_error_auc(mean_pred[0], mask_cpu[0,0], std_dev[0])

            metrics_dict = {
                'img_id': str(img_id),
                'dice': dice,
                'ece': ece,
                'sparsification_error': se,
                'uncertainty_error_dice': ue_dice,  
                'error_auroc': auroc,
                'error_auprc': auprc,
            }
            metrics_dict = ensure_dict_python_scalars(metrics_dict)
            metrics_data.append(metrics_dict)

            processed_ids.add(img_id)

            if log_wandb:
                try:
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
                    pred_vis_prob_norm = (pred_vis_prob - pred_vis_prob.min()) / (pred_vis_prob.max() - pred_vis_prob.min() + 1e-8)
                    pred_vis = (pred_vis_prob_norm * 255).clip(0, 255).astype(np.uint8)
                    
                    uncert_map_raw = std_dev[0].numpy()
                    uncert_map_norm = (uncert_map_raw - uncert_map_raw.min()) / (uncert_map_raw.max() - uncert_map_raw.min() + 1e-8)
                    colormap_hot = cm.get_cmap('hot')
                    uncert_vis_colored = (colormap_hot(uncert_map_norm)[:, :, :3] * 255).astype(np.uint8)

                    wandb.log({
                        f"visualizations/{img_id}/original_image": wandb.Image(img_vis),
                        f"visualizations/{img_id}/ground_truth": wandb.Image(gt_vis),
                        f"visualizations/{img_id}/mean_prediction": wandb.Image(pred_vis),
                        f"visualizations/{img_id}/uncertainty_map_std_dev": wandb.Image(
                            uncert_vis_colored, 
                            caption=f"Std Dev - Mean: {std_dev[0].mean().item():.4f}"
                        ),
                        "image_index": i
                    })
                    
                    del img_vis, gt_vis, pred_vis, uncert_map_raw, uncert_map_norm, uncert_vis_colored
                    del img_pil, img_pil_scaled
                    if 'img_array_vis' in locals(): del img_array_vis

                except Exception as e:
                    logging.warning(f"Could not log visualizations to W&B for {img_id}: {e}")
                    traceback.print_exc()

            del segmentations, mask, mu, logvar, segmentations_cpu, mask_cpu, mean_pred, std_dev
            del pred_flat, gt_flat, errors, uncertainties_flat, pred_binary, pred_binary_np, gt_np, unc_map
            del uncertainties_correct, uncertainties_incorrect
            del frac_removed, err_random, err_uncertainty
      
            torch.cuda.empty_cache() 
            gc.collect()

        except Exception as e:
            logging.error(f"Error processing image {img_id}: {e}")
            traceback.print_exc()
            for suffix in ["_pred_flat.npy", "_gt_flat.npy", "_errors.npy", "_uncertainties.npy", "_sparsification.npz", "_uncertainty_dist.npz"]:
                temp_file = temp_pixel_data_dir / f"{img_id}{suffix}"
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
            gc.collect()
            continue
        finally:
            if hook_handles:
                logging.debug(f"Removing {len(hook_handles)} attention hooks for image {img_id}")
                for handle in hook_handles:
                    handle.remove()
                hook_handles = []
            captured_attention_maps.clear()
            gc.collect()

        if args.max_images and len(processed_ids) >= args.max_images:
            break

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = fix_dataframe_tensors(metrics_df)
    numeric_cols = []
    for col in metrics_df.columns:
        if col != 'img_id':
            try:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
                numeric_cols.append(col)
            except Exception as e:
                logging.warning(f"Could not convert {col} to numeric: {e}")
    
    metrics_csv_path = output_dir / "analysis_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved metrics data to {metrics_csv_path}")
    
    if log_wandb:
        try:
            summary_row = metrics_df[numeric_cols].mean().to_dict()
            summary_row['img_id'] = 'Overall Mean'
            summary_df = pd.DataFrame([summary_row])
            df_for_wandb = pd.concat([metrics_df, summary_df], ignore_index=True)
            wandb.log({"analysis_summary_table": wandb.Table(dataframe=df_for_wandb)})
        except Exception as e:
            logging.warning(f"Could not log metrics DataFrame with summary row to W&B: {e}")
     # create_uncertainty_visualizations(metrics_df, output_dir, log_wandb=log_wandb)
     # --- Compute global segmentation metrics (Garifullin-style) ---
   # Replace the section where global segmentation metrics are calculated with this:

    # --- Compute global segmentation metrics with memory-efficient chunking ---
    logging.info("Computing global segmentation metrics with memory-efficient chunking...")

    try:
        # Direct calculation of ROC/PR curves for visualization
        all_preds = []
        all_gts = []
        
        # Only collect a subset from each image for plotting ROC/PR 
        for img_id in processed_ids:
            try:
                pred_path = temp_pixel_data_dir / f"{img_id}_pred_flat.npy"
                gt_path = temp_pixel_data_dir / f"{img_id}_gt_flat.npy"
                if pred_path.exists() and gt_path.exists():
                    # Load and sample to limit memory usage
                    pred = np.load(pred_path)
                    gt = np.load(gt_path)
                    
                    # If large image, take a random sample
                    if len(pred) > 50000:
                        sample_size = 50000
                        indices = np.random.choice(len(pred), sample_size, replace=False)
                        all_preds.append(pred[indices])
                        all_gts.append(gt[indices])
                    else:
                        all_preds.append(pred)
                        all_gts.append(gt)
                    
                    # Clear variables to free memory
                    del pred, gt
                    gc.collect()
            except Exception as e:
                logging.warning(f"Error sampling prediction/GT for {img_id}: {e}")

        # Calculate ROC/PR curves for visualization
        if all_preds and all_gts:
            log_memory_usage("[Before Plot ROC/PR Concat]")
            all_preds_flat = np.concatenate(all_preds)
            all_gts_flat = np.concatenate(all_gts).astype(int)
            log_memory_usage("[After Plot ROC/PR Concat]")
            
            # Calculate and plot ROC curve
            fpr, tpr, _ = roc_curve(all_gts_flat, all_preds_flat)
            seg_roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f'{args.model_label} (AUC={seg_roc_auc:.4f})', lw=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Chance')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Global Segmentation ROC Curve ({args.model_label})')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            seg_roc_path = output_dir / "global_segmentation_roc_curve.png"
            plt.savefig(seg_roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate and plot PR curve
            precision, recall, _ = precision_recall_curve(all_gts_flat, all_preds_flat)
            seg_prc_auc = auc(recall, precision)
            baseline = np.sum(all_gts_flat) / len(all_gts_flat)
            
            plt.figure(figsize=(6, 6))
            plt.plot(recall, precision, label=f'{args.model_label} (AUC={seg_prc_auc:.4f})', lw=2)
            plt.plot([0, 1], [baseline, baseline], 'k--', label=f'Chance={baseline:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Global Segmentation PR Curve ({args.model_label})')
            plt.legend(loc='lower left')
            plt.grid(alpha=0.3)
            seg_pr_path = output_dir / "global_segmentation_pr_curve.png"
            plt.savefig(seg_pr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Clean up temporary variables
            del all_preds_flat, all_gts_flat, fpr, tpr, precision, recall
            gc.collect()
            log_memory_usage("[After Plot ROC/PR Cleanup]")
            
            logging.info(f"Global Segmentation ROC curve saved to {seg_roc_path} (AUC={seg_roc_auc:.4f})")
            logging.info(f"Global Segmentation PR curve saved to {seg_pr_path} (AUC={seg_prc_auc:.4f})")
            
            if log_wandb:
                try:
                    wandb.log({
                        "plots/global_segmentation_roc_curve": wandb.Image(str(seg_roc_path)),
                        "plots/global_segmentation_pr_curve": wandb.Image(str(seg_pr_path))
                    })
                    # Log AUCs
                    wandb.summary["segmentation/auroc"] = seg_roc_auc
                    wandb.summary["segmentation/auprc"] = seg_prc_auc
                except Exception as e:
                    logging.warning(f"Could not log global segmentation ROC/PR plots to W&B: {e}")
        
        # Calculate other metrics in a memory-efficient way
        from utils.uncertainty_metrics import calculate_segmentation_metrics_chunked
        
        seg_metrics = calculate_segmentation_metrics_chunked(
            processed_ids, 
            temp_pixel_data_dir,
            threshold=0.5,
            chunk_size=100000
        )
        
        if log_wandb:
            wandb.summary.update({
                "segmentation/precision": seg_metrics['precision'],
                "segmentation/recall": seg_metrics['recall'],
                "segmentation/specificity": seg_metrics['specificity'],
                "segmentation/accuracy": seg_metrics['accuracy'],
                "segmentation/f1_score": seg_metrics['f1_score']
            })

        logging.info(f"[Segmentation Metrics - Global]")
        for k, v in seg_metrics.items():
            logging.info(f"{k}: {v:.4f}")

    except Exception as e:
        logging.error(f"Error in global segmentation metrics calculation: {e}")
        traceback.print_exc()
    # --- Ensure these plotting calls are present ---
    create_calibration_visualizations(processed_ids, temp_pixel_data_dir, output_dir, log_wandb=log_wandb)
    perform_temperature_analysis(processed_ids, temp_pixel_data_dir, output_dir, args.temp_values, log_wandb=log_wandb)
    plot_global_sparsification_curve(processed_ids, temp_pixel_data_dir, output_dir, model_label=args.model_label, log_wandb=log_wandb)
    plot_global_uncertainty_distribution(processed_ids, temp_pixel_data_dir, output_dir, model_label=args.model_label, log_wandb=log_wandb)
    plot_global_roc_pr(processed_ids, temp_pixel_data_dir, output_dir, model_label=args.model_label, prefix="global_", log_wandb=log_wandb)

    # --- End of plotting calls ---

    if log_wandb:
        try:
            summary_stats = {
                "summary/avg_dice": metrics_df['dice'].mean(),
                "summary/avg_ece": metrics_df['ece'].mean(),
                "summary/avg_sparsification_error": metrics_df['sparsification_error'].mean(),
                "summary/avg_uncertainty_error_dice": metrics_df['uncertainty_error_dice'].mean(),
                "summary/avg_error_auroc": metrics_df['error_auroc'].mean(),
                "summary/avg_error_auprc": metrics_df['error_auprc'].mean(),
              
      
            }
            wandb.summary.update(summary_stats)
        except Exception as e:
            logging.warning(f"Could not log summary statistics to W&B: {e}")
    
    logging.info("\nSummary Statistics:")
    logging.info(f"Number of images analyzed: {len(metrics_df)}")
    logging.info(f"Average Dice Score: {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
    logging.info(f"Average ECE: {metrics_df['ece'].mean():.4f} ± {metrics_df['ece'].std():.4f}")
    logging.info(f"Average Sparsification Error: {metrics_df['sparsification_error'].mean():.4f} ± {metrics_df['sparsification_error'].std():.4f}")
    logging.info(f"Average Uncertainty-Error Dice: {metrics_df['uncertainty_error_dice'].mean():.4f} ± {metrics_df['uncertainty_error_dice'].std():.4f}")
    logging.info(f"Average Error AUROC: {metrics_df['error_auroc'].mean():.4f} ± {metrics_df['error_auroc'].std():.4f}")
    logging.info(f"Average Error AUPRC: {metrics_df['error_auprc'].mean():.4f} ± {metrics_df['error_auprc'].std():.4f}")
    
    try:
        logging.info(f"Removing temporary directory: {temp_pixel_data_dir}")
        shutil.rmtree(temp_pixel_data_dir)
    except Exception as e:
        logging.warning(f"Could not remove temporary directory {temp_pixel_data_dir}: {e}")

    return metrics_df


def get_args():
    parser = argparse.ArgumentParser(description='Analyze VAE-UNet model performance and uncertainty')
    parser.add_argument('--model', '-m', default='best_model.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--lesion_type', type=str, required=True, choices=['EX', 'HE', 'MA', 'SE', 'OD'],
                        help='Lesion type to analyze')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling from latent space (default: 1.0)')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples for uncertainty estimation (default: 10)')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size for prediction. If None, uses full image.')
    parser.add_argument('--overlap', type=int, default=100,
                        help='Overlap between patches if using patch_size (default: 100)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for resizing images (default: 1.0)')
    parser.add_argument('--attention', dest='use_attention', action='store_true',
                        help='Enable attention mechanism (default)')
    parser.add_argument('--no-attention', dest='use_attention', action='store_false',
                        help='Disable attention mechanism')
    parser.add_argument('--latent-injection', type=str, default='all', 
                        choices=['all', 'first', 'last', 'bottleneck', 'inject_no_bottleneck', 'none'],
                        help='Latent space injection strategy')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results and plots')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of test images to process (default: all)')
    parser.add_argument('--temp_values', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                        help='List of temperatures for temperature scaling analysis')
    parser.add_argument('--model_label', type=str, default='VAE-UNet',
                        help='Label for the model in plots')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for patch processing (default: 4)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='VAE_UNet_Analysis', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (defaults to auto-generated)')

    parser.set_defaults(use_attention=True)
    return parser.parse_args()


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

    metrics_df = analyze_model(model, test_dataset, args, wandb_run=wandb_run)

    if wandb_run:
        wandb.finish()
        logging.info("W&B run finished.")
