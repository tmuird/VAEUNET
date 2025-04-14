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

from utils.data_loading import IDRIDDataset, load_image
from unet.unet_resnet import UNetResNet
from utils.uncertainty_metrics import (
    calculate_expected_calibration_error, 
    calculate_sparsification_metrics,
    plot_reliability_diagram,
    plot_sparsification_curve,
    calculate_uncertainty_error_auc,
    calculate_uncertainty_error_dice
)
from utils.tensor_utils import ensure_dict_python_scalars, fix_dataframe_tensors
from visualize_vae import get_segmentation_distribution_from_image, track_memory

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
    log_wandb=False
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
    
    parser.add_argument('--latent-injection', type=str, default='all', 
                        choices=['all', 'first', 'last', 'bottleneck', 'inject_no_bottleneck', 'none'],
                        help='Latent space injection strategy')

    parser.add_argument('--temp_values', type=float, nargs='+', 
                       default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
                       help='Temperature values to analyze for temperature scaling')
    
    parser.add_argument('--store_data', action='store_true',
                        help='If set, store per-image sparsification and uncertainty arrays to disk for later comparisons')
    parser.add_argument('--model_label', type=str, default='Model',
                        help='Label to use when plotting ROC/PR (e.g. "High-KL")')
    
    parser.add_argument('--wandb_project', type=str, default='VAEUNET-Analysis', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team/user name)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (defaults to auto-generated)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging')

    parser.set_defaults(use_attention=True, enable_dropout=False)
    args = parser.parse_args()
    
    if args.patch_size == 0:
        args.patch_size = None
        
    return args


@track_memory
def analyze_model(model, dataset, args, wandb_run=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    output_dir = Path(args.output_dir) / f"{args.lesion_type}_T{args.temperature}_N{args.samples}"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_pixel_data_dir = output_dir / "temp_pixel_data"
    temp_pixel_data_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_data = []
    processed_ids = set()
    log_wandb = wandb_run is not None
    
    for i in tqdm(range(len(dataset)), desc="Analyzing images"):
        sample = dataset[i]
        img_id = sample['img_id']
        
        if img_id in processed_ids:
            continue
        
        logging.info(f"Processing image {img_id}")
        
        try:
            segmentations, mask, mu, logvar = get_segmentation_distribution_from_image(
                model, img_id, dataset=dataset, num_samples=args.samples,
                patch_size=args.patch_size, overlap=args.overlap, 
                temperature=args.temperature, enable_dropout=False,
                batch_size=args.batch_size
            )
            
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
                'error_auprc': auprc
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
                    import traceback
                    traceback.print_exc()

            del segmentations, mask, mu, logvar, segmentations_cpu, mask_cpu, mean_pred, std_dev
            del pred_flat, gt_flat, errors, uncertainties_flat, pred_binary, pred_binary_np, gt_np, unc_map
            del uncertainties_correct, uncertainties_incorrect
            del frac_removed, err_random, err_uncertainty
            torch.cuda.empty_cache() 
            gc.collect()

        except Exception as e:
            logging.error(f"Error processing image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            for suffix in ["_pred_flat.npy", "_gt_flat.npy", "_errors.npy", "_uncertainties.npy", "_sparsification.npz", "_uncertainty_dist.npz"]:
                temp_file = temp_pixel_data_dir / f"{img_id}{suffix}"
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
            gc.collect()
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
    
    metrics_csv_path = output_dir / "analysis_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved metrics data to {metrics_csv_path}")
    
    if log_wandb:
        try:
            wandb.log({"analysis_summary_table": wandb.Table(dataframe=metrics_df)})
        except Exception as e:
            logging.warning(f"Could not log metrics DataFrame to W&B: {e}")

    logging.info(f"Loading temporary pixel data for {len(processed_ids)} images...")
    all_predictions_loaded = []
    all_ground_truths_loaded = []
    all_errors_loaded = []
    all_uncertainties_loaded = []

    for img_id in tqdm(processed_ids, desc="Loading temp data"):
        pred_flat_path = temp_pixel_data_dir / f"{img_id}_pred_flat.npy"
        gt_flat_path = temp_pixel_data_dir / f"{img_id}_gt_flat.npy"
        errors_path = temp_pixel_data_dir / f"{img_id}_errors.npy"
        uncertainties_path = temp_pixel_data_dir / f"{img_id}_uncertainties.npy"

        if pred_flat_path.exists() and gt_flat_path.exists():
            all_predictions_loaded.append(np.load(pred_flat_path))
            all_ground_truths_loaded.append(np.load(gt_flat_path))
        if errors_path.exists() and uncertainties_path.exists():
            all_errors_loaded.append(np.load(errors_path))
            all_uncertainties_loaded.append(np.load(uncertainties_path))

    logging.info("Finished loading temporary data.")
    gc.collect()

    create_uncertainty_visualizations(metrics_df, output_dir, log_wandb=log_wandb)
    
    if all_predictions_loaded and all_ground_truths_loaded:
        create_calibration_visualizations(all_predictions_loaded, all_ground_truths_loaded, output_dir, log_wandb=log_wandb)
        perform_temperature_analysis(all_predictions_loaded, all_ground_truths_loaded, output_dir, args.temp_values, log_wandb=log_wandb)
    else:
        logging.warning("Skipping calibration and temperature analysis due to missing loaded data.")

    plot_global_sparsification_curve(processed_ids, temp_pixel_data_dir, output_dir, model_label=args.model_label, log_wandb=log_wandb)
    plot_global_uncertainty_distribution(processed_ids, temp_pixel_data_dir, output_dir, model_label=args.model_label, log_wandb=log_wandb)
    
    if all_errors_loaded and all_uncertainties_loaded:
        logging.info("Concatenating global errors and uncertainties for ROC/PR plots...")
        try:
            if len(all_errors_loaded) > 0:
                 logging.debug(f"Shapes of first few error arrays: {[e.shape for e in all_errors_loaded[:3]]}")
                 logging.debug(f"Shapes of first few uncertainty arrays: {[u.shape for u in all_uncertainties_loaded[:3]]}")
            
            all_err_concat = np.concatenate(all_errors_loaded)
            all_unc_concat = np.concatenate(all_uncertainties_loaded)
            logging.info(f"Concatenated shapes: errors={all_err_concat.shape}, uncertainties={all_unc_concat.shape}")
            plot_global_roc_pr(all_err_concat, all_unc_concat, output_dir, model_label=args.model_label, prefix="global_", log_wandb=log_wandb)
            del all_err_concat, all_unc_concat
        except ValueError as e:
             logging.error(f"Error concatenating global arrays for ROC/PR: {e}. Check shapes of temporary .npy files in {temp_pixel_data_dir}. Skipping plot.")
        except Exception as e:
             logging.error(f"Unexpected error during global ROC/PR generation: {e}. Skipping plot.")
    else:
        logging.warning("Skipping global ROC/PR plot due to missing loaded data.")
    
    del all_predictions_loaded, all_ground_truths_loaded, all_errors_loaded, all_uncertainties_loaded
    gc.collect()

    if log_wandb:
        try:
            summary_stats = {
                "summary/avg_dice": metrics_df['dice'].mean(),
                "summary/std_dice": metrics_df['dice'].std(),
                "summary/avg_ece": metrics_df['ece'].mean(),
                "summary/std_ece": metrics_df['ece'].std(),
                "summary/avg_sparsification_error": metrics_df['sparsification_error'].mean(),
                "summary/std_sparsification_error": metrics_df['sparsification_error'].std(),
                "summary/avg_uncertainty_error_dice": metrics_df['uncertainty_error_dice'].mean(),
                "summary/std_uncertainty_error_dice": metrics_df['uncertainty_error_dice'].std(),
                "summary/avg_error_auroc": metrics_df['error_auroc'].mean(),
                "summary/std_error_auroc": metrics_df['error_auroc'].std(),
                "summary/avg_error_auprc": metrics_df['error_auprc'].mean(),
                "summary/std_error_auprc": metrics_df['error_auprc'].std(),
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


def create_calibration_visualizations(all_predictions, all_ground_truths, output_dir, log_wandb=False):  
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
        eps = 1e-7
        pred_clipped = np.clip(all_pred_flat, np.nextafter(0., 1.), np.nextafter(1., 0.))
        
        if not np.all(np.isfinite(pred_clipped)):
             logging.warning(f"Non-finite values found in predictions before logit calculation for T={temp}. Skipping this temperature.")
             results.append({'temperature': temp, 'ece': np.inf})
             continue

        logits = np.log(pred_clipped / (1 - pred_clipped))
        
        if not np.all(np.isfinite(logits)):
             logging.warning(f"Non-finite values found in logits for T={temp}. Skipping this temperature.")
             results.append({'temperature': temp, 'ece': np.inf})
             continue
             
        scaled_logits = logits / temp
        calibrated_pred = 1 / (1 + np.exp(-scaled_logits))
        
        if not np.all(np.isfinite(calibrated_pred)):
             logging.warning(f"Non-finite values found in calibrated predictions for T={temp}. Skipping this temperature.")
             results.append({'temperature': temp, 'ece': np.inf})
             continue

        try:
            prob_true, prob_pred = calibration_curve(all_gt_flat, calibrated_pred, n_bins=10, strategy='uniform')
            hist_counts, _ = np.histogram(calibrated_pred, bins=10, range=(0,1))
            total_count = len(calibrated_pred)
            if total_count == 0:
                 ece = 0.0
            else:
                 bin_weights = hist_counts / total_count
                 ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights)

            results.append({'temperature': temp, 'ece': ece})
            
            plt.plot(prob_pred, prob_true, marker='o', linestyle='--', linewidth=1.5, alpha=0.8,
                     label=f'T={temp:.2f} (ECE={ece:.4f})')
        except Exception as e:
             logging.error(f"Error calculating calibration curve or ECE for T={temp}: {e}")
             results.append({'temperature': temp, 'ece': np.inf})


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
