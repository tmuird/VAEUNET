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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from textwrap import wrap

from utils.data_loading import IDRIDDataset
from unet.unet_resnet import UNetResNet
from utils.uncertainty_metrics import (
    calculate_expected_calibration_error, 
    brier_score,
    calculate_sparsification_metrics,
    plot_reliability_diagram,
    plot_sparsification_curve
)
from utils.tensor_utils import ensure_dict_python_scalars, fix_dataframe_tensors
from visualize_vae import get_segmentation_distribution, track_memory

def get_args():
    parser = argparse.ArgumentParser(description='Analyze model performance, calibration and uncertainty')
    parser.add_argument('--model', '-m', default='best_model.pth', metavar='FILE', help='Model file')
    parser.add_argument('--lesion_type', type=str, required=True, default='EX', 
                      choices=['EX', 'HE', 'MA','OD'], help='Lesion type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--samples', type=int, default=30, help='Number of samples for ensemble prediction')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size (0 for full image)')
    parser.add_argument('--overlap', type=int, default=100, help='Overlap between patches')
    parser.add_argument('--output_dir', type=str, default='./analysis', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for resizing (default: 1.0)')
    
    # Analysis mode flags
    parser.add_argument('--calibration', action='store_true', help='Run calibration analysis')
    parser.add_argument('--uncertainty', action='store_true', help='Run uncertainty analysis')
    parser.add_argument('--temperature_sweep', action='store_true', 
                       help='Analyze effect of temperature on calibration')
    parser.add_argument('--temp_values', type=float, nargs='+', 
                       default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
                       help='Temperature values to analyze')
    
    args = parser.parse_args()
    
    # If no analysis flag is set, enable all by default
    if not (args.calibration or args.uncertainty or args.temperature_sweep):
        args.calibration = True
        args.uncertainty = True
        args.temperature_sweep = True
    
    # Convert patch_size=0 to None for full image mode
    if args.patch_size == 0:
        args.patch_size = None
        
    return args

@track_memory
def analyze_model(model, dataset, args):
    """Unified analysis function that handles both uncertainty and calibration analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.lesion_type}_T{args.temperature}_N{args.samples}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data storage
    metrics_data = []
    all_predictions = []
    all_ground_truths = []
    
    # Process each image only once (avoid duplicates from patches)
    processed_ids = set()
    
    for i in tqdm(range(len(dataset)), desc="Analyzing images"):
        sample = dataset[i]
        img_id = sample['img_id']
        
        # Skip if already processed
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
            
            # Move to CPU to save GPU memory
            segmentations_cpu = segmentations.cpu()
            mask_cpu = mask.cpu()
            
            # Calculate mean prediction
            mean_pred = segmentations_cpu.mean(dim=0)
            std_dev = segmentations_cpu.std(dim=0)
            
            # Calculate calibration metrics
            ece, bin_accs, bin_confs, bin_counts = calculate_expected_calibration_error(
                mean_pred[0], mask_cpu[0, 0]
            )
            
            # Calculate Brier score
            brier = brier_score(mean_pred[0], mask_cpu[0, 0])
            
            # Calculate Dice score - ensure proper conversion to Python scalar
            pred_binary = (mean_pred[0] > 0.5).float()
            dice_tensor = (2.0 * (pred_binary * mask_cpu[0, 0]).sum()) / (
                pred_binary.sum() + mask_cpu[0, 0].sum() + 1e-8
            )
            dice = float(dice_tensor.item())  # Explicitly convert to Python float
            
            # Store flattened predictions and ground truth for global calibration analysis
            if args.calibration or args.temperature_sweep:
                pred_flat = mean_pred[0].flatten().numpy()
                gt_flat = mask_cpu[0, 0].flatten().numpy()
                
                # Fix for calibration curve - ensure ground truth is exactly 0 or 1
                gt_flat = np.round(gt_flat).astype(int)
                
                all_predictions.append(pred_flat)
                all_ground_truths.append(gt_flat)
            
            # Calculate sparsification metrics (only if uncertainty analysis is enabled)
            if args.uncertainty:
                frac_removed, err_random, err_uncertainty = calculate_sparsification_metrics(
                    mean_pred, std_dev, mask_cpu[:, 0], num_points=20
                )
                
                # Calculate sparsification error (area between curves)
                if err_random[0] > 0:
                    norm_random = err_random / err_random[0]
                    norm_uncertainty = err_uncertainty / err_random[0]
                else:
                    norm_random = err_random
                    norm_uncertainty = err_uncertainty
                    
                se = float(np.trapz(norm_random - norm_uncertainty, frac_removed))  # Ensure Python float
                
                # Calculate percent of uncertain pixels (std > threshold)
                uncertainty_threshold = 0.2
                uncertain_percent_tensor = (std_dev[0] > uncertainty_threshold).float().mean() * 100
                uncertain_pixel_percent = float(uncertain_percent_tensor.item())  # Ensure Python float
            else:
                # Default values if not running uncertainty analysis
                se = 0.0
                uncertain_pixel_percent = 0.0
                frac_removed = None
                err_random = None
                err_uncertainty = None
            
            # Calculate additional metrics
            max_calibration_error = float(np.max(np.abs(bin_accs - bin_confs)))
            mean_abs_calib_error = float(np.mean(np.abs(bin_accs - bin_confs)))
            
            # Create a metrics dictionary with explicit Python primitives
            metrics_dict = {
                'img_id': str(img_id),  # Ensure string
                'dice': dice,  # Already converted to float
                'ece': ece,  # Already float from calculate_expected_calibration_error
                'brier': brier,  # Already float from brier_score
                'sparsification_error': se,  # Explicitly converted to float
                'uncertain_pixel_percent': uncertain_pixel_percent,  # Explicitly converted to float
                'max_calibration_error': max_calibration_error,  # Explicitly converted to float
                'mean_abs_calib_error': mean_abs_calib_error  # Explicitly converted to float
            }
            
            # Double-check to ensure all values are primitive types
            metrics_dict = ensure_dict_python_scalars(metrics_dict)
            metrics_data.append(metrics_dict)
            
            # Generate individual image reports (conditional on analysis mode)
            img_output_dir = output_dir / "individual_reports"
            img_output_dir.mkdir(exist_ok=True)
            
            # Generate individual calibration report
            if args.calibration:
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
                plt.savefig(img_output_dir / f"{img_id}_calibration_curve.png", dpi=200)
                plt.close(fig)
            
            # Generate combined uncertainty metrics visualization
            if args.uncertainty:
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
                plt.savefig(img_output_dir / f"{img_id}_uncertainty_metrics.png", dpi=200)
                plt.close(fig)
                
                # Generate a more detailed analysis figure
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Show image, mask, and prediction
                if hasattr(dataset, 'get_display_image'):
                    # If dataset has a method to get display-ready images
                    display_img, display_mask = dataset.get_display_image(img_id)
                    axes[0, 0].imshow(display_img)
                    axes[0, 1].imshow(display_mask, cmap='gray')
                else:
                    # Just show placeholders
                    axes[0, 0].text(0.5, 0.5, "Original Image", 
                                 ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
                    axes[0, 1].text(0.5, 0.5, "Ground Truth Mask", 
                                 ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
                    
                # Show uncertainty map
                im = axes[1, 0].imshow(std_dev[0].numpy(), cmap='hot')
                plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
                axes[1, 0].set_title(f'Uncertainty Map (Mean Std: {std_dev[0].mean().item():.4f})')
                axes[1, 0].axis('off')
                
                # Show areas of high uncertainty
                # Create a mask showing where prediction is uncertain but wrong
                uncertainty_threshold = 0.2  # Define threshold for high uncertainty
                pred_binary_np = pred_binary.numpy()
                mask_np = mask_cpu[0, 0].numpy()
                uncertain_mask = (std_dev[0].numpy() > uncertainty_threshold).astype(np.float32)
                error_mask = (pred_binary_np != mask_np).astype(np.float32)
                uncertain_errors = uncertain_mask * error_mask
                
                # Display this composite mask - red where both high uncertainty and errors
                rgb_overlay = np.zeros((*uncertain_mask.shape, 3))
                rgb_overlay[..., 0] = uncertain_mask  # Red channel = high uncertainty
                rgb_overlay[..., 1] = error_mask      # Green channel = errors
                
                axes[1, 1].imshow(rgb_overlay)
                axes[1, 1].set_title('Red: High Uncertainty, Green: Errors, Yellow: Both')
                axes[1, 1].axis('off')
                
                plt.suptitle(f'Detailed Analysis for Image {img_id}', fontsize=16)
                plt.tight_layout()
                plt.savefig(img_output_dir / f"{img_id}_detailed_analysis.png", dpi=200)
                plt.close(fig)
            
            # Free up memory
            del segmentations_cpu, mask_cpu, mean_pred, std_dev
            
        except Exception as e:
            logging.error(f"Error processing image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Break if we've processed enough images
        if args.max_images and len(processed_ids) >= args.max_images:
            break
    
    # Create metrics dataframe and apply tensor fixing utility
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = fix_dataframe_tensors(metrics_df)  # Extra safety check before visualization
    
    # Additional check: ensure all numeric columns are properly converted
    for col in metrics_df.columns:
        if col != 'img_id':  # Skip the image ID column
            try:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
            except Exception as e:
                logging.warning(f"Could not convert column {col} to numeric: {e}")
    
    # Save metrics to CSV
    metrics_csv_path = output_dir / "analysis_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved metrics data to {metrics_csv_path}")
    
    # Generate appropriate analysis reports based on enabled modes
    if args.uncertainty:
        create_uncertainty_visualizations(metrics_df, output_dir)
    
    if args.calibration:
        create_calibration_visualizations(all_predictions, all_ground_truths, output_dir)
    
    if args.temperature_sweep:
        perform_temperature_analysis(all_predictions, all_ground_truths, output_dir, args.temp_values)
    
    # Print summary statistics
    logging.info("\nSummary Statistics:")
    logging.info(f"Number of images analyzed: {len(metrics_df)}")
    logging.info(f"Average Dice Score: {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
    logging.info(f"Average ECE: {metrics_df['ece'].mean():.4f} ± {metrics_df['ece'].std():.4f}")
    logging.info(f"Average Brier Score: {metrics_df['brier'].mean():.4f} ± {metrics_df['brier'].std():.4f}")
    
    if args.uncertainty:
        logging.info(f"Average Sparsification Error: {metrics_df['sparsification_error'].mean():.4f} ± {metrics_df['sparsification_error'].std():.4f}")
        logging.info(f"Average Uncertain Pixel %: {metrics_df['uncertain_pixel_percent'].mean():.2f}% ± {metrics_df['uncertain_pixel_percent'].std():.2f}%")
    
    print_interpretation_guide(args)
    
    return metrics_df

def create_uncertainty_visualizations(metrics_df, output_dir):
    """Create visualizations for uncertainty analysis."""
    # Set Seaborn style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11})
    
    # Create a 2x2 grid of plots with more informative descriptions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Ensure we have proper numeric columns by converting everything except img_id
    numeric_cols = [col for col in metrics_df.columns if col != 'img_id']
    for col in numeric_cols:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
        
    # Log information about the dataframe
    logging.info(f"DataFrame columns: {metrics_df.columns.tolist()}")
    logging.info(f"DataFrame dtypes: {metrics_df.dtypes}")
    
    # 1. Scatter plot of Dice score vs ECE - with explicit error handling
    try:
        sns.scatterplot(x='dice', y='ece', data=metrics_df, ax=axes[0, 0], s=80, alpha=0.7)
        axes[0, 0].set_title('Segmentation Accuracy vs. Calibration Error', fontsize=14)
        axes[0, 0].set_xlabel('Dice Score (higher is better)', fontsize=12)
        axes[0, 0].set_ylabel('ECE (lower is better)', fontsize=12)
        
        # Add correlation coefficient
        corr = metrics_df['dice'].corr(metrics_df['ece'])
        axes[0, 0].annotate(f'Correlation: {corr:.3f}', 
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add explanation
        axes[0, 0].annotate(
            "This plot shows the relationship between segmentation\n"
            "accuracy (Dice) and calibration quality (ECE).\n"
            "Lower ECE means better calibrated probabilities.",
            xy=(0.5, 0.05), xycoords='axes fraction', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
    except Exception as e:
        logging.error(f"Error creating Dice vs ECE plot: {e}")
    
    # 2. Scatter plot of Dice score vs Sparsification Error - with explicit error handling
    try:
        # Use color parameter directly in scatterplot instead of trying to modify points after
        colors = ['green' if se > 0 else 'red' for se in metrics_df['sparsification_error']]
        scatter = sns.scatterplot(x='dice', y='sparsification_error', data=metrics_df, 
                                 ax=axes[0, 1], s=80, alpha=0.7, color=colors)
        axes[0, 1].set_title('Segmentation Accuracy vs. Uncertainty Quality', fontsize=14)
        axes[0, 1].set_xlabel('Dice Score (higher is better)', fontsize=12)
        axes[0, 1].set_ylabel('Sparsification Error (higher is better)', fontsize=12)
        
        # Add a legend to explain the colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                  label='Good uncertainty (SE > 0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
                  label='Poor uncertainty (SE < 0)')
        ]
        axes[0, 1].legend(handles=legend_elements, loc='lower right')
        
        # Add correlation coefficient
        corr = metrics_df['dice'].corr(metrics_df['sparsification_error'])
        axes[0, 1].annotate(f'Correlation: {corr:.3f}', 
                          xy=(0.05, 0.95), xycoords='axes fraction',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add explanation
        axes[0, 1].annotate(
            "This plot shows how uncertainty quality (SE) relates\n"
            "to segmentation accuracy (Dice).\n"
            "Positive SE (green): Uncertainty is meaningful\n"
            "Negative SE (red): Uncertainty is poorly estimated",
            xy=(0.5, 0.05), xycoords='axes fraction', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
    except Exception as e:
        logging.error(f"Error creating Dice vs Sparsification Error plot: {e}")
        logging.exception("Detailed traceback:")
    
    # 3. Histogram of ECE values - with explicit error handling
    try:
        sns.histplot(x='ece', data=metrics_df, kde=True, ax=axes[1, 0], color='teal')
        axes[1, 0].axvline(metrics_df['ece'].mean(), color='r', linestyle='--', 
                         label=f'Mean: {metrics_df["ece"].mean():.3f}')
        
        # Add quality bands
        axes[1, 0].axvspan(0, 0.01, alpha=0.2, color='green', label='Excellent (<0.01)')
        axes[1, 0].axvspan(0.01, 0.05, alpha=0.2, color='yellowgreen', label='Good (<0.05)')
        axes[1, 0].axvspan(0.05, 0.15, alpha=0.2, color='orange', label='Fair (<0.15)')
        axes[1, 0].axvspan(0.15, 1, alpha=0.2, color='red', label='Poor (>0.15)')
        
        axes[1, 0].set_title('Distribution of Expected Calibration Error', fontsize=14)
        axes[1, 0].set_xlabel('ECE (lower is better)', fontsize=12)
        axes[1, 0].legend(loc='upper right', fontsize=9)
        
        # Add explanation
        axes[1, 0].annotate(
            "ECE measures how well confidence values\n"
            "match actual frequencies.\n"
            "Lower values indicate better calibration.",
            xy=(0.7, 0.5), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
    except Exception as e:
        logging.error(f"Error creating ECE histogram: {e}")
    
    # 4. Histogram of Uncertain Pixel Percentage - with explicit error handling
    try:
        sns.histplot(x='uncertain_pixel_percent', data=metrics_df, kde=True, ax=axes[1, 1], color='purple')
        axes[1, 1].axvline(metrics_df['uncertain_pixel_percent'].mean(), color='r', linestyle='--',
                          label=f'Mean: {metrics_df["uncertain_pixel_percent"].mean():.1f}%')
        axes[1, 1].set_title('Distribution of Uncertain Pixel Percentage', fontsize=14)
        axes[1, 1].set_xlabel('Uncertain Pixel %', fontsize=12)
        axes[1, 1].legend(loc='upper right')
        
        # Add explanation
        axes[1, 1].annotate(
            "Shows what percentage of pixels have high uncertainty.\n"
            "Ideally correlates with difficult regions and errors.\n"
            "Too high: model is underconfident\n"
            "Too low: model might be overconfident",
            xy=(0.3, 0.7), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
    except Exception as e:
        logging.error(f"Error creating Uncertain Pixel histogram: {e}")
    
    plt.suptitle(f'Uncertainty Analysis Summary - {len(metrics_df)} Images', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / "uncertainty_summary.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create correlation heatmap with improved visualization
    try:
        plt.figure(figsize=(10, 8))
        # Only calculate correlation for numeric columns
        numeric_df = metrics_df[numeric_cols]
        corr_matrix = numeric_df.corr()
        
        # Use a diverging colormap centered at zero
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, 
                   annot=True, fmt='.2f', center=0, square=True, linewidths=.5)
        
        plt.title('Correlation Matrix of Uncertainty Metrics', fontsize=15)
        
        # Add explanation as text below
        plt.figtext(0.5, 0.01, 
                  "This heatmap shows how different metrics relate to each other.\n"
                  "Values close to 1 or -1 indicate strong correlation.\n"
                  "Look for strong correlations between uncertainty and performance metrics.",
                  ha="center", fontsize=12, 
                  bbox={"facecolor":"lightgoldenrodyellow", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_dir / "correlation_matrix.png", dpi=300)
        plt.close()
    except Exception as e:
        logging.error(f"Error creating correlation heatmap: {e}")
    
    # Create pairplot with more informative visualization
    try:
        selected_cols = ['dice', 'ece', 'brier', 'sparsification_error', 'uncertain_pixel_percent']
        # Make sure all selected columns exist in the dataframe
        existing_cols = [col for col in selected_cols if col in metrics_df.columns]
        if len(existing_cols) >= 2:  # Need at least 2 columns for a pairplot
            # Create a custom palette based on the dice score
            g = sns.pairplot(
                metrics_df[existing_cols],
                diag_kind="kde",
                plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k', 'linewidth': 0.5},
                corner=True,  # Only show the lower triangle
            )
            g.fig.suptitle('Pairwise Relationships Between Uncertainty Metrics', y=1.02, fontsize=16)
            
            # Add explanation text below the plot
            plt.figtext(0.5, 0.01, 
                      "This matrix shows how each pair of metrics relates to each other.\n"
                      "Look for patterns and correlations that can help interpret uncertainty.\n"
                      "Diagonal plots show the distribution of each individual metric.",
                      ha="center", fontsize=12, 
                      bbox={"facecolor":"lightgoldenrodyellow", "alpha":0.5, "pad":5})
            
            plt.subplots_adjust(top=0.95, bottom=0.1)
            plt.savefig(output_dir / "metrics_pairplot.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logging.warning(f"Not enough numeric columns for pairplot. Found: {existing_cols}")
    except Exception as e:
        logging.error(f"Error creating pairplot: {e}")
    
    # Create calibration analysis chart with better explanations
    try:
        if all(col in metrics_df.columns for col in ['max_calibration_error', 'mean_abs_calib_error', 'ece', 'dice']):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create a scatter plot where:
            # - x-axis is maximum calibration error
            # - y-axis is mean absolute calibration error
            # - color is based on ECE
            # - size is based on dice score
            scatter = ax.scatter(
                metrics_df['max_calibration_error'], 
                metrics_df['mean_abs_calib_error'],
                c=metrics_df['ece'], 
                s=metrics_df['dice'] * 200,  # Scale up for visibility
                alpha=0.7,
                cmap='viridis',
                edgecolors='k',
                linewidths=0.5
            )
            
            # Add colorbar and legend
            cbar = plt.colorbar(scatter)
            cbar.set_label('Expected Calibration Error (ECE)', fontsize=12)
            
            # Add a legend for size
            handles, labels = [], []
            for dice in [0.25, 0.5, 0.75, 1.0]:
                handles.append(plt.scatter([], [], s=dice*200, color='gray', alpha=0.7, edgecolors='k'))
                labels.append(f'Dice = {dice:.2f}')
            ax.legend(handles, labels, title="Dice Score", loc="upper left")
            
            # Add diagonal reference line
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='MCE = MACE')
            
            # Add quadrant labels
            ax.text(0.25, 0.9, "Consistent Calibration Errors", transform=ax.transAxes, 
                   ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
            ax.text(0.75, 0.1, "Outlier-dominated Errors", transform=ax.transAxes, 
                   ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
            
            # Improve axis labels and title
            ax.set_title('Calibration Error Analysis', fontsize=15)
            ax.set_xlabel('Maximum Calibration Error (MCE)\nHighest error in any confidence bin', fontsize=12)
            ax.set_ylabel('Mean Absolute Calibration Error (MACE)\nAverage error across all bins', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add explanation textbox
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
            plt.savefig(output_dir / "calibration_analysis.png", dpi=300)
            plt.close(fig)
        else:
            logging.warning("Missing required columns for calibration analysis chart")
    except Exception as e:
        logging.error(f"Error creating calibration analysis chart: {e}")

    # Generate detailed explanation PDF for uncertainty metrics
    create_explanation_report(metrics_df, output_dir)

def create_explanation_report(metrics_df, output_dir):
    """Create a PDF report with explanations of uncertainty metrics and visualizations."""
    pdf_path = output_dir / "uncertainty_metrics_explanation.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Understanding Uncertainty Metrics', fontsize=16, y=0.95)
        
        plt.figtext(0.5, 0.5, 
                  "This report explains the uncertainty metrics and visualizations\n"
                  f"for {len(metrics_df)} images analyzed with VAE-UNet.\n\n"
                  f"Mean Expected Calibration Error (ECE): {metrics_df['ece'].mean():.4f}\n"
                  f"Mean Dice Score: {metrics_df['dice'].mean():.4f}\n"
                  f"Mean Brier Score: {metrics_df['brier'].mean():.4f}\n"
                  f"Mean Sparsification Error: {metrics_df['sparsification_error'].mean():.4f}",
                  ha='center', fontsize=12, 
                  bbox={"facecolor":"lightgray", "alpha":0.5, "pad":15, "boxstyle":"round"})
        
        plt.axis('off')
        pdf.savefig(fig)
        plt.close()
        
        # Explanation of reliability diagrams
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Understanding Reliability Diagrams', fontsize=16, y=0.95)
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])
        
        # Create a sample reliability diagram
        ax1 = fig.add_subplot(gs[0])
        
        # Create sample data for the reliability diagram
        bin_accs = np.array([0.1, 0.23, 0.38, 0.49, 0.61, 0.68, 0.74, 0.81, 0.89, 0.95])
        bin_confs = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        bin_counts = np.array([100, 150, 200, 250, 300, 250, 200, 150, 100, 50])
        
        plot_reliability_diagram(bin_accs, bin_confs, bin_counts, ax=ax1)
        ax1.set_title('Example Reliability Diagram', fontsize=14)
        
        # Add explanatory text
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        explanation = (
            "RELIABILITY DIAGRAM EXPLANATION:\n\n"
            "A reliability diagram shows how well a model's predicted probabilities match actual outcomes.\n\n"
            "• Blue bars: The actual frequency of positive pixels in each confidence bin\n"
            "• Green bars: The mean predicted probability (confidence) for each bin\n"
            "• Gray histogram: Distribution of predictions across confidence levels\n"
            "• Red lines: Highlight gaps between confidence and actual frequency\n"
            "• Diagonal line: Perfect calibration (confidence = actual frequency)\n\n"
            "INTERPRETATION:\n\n"
            "• When blue bars are higher than green bars: Model is underconfident\n"
            "• When green bars are higher than blue bars: Model is overconfident\n"
            "• Expected Calibration Error (ECE): Weighted average of gaps between bars\n"
            "• Lower ECE values (closer to 0) indicate better calibration\n\n"
            "ECE Values Interpretation:\n"
            "• < 0.01: Excellent calibration\n"
            "• 0.01-0.05: Good calibration\n"
            "• 0.05-0.15: Fair calibration\n"
            "• > 0.15: Poor calibration\n\n"
            "WHY IT MATTERS:\n"
            "Good calibration means the confidence values from your model are reliable. For medical segmentation,\n"
            "this helps clinicians know when to trust the model's predictions."
        )
        
        ax2.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=11,
                transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])
        pdf.savefig(fig)
        plt.close()
        
        # Explanation of sparsification curves
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Understanding Sparsification Curves', fontsize=16, y=0.95)
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])
        
        # Create a sample sparsification curve
        ax1 = fig.add_subplot(gs[0])
        
        # Create sample data for the sparsification curve
        frac_removed = np.linspace(0, 0.99, 20)
        err_random = 1 - 0.2 * frac_removed  # Linear decrease
        err_uncertainty = 1 - 0.5 * frac_removed  # Steeper decrease
        
        plot_sparsification_curve(frac_removed, err_random, err_uncertainty, ax=ax1)
        ax1.set_title('Example Sparsification Curve', fontsize=14)
        
        # Add explanatory text
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        explanation = (
            "SPARSIFICATION CURVE EXPLANATION:\n\n"
            "A sparsification curve shows whether your model's uncertainty estimates correlate with actual errors.\n\n"
            "• Blue dashed line: Error when removing pixels randomly\n"
            "• Red solid line: Error when removing pixels with highest uncertainty first\n"
            "• Green/Red fill: Area between curves (Sparsification Error)\n"
            "• Red dot: Fraction of pixels that must be removed to halve the error\n\n"
            "INTERPRETATION:\n\n"
            "• If red line is below blue line (green area): Good uncertainty estimates!\n"
            "  This means removing high-uncertainty pixels reduces error faster than random removal.\n\n"
            "• If red line is above blue line (red area): Poor uncertainty estimates.\n"
            "  Your model's uncertainty doesn't correlate well with actual errors.\n\n"
            "• Sparsification Error (SE): Area between the curves\n"
            "  - Positive SE: Good uncertainty estimates\n"
            "  - Negative SE: Poor uncertainty estimates\n"
            "  - Larger positive values indicate better uncertainty quality\n\n"
            "WHY IT MATTERS:\n"
            "Good uncertainty estimates help identify which predictions might be wrong and\n"
            "where the model needs human verification in clinical applications."
        )
        
        ax2.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=11,
                transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])
        pdf.savefig(fig)
        plt.close()
        
        # Explanation of correlation matrix and calibration analysis
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Understanding the Visualization Plots', fontsize=16, y=0.95)
        
        explanation = (
            "CORRELATION MATRIX EXPLANATION:\n\n"
            "The correlation matrix shows how different metrics relate to each other:\n"
            "• Values close to 1: Strong positive correlation (one increases, the other increases)\n"
            "• Values close to -1: Strong negative correlation (one increases, the other decreases)\n"
            "• Values close to 0: Little or no correlation\n\n"
            "Key relationships to look for:\n"
            "• Dice Score vs. Uncertainty Metrics: Does better performance correlate with better calibration?\n"
            "• ECE vs. Sparsification Error: Do different uncertainty metrics agree with each other?\n\n"
            "CALIBRATION ANALYSIS PLOT EXPLANATION:\n\n"
            "This plot helps understand the pattern of calibration errors:\n"
            "• X-axis: Maximum Calibration Error (MCE) - the largest calibration error in any bin\n"
            "• Y-axis: Mean Absolute Calibration Error (MACE) - the average calibration error\n"
            "• Color: Expected Calibration Error (ECE) - weighted average of calibration errors\n"
            "• Size: Dice Score - larger points indicate better segmentation performance\n\n"
            "Interpretation by location:\n"
            "• Points near the diagonal: Errors are consistent across all confidence levels\n"
            "• Points below diagonal: Errors concentrated in specific confidence bins\n"
            "• Bottom-left corner: Best calibration overall (low errors)\n"
            "• Larger points in bottom-left: Ideal models (good performance, good calibration)\n\n"
            "PAIRPLOT EXPLANATION:\n\n"
            "The pairplot shows the relationships between all pairs of metrics:\n"
            "• Diagonal: Distribution of each individual metric\n"
            "• Off-diagonal: Scatter plots showing relationship between pairs of metrics\n\n"
            "TEMPERATURE SCALING:\n\n"
            "If your model has an ECE of 0.005, that's excellent calibration!\n"
            "With temperature=2.0 giving better results, this suggests your model was slightly overconfident\n"
            "at the default temperature (T=1.0). Higher temperatures 'soften' predictions, making very\n"
            "confident predictions less extreme."
        )
        
        plt.figtext(0.5, 0.5, explanation, ha='center', va='center', fontsize=11,
                  wrap=True, bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        pdf.savefig(fig)
        plt.close()
    
    logging.info(f"Created explanation report at {pdf_path}")

def create_calibration_visualizations(all_predictions, all_ground_truths, output_dir):
    """Create visualizations specifically for calibration analysis."""
    if not all_predictions or not all_ground_truths:
        logging.warning("No prediction data available for calibration analysis")
        return
        
    # Concatenate all predictions and ground truths
    all_pred_flat = np.concatenate(all_predictions)
    all_gt_flat = np.concatenate(all_ground_truths)
    
    # Ensure ground truth values are exactly 0 or 1 for calibration curve
    all_gt_flat = np.round(all_gt_flat).astype(int)
    
    # Calculate global calibration curve
    global_prob_true, global_prob_pred = calibration_curve(
        all_gt_flat, all_pred_flat, n_bins=10, strategy='uniform'
    )
    
    # Calculate global ECE
    global_ece = np.sum(
        np.abs(global_prob_true - global_prob_pred) * 
        np.histogram(all_pred_flat, bins=10)[0] / len(all_pred_flat)
    )
    
    # Create global calibration plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot calibration curve
    ax.plot(global_prob_pred, global_prob_true, marker='o', linewidth=2,
           label=f'Calibration Curve (ECE={global_ece:.4f})')
    
    # Plot the diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Add histogram of prediction confidences
    ax2 = ax.twinx()
    ax2.hist(all_pred_flat, bins=20, alpha=0.3, density=True,
            color='gray', label='Prediction Distribution')
    ax2.set_ylabel('Density')
    
    # Set labels and title
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Global Calibration Curve (All Images)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "global_calibration_curve.png", dpi=200)
    plt.close(fig)

def perform_temperature_analysis(all_predictions, all_ground_truths, output_dir, temperatures):
    """Analyze the effect of temperature on calibration."""
    if not all_predictions or not all_ground_truths:
        logging.warning("No prediction data available for temperature analysis")
        return
    
    # Concatenate all predictions and ground truths
    all_pred_flat = np.concatenate(all_predictions)
    all_gt_flat = np.concatenate(all_ground_truths)
    
    # Temperature scaling analysis
    ece_values = []
    
    for temp in temperatures:
        # Apply temperature scaling
        # Scale logits with temperature: logits / T
        # which means we need to convert probabilities back to logits
        # logits = log(p / (1-p))
        # then scale and convert back
        # p_scaled = sigmoid(logits / T)
        # This simplifies to:
        scaled_preds = 1 / (1 + np.exp(-(np.log(all_pred_flat / (1 - all_pred_flat + 1e-7)) / temp)))
        
        # Calculate ECE for this temperature
        temp_prob_true, temp_prob_pred = calibration_curve(
            all_gt_flat, scaled_preds, n_bins=10, strategy='uniform'
        )
        
        temp_ece = np.sum(
            np.abs(temp_prob_true - temp_prob_pred) * 
            np.histogram(scaled_preds, bins=10)[0] / len(scaled_preds)
        )
        
        ece_values.append(temp_ece)
    
    # Find optimal temperature (lowest ECE)
    optimal_temp_idx = np.argmin(ece_values)
    optimal_temp = temperatures[optimal_temp_idx]
    optimal_ece = ece_values[optimal_temp_idx]
    
    # Plot ECE vs temperature
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, ece_values, marker='o', linewidth=2)
    plt.axvline(optimal_temp, color='r', linestyle='--', 
               label=f'Optimal T={optimal_temp:.2f}, ECE={optimal_ece:.4f}')
    plt.xlabel('Temperature')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('Effect of Temperature on Calibration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_effect.png", dpi=200)
    plt.close()
    
    # Create dataframe and save results
    summary_df = pd.DataFrame({
        'temperature': temperatures,
        'ece': ece_values
    })
    summary_df.to_csv(output_dir / "temperature_scaling.csv", index=False)
    
    # Generate multi-temperature calibration curves
    plt.figure(figsize=(12, 8))
    
    # Plot diagonal reference
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot curves for selected temperatures
    selected_temps = [temperatures[0], 1.0, optimal_temp, temperatures[-1]]
    selected_temps = sorted(list(set(selected_temps)))  # Ensure no duplicates and order properly
    
    for temp in selected_temps:
        # Skip if temp not in our temperature list
        if temp not in temperatures:
            continue
            
        # Get index of this temperature
        temp_idx = temperatures.index(temp)
        
        # Apply temperature scaling
        scaled_preds = 1 / (1 + np.exp(-(np.log(all_pred_flat / (1 - all_pred_flat + 1e-7)) / temp)))
        
        # Calculate calibration curve
        temp_prob_true, temp_prob_pred = calibration_curve(
            all_gt_flat, scaled_preds, n_bins=10, strategy='uniform'
        )
        
        # Special formatting for optimal temperature
        if temp == optimal_temp:
            plt.plot(temp_prob_pred, temp_prob_true, marker='o', linewidth=3,
                   label=f'T={temp:.2f} (Optimal, ECE={ece_values[temp_idx]:.4f})')
        else:
            plt.plot(temp_prob_pred, temp_prob_true, marker='o', linewidth=2,
                   label=f'T={temp:.2f} (ECE={ece_values[temp_idx]:.4f})')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Effect of Temperature on Calibration Curves')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_calibration_curves.png", dpi=200)
    plt.close()
    
    # Log the recommendation
    logging.info(f"TEMPERATURE ANALYSIS RESULTS:")
    logging.info(f"Optimal temperature: T={optimal_temp:.2f} (ECE={optimal_ece:.4f})")
    if optimal_temp > 1.0:
        logging.info(f"Model is likely overconfident at default temperature (T=1.0)")
    elif optimal_temp < 1.0:
        logging.info(f"Model is likely underconfident at default temperature (T=1.0)")
    else:
        logging.info(f"Default temperature (T=1.0) is optimal")
    
    return optimal_temp, optimal_ece

def print_interpretation_guide(args):
    """Print a guide for interpreting the analysis results."""
    print("\n" + "="*80)
    print("ANALYSIS RESULTS INTERPRETATION GUIDE".center(80))
    print("="*80)
    
    if args.uncertainty:
        print("\n1. UNCERTAINTY ANALYSIS")
        print("   - Reliability diagrams show if your model's probabilities match actual frequencies")
        print("   - Blue bars should match green bars closely for well-calibrated models")
        print("   - Sparsification curves show if uncertainty estimates correlate with errors")
        print("   - Green area (red line below blue) means good uncertainty estimates")
        print("   - The red dot shows what fraction of pixels to remove to halve the error")
    
    if args.calibration:
        print("\n2. CALIBRATION ANALYSIS")
        print("   - Calibration curves show if predicted probabilities match observed frequencies")
        print("   - Closer to the diagonal line means better calibration")
        print("   - ECE values under 0.01 are excellent, 0.01-0.05 good, 0.05-0.15 fair, >0.15 poor")
    
    if args.temperature_sweep:
        print("\n3. TEMPERATURE ANALYSIS")
        print("   - Temperature scaling can improve calibration without affecting rankings")
        print("   - Higher temperatures (T>1) reduce overconfidence")
        print("   - Lower temperatures (T<1) reduce underconfidence")
        print("   - The optimal temperature minimizes calibration error (ECE)")
    
    # Provide inference recommendation
    if args.temperature_sweep:
        print("\nFor optimal calibration in production, use:")
        # Get the optimal temperature (this information would need to be available)
        print(f"   python inference.py --model {os.path.basename(args.model)} --lesion_type {args.lesion_type} --temperature T_OPT")
        print("   where T_OPT is the optimal temperature shown in temperature_effect.png")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = get_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetResNet(n_channels=3, n_classes=1, latent_dim=32, use_attention=True)
    
    logging.info(f'Loading model {args.model}')
    model_path = Path(f'./checkpoints/{args.model}')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load dataset
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
    
    # Run unified analysis
    metrics_df = analyze_model(model, test_dataset, args)
