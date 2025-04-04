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
from utils.data_loading import IDRIDDataset
from unet.unet_resnet import UNetResNet
from utils.uncertainty_metrics import (
    calculate_expected_calibration_error, 
    brier_score,
    calculate_sparsification_metrics,
    plot_reliability_diagram,
    plot_sparsification_curve
)
from utils.tensor_utils import ensure_dict_python_scalars, fix_dataframe_tensors  # Add this import
from visualize_vae import get_segmentation_distribution, track_memory

def get_args():
    parser = argparse.ArgumentParser(description='Analyze uncertainty metrics across test images')
    parser.add_argument('--model', '-m', default='best_model.pth', metavar='FILE', help='Model file')
    parser.add_argument('--lesion_type', type=str, required=True, default='EX', choices=['EX', 'HE', 'MA','OD'], help='Lesion type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--samples', type=int, default=30, help='Number of samples for ensemble prediction')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size (0 for full image)')
    parser.add_argument('--overlap', type=int, default=100, help='Overlap between patches')
    parser.add_argument('--output_dir', type=str, default='./analysis', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for resizing (default: 1.0)')
    args = parser.parse_args()
    
    # Convert patch_size=0 to None for full image mode
    if args.patch_size == 0:
        args.patch_size = None
        
    return args

@track_memory
def analyze_dataset_uncertainty(model, dataset, args):
    """Analyze uncertainty metrics across the dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.lesion_type}_T{args.temperature}_N{args.samples}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists to store metrics
    metrics_data = []
    
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
            
            # Calculate sparsification metrics
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
            
            # Generate individual image report
            img_output_dir = output_dir / "individual_reports"
            img_output_dir.mkdir(exist_ok=True)
            
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
    
    # Create metrics dataframe and apply our tensor fixing utility
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = fix_dataframe_tensors(metrics_df)  # Extra safety check before visualization
    
    # Additional check: ensure all numeric columns are properly converted
    for col in metrics_df.columns:
        if col != 'img_id':  # Skip the image ID column
            try:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
            except Exception as e:
                logging.warning(f"Could not convert column {col} to numeric: {e}")
    
    metrics_csv_path = output_dir / "uncertainty_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved metrics data to {metrics_csv_path}")
    
    # Create summary visualizations
    create_summary_visualizations(metrics_df, output_dir)
    
    return metrics_df

def create_summary_visualizations(metrics_df, output_dir):
    """Create summary visualizations for uncertainty metrics."""
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Create a 2x2 grid of plots
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
        sns.scatterplot(x='dice', y='ece', data=metrics_df, ax=axes[0, 0])
        axes[0, 0].set_title('Dice Score vs Expected Calibration Error')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('ECE')
        
        # Add correlation coefficient
        corr = metrics_df['dice'].corr(metrics_df['ece'])
        axes[0, 0].annotate(f'Correlation: {corr:.3f}', 
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    except Exception as e:
        logging.error(f"Error creating Dice vs ECE plot: {e}")
    
    # 2. Scatter plot of Dice score vs Sparsification Error - with explicit error handling
    try:
        sns.scatterplot(x='dice', y='sparsification_error', data=metrics_df, ax=axes[0, 1])
        axes[0, 1].set_title('Dice Score vs Sparsification Error')
        axes[0, 1].set_xlabel('Dice Score')
        axes[0, 1].set_ylabel('Sparsification Error')
        
        # Add correlation coefficient
        corr = metrics_df['dice'].corr(metrics_df['sparsification_error'])
        axes[0, 1].annotate(f'Correlation: {corr:.3f}', 
                          xy=(0.05, 0.95), xycoords='axes fraction',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    except Exception as e:
        logging.error(f"Error creating Dice vs Sparsification Error plot: {e}")
    
    # 3. Histogram of ECE values - with explicit error handling
    try:
        sns.histplot(x='ece', data=metrics_df, kde=True, ax=axes[1, 0])
        axes[1, 0].axvline(metrics_df['ece'].mean(), color='r', linestyle='--', 
                         label=f'Mean: {metrics_df["ece"].mean():.3f}')
        axes[1, 0].set_title('Distribution of Expected Calibration Error')
        axes[1, 0].set_xlabel('ECE')
        axes[1, 0].legend()
    except Exception as e:
        logging.error(f"Error creating ECE histogram: {e}")
    
    # 4. Histogram of Uncertain Pixel Percentage - with explicit error handling
    try:
        sns.histplot(x='uncertain_pixel_percent', data=metrics_df, kde=True, ax=axes[1, 1])
        axes[1, 1].axvline(metrics_df['uncertain_pixel_percent'].mean(), color='r', linestyle='--',
                          label=f'Mean: {metrics_df["uncertain_pixel_percent"].mean():.1f}%')
        axes[1, 1].set_title('Distribution of Uncertain Pixel Percentage')
        axes[1, 1].set_xlabel('Uncertain Pixel %')
        axes[1, 1].legend()
    except Exception as e:
        logging.error(f"Error creating Uncertain Pixel histogram: {e}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_summary.png", dpi=300)
    plt.close(fig)
    
    # Create correlation heatmap - only using numeric columns
    try:
        plt.figure(figsize=(10, 8))
        # Only calculate correlation for numeric columns
        numeric_df = metrics_df[numeric_cols]
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlation between Metrics')
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_matrix.png", dpi=300)
        plt.close()
    except Exception as e:
        logging.error(f"Error creating correlation heatmap: {e}")
    
    # Create pairplot for all metrics - only using numeric columns
    try:
        selected_cols = ['dice', 'ece', 'brier', 'sparsification_error', 'uncertain_pixel_percent']
        # Make sure all selected columns exist in the dataframe
        existing_cols = [col for col in selected_cols if col in metrics_df.columns]
        if len(existing_cols) >= 2:  # Need at least 2 columns for a pairplot
            g = sns.pairplot(
                metrics_df[existing_cols],
                plot_kws={'alpha': 0.6}
            )
            g.fig.suptitle('Pairwise Relationships between Metrics', y=1.02)
            plt.savefig(output_dir / "metrics_pairplot.png", dpi=300)
            plt.close()
        else:
            logging.warning(f"Not enough numeric columns for pairplot. Found: {existing_cols}")
    except Exception as e:
        logging.error(f"Error creating pairplot: {e}")
    
    # Create calibration analysis chart
    try:
        if all(col in metrics_df.columns for col in ['max_calibration_error', 'mean_abs_calib_error', 'ece', 'dice']):
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df.plot.scatter(x='max_calibration_error', y='mean_abs_calib_error', 
                                c='ece', colormap='viridis', 
                                s=metrics_df['dice']*100, alpha=0.7, ax=ax)
            plt.colorbar(ax.collections[0], label='ECE')
            ax.set_title('Calibration Error Analysis')
            ax.set_xlabel('Maximum Calibration Error (MCE)')
            ax.set_ylabel('Mean Absolute Calibration Error (MACE)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "calibration_analysis.png", dpi=300)
            plt.close(fig)
        else:
            logging.warning("Missing required columns for calibration analysis chart")
    except Exception as e:
        logging.error(f"Error creating calibration analysis chart: {e}")

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
    
    # Run analysis
    metrics_df = analyze_dataset_uncertainty(model, test_dataset, args)
    
    # Print summary statistics
    logging.info("\nSummary Statistics:")
    logging.info(f"Number of images analyzed: {len(metrics_df)}")
    logging.info(f"Average Dice Score: {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
    logging.info(f"Average ECE: {metrics_df['ece'].mean():.4f} ± {metrics_df['ece'].std():.4f}")
    logging.info(f"Average Brier Score: {metrics_df['brier'].mean():.4f} ± {metrics_df['brier'].std():.4f}")
    logging.info(f"Average Sparsification Error: {metrics_df['sparsification_error'].mean():.4f} ± {metrics_df['sparsification_error'].std():.4f}")
    logging.info(f"Average Uncertain Pixel %: {metrics_df['uncertain_pixel_percent'].mean():.2f}% ± {metrics_df['uncertain_pixel_percent'].std():.2f}%")
