import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.calibration import calibration_curve

from utils.data_loading import IDRIDDataset
from unet.unet_resnet import UNetResNet
from visualize_vae import get_segmentation_distribution, track_memory
from utils.uncertainty_metrics import calculate_expected_calibration_error

def get_args():
    parser = argparse.ArgumentParser(description='Analyze model calibration')
    parser.add_argument('--model', '-m', default='best_model.pth', metavar='FILE', help='Model file')
    parser.add_argument('--lesion_type', type=str, required=True, default='EX', choices=['EX', 'HE', 'MA','OD'], help='Lesion type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--samples', type=int, default=30, help='Number of samples')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size (0 for full image)')
    parser.add_argument('--output_dir', type=str, default='./calibration', help='Output directory')
    parser.add_argument('--max_images', type=int, default=10, help='Maximum number of images to process')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for resizing (default: 1.0)')
    args = parser.parse_args()
    
    if args.patch_size == 0:
        args.patch_size = None
        
    return args

@track_memory
def analyze_calibration(model, dataset, args):
    """Analyze model calibration in detail."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    output_dir = Path(args.output_dir) / f"{args.lesion_type}_T{args.temperature}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data storage
    all_predictions = []
    all_ground_truths = []
    image_data = []
    
    # Process each image only once
    processed_ids = set()
    
    for i in tqdm(range(len(dataset)), desc="Analyzing calibration"):
        sample = dataset[i]
        img_id = sample['img_id']
        
        if img_id in processed_ids:
            continue
            
        processed_ids.add(img_id)
        logging.info(f"Processing image {img_id}")
        
        try:
            # Generate segmentations
            segmentations, mask, _, _ = get_segmentation_distribution(
                model, img_id, dataset=dataset, 
                num_samples=args.samples,
                patch_size=args.patch_size,
                temperature=args.temperature,
                enable_dropout=False
            )
            
            # Move to CPU
            segmentations_cpu = segmentations.cpu()
            mask_cpu = mask.cpu()
            
            # Calculate mean prediction
            mean_pred = segmentations_cpu.mean(dim=0)[0]  # Shape: [H, W]
            
            # Flatten for analysis
            pred_flat = mean_pred.flatten().numpy()
            gt_flat = mask_cpu[0, 0].flatten().numpy()
            
            # Store for global analysis
            all_predictions.append(pred_flat)
            all_ground_truths.append(gt_flat)
            
            # Calculate ECE for this image
            ece, bin_accs, bin_confs, bin_counts = calculate_expected_calibration_error(
                mean_pred.unsqueeze(0), mask_cpu[0, 0]
            )
            
            # Calculate calibration curve using scikit-learn
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
            plt.savefig(output_dir / f"{img_id}_calibration_curve.png", dpi=200)
            plt.close(fig)
            
            # Store image metrics
            image_data.append({
                'img_id': img_id,
                'ece': ece,
                'fraction_positive': gt_flat.mean(),
                'mean_confidence': pred_flat.mean(),
                'confidence_std': pred_flat.std()
            })
            
            # Free memory
            del segmentations_cpu, mask_cpu, mean_pred
            
        except Exception as e:
            logging.error(f"Error processing image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        if len(processed_ids) >= args.max_images:
            break
    
    # Concatenate all predictions and ground truths
    if all_predictions and all_ground_truths:
        all_pred_flat = np.concatenate(all_predictions)
        all_gt_flat = np.concatenate(all_ground_truths)
        
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
        
        # Temperature scaling analysis
        temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
        ece_values = []
        
        for temp in temperatures:
            # Apply temperature scaling
            scaled_preds = 1 / (1 + np.exp(-(np.log(all_pred_flat / (1 - all_pred_flat)) / temp)))
            
            # Calculate ECE for this temperature
            temp_prob_true, temp_prob_pred = calibration_curve(
                all_gt_flat, scaled_preds, n_bins=10, strategy='uniform'
            )
            
            temp_ece = np.sum(
                np.abs(temp_prob_true - temp_prob_pred) * 
                np.histogram(scaled_preds, bins=10)[0] / len(scaled_preds)
            )
            
            ece_values.append(temp_ece)
        
        # Plot ECE vs temperature
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, ece_values, marker='o', linewidth=2)
        plt.xlabel('Temperature')
        plt.ylabel('Expected Calibration Error (ECE)')
        plt.title('Effect of Temperature on Calibration')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "temperature_effect.png", dpi=200)
        plt.close()
        
        # Create dataframe and save results
        image_df = pd.DataFrame(image_data)
        image_df.to_csv(output_dir / "calibration_metrics.csv", index=False)
        
        # Create summary visualizations
        summary_df = pd.DataFrame({
            'temperature': temperatures,
            'ece': ece_values
        })
        summary_df.to_csv(output_dir / "temperature_scaling.csv", index=False)
        
        # Final summary visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(data=image_df, x='img_id', y='ece')
        plt.title('Expected Calibration Error by Image')
        plt.xlabel('Image ID')
        plt.ylabel('ECE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "ece_by_image.png", dpi=200)
        plt.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = get_args()
    
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
    
    # Run calibration analysis
    analyze_calibration(model, test_dataset, args)
