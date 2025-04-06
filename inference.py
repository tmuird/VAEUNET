import torch
import argparse
import logging
import numpy as np
from pathlib import Path
from unet.unet_resnet import UNetResNet
from utils.data_loading import IDRIDDataset
from visualize_vae import get_segmentation_distribution

def get_args():
    parser = argparse.ArgumentParser(description='Run inference with optimal calibration')
    parser.add_argument('--model', '-m', default='best_model.pth', help='Model file')
    parser.add_argument('--lesion_type', type=str, required=True, default='EX', 
                      choices=['EX', 'HE', 'MA', 'OD'], help='Lesion type')
    parser.add_argument('--temperature', type=float, default=2.0, 
                      help='Temperature for sampling (default: 2.0 for optimal calibration)')
    parser.add_argument('--samples', type=int, default=30, help='Number of samples')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    # Add other arguments as needed
    return parser.parse_args()

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.lesion_type}_T{args.temperature}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetResNet(n_channels=3, n_classes=1, latent_dim=32, use_attention=True)
    
    logging.info(f'Loading model {args.model}')
    model_path = Path(f'./checkpoints/{args.model}')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load dataset 
    test_dataset = IDRIDDataset(
        base_dir='./data',
        split='test',
        scale=1.0,
        patch_size=None,  # Use full images for inference
        lesion_type=args.lesion_type,
        skip_border_check=True
    )
    
    # Process each unique image
    processed_ids = set()
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        img_id = sample['img_id']
        
        if img_id in processed_ids:
            continue
            
        processed_ids.add(img_id)
        logging.info(f"Processing image {img_id}")
        
        try:
            # Generate calibrated segmentations using the optimal temperature
            segmentations, _, _, _ = get_segmentation_distribution(
                model, img_id, dataset=test_dataset, 
                num_samples=args.samples,
                temperature=args.temperature,  # Using the optimal temperature
                batch_size=args.batch_size
            )
            
            # Calculate mean prediction (well-calibrated)
            mean_pred = segmentations.mean(dim=0)[0]
            
            # Save the calibrated prediction
            np.save(output_dir / f"{img_id}_calibrated_prediction.npy", 
                   mean_pred.cpu().numpy())
            
            # You might also want to save a thresholded binary version
            binary_pred = (mean_pred > 0.5).float()
            np.save(output_dir / f"{img_id}_binary_prediction.npy", 
                   binary_pred.cpu().numpy())
            
            logging.info(f"Saved calibrated prediction for {img_id}")
            
        except Exception as e:
            logging.error(f"Error processing image {img_id}: {e}")
            continue

if __name__ == '__main__':
    main()
