import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Optional

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import IDRIDDataset
from utils.dice_score import dice_loss
import numpy as np

# Add CUDA error handling
if torch.cuda.is_available():
    try:
        _ = torch.cuda.device_count()
    except RuntimeError as e:
        logging.error(f"CUDA initialization error: {e}")
        logging.warning("Falling back to CPU")
        torch.cuda.is_available = lambda: False

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        img_scale: float = 0.25,
        patch_size: Optional[int] = 128,
        amp: bool = True,
        gradient_clipping: float = 1.0,
):
    # Check if data directories exist
    data_dir = Path('./data')
    if not (data_dir / 'imgs' / 'train').exists():
        raise RuntimeError(f"Training images directory not found at {data_dir / 'imgs' / 'train'}")
    if not (data_dir / 'masks' / 'train').exists():
        raise RuntimeError(f"Training masks directory not found at {data_dir / 'masks' / 'train'}")

    # Load datasets with logging
    train_dataset = IDRIDDataset(base_dir='./data', split='train', 
                                scale=img_scale, patch_size=patch_size,
                                lesion_type='EX')
    val_dataset = IDRIDDataset(base_dir='./data', split='val', 
                              scale=img_scale, patch_size=patch_size,
                              lesion_type='EX')
    
    logging.info(f'Dataset sizes:')
    logging.info(f'- Training: {len(train_dataset)} images')
    logging.info(f'- Validation: {len(val_dataset)} images')

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, 
                           shuffle=False, 
                           batch_size=1,  # Force batch_size=1 for validation
                           num_workers=1, 
                           pin_memory=True)

    # Initialize logging
    experiment = wandb.init(project='IDRID-UNET', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # Use appropriate optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,       # Less aggressive reduction
        patience=15,      # Wait longer before reducing
        min_lr=1e-5,     # Don't go too low
        verbose=True
    )
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    
    # Modify the class weights calculation for binary classification
    def calculate_class_weights(dataset):
        pos_count = 0
        total_pixels = 0
        for sample in dataset:
            mask = sample['mask']
            pos_count += (mask > 0).float().sum()
            total_pixels += mask.numel()
        
        neg_count = total_pixels - pos_count
        # Calculate weights for background (0) and lesion (1)
        weights = torch.tensor([neg_count, pos_count])
        weights = 1.0 / (weights + 1e-8)
        weights = weights / weights.sum()
        
        logging.info(f'Class weights calculated - Background: {weights[0]:.4f}, Lesion: {weights[1]:.4f}')
        return weights.to(device)

    class_weights = calculate_class_weights(train_dataset)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))  # Weight positive examples more
    
    # Modify loss calculation
    global_step = 0

    # Move model to device and set to appropriate dtype
    model = model.to(device=device)
    
    # Add debug logging for shapes
    for batch in train_loader:
        images = batch['image']
        masks = batch['mask']
        logging.info(f"Image shape: {images.shape}")
        logging.info(f"Mask shape: {masks.shape}")
        break

    # Add memory debugging
    if torch.cuda.is_available():
        logging.info(f"GPU Memory before data loading: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)

                logging.debug(f"Batch image shape: {images.shape}")
                logging.debug(f"Batch mask shape: {true_masks.shape}")

                with torch.autocast(device_type='cuda', enabled=amp):
                    masks_pred = model(images)
                    logging.debug(f"Predicted mask shape: {masks_pred.shape}")
                    
                    # Ensure shapes match
                    if masks_pred.shape != true_masks.shape:
                        logging.error(f"Shape mismatch - Pred: {masks_pred.shape}, True: {true_masks.shape}")
                        raise RuntimeError(f"Shape mismatch in model output")

                    # Add loss debugging
                    if global_step % 10 == 0:  # Log every 10 steps
                        with torch.no_grad():
                            pred_sigmoid = torch.sigmoid(masks_pred)
                            pos_pixels = (true_masks > 0.5).float().sum()
                            pred_pos_pixels = (pred_sigmoid > 0.5).float().sum()
                            logging.info(f'Positive pixels - True: {pos_pixels.item()}, Predicted: {pred_pos_pixels.item()}')

                    # Calculate BCE loss
                    bce_loss = criterion(masks_pred, true_masks)
                    # Add Dice loss
                    dice = dice_loss(
                        torch.sigmoid(masks_pred), 
                        true_masks,
                        multiclass=False  # Changed to False since we're doing binary
                    )
                    # Combine losses with weighting
                    loss = bce_loss + dice

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                
                if gradient_clipping > 0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (len(train_dataset) // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score['mean_dice'])
                        
                        # Log validation metrics
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score['mean_dice'],
                            'epoch': epoch
                        })

                        # Log a batch of validation images
                        with torch.no_grad():
                            # Get a batch of validation images
                            val_batch = next(iter(val_loader))
                            images = val_batch['image']
                            true_masks = val_batch['mask']
                            
                            # Get model prediction
                            model.eval()
                            mask_pred = model(images.to(device))
                            mask_pred = torch.sigmoid(mask_pred)
                            
                            # Convert to numpy for visualization
                            img = images[0].cpu().numpy()
                            true_mask = true_masks[0].cpu().numpy()
                            pred_mask = mask_pred[0].cpu().numpy()
                            
                            # Debug logging
                            logging.info(f'Validation sample ranges:')
                            logging.info(f'- True mask: min={true_mask.min():.3f}, max={true_mask.max():.3f}')
                            logging.info(f'- Pred mask: min={pred_mask.min():.3f}, max={pred_mask.max():.3f}')
                            
                            # Ensure proper normalization for visualization
                            img = img.transpose(1, 2, 0)  # CHW -> HWC
                            img = (img * 255).astype(np.uint8)
                            true_mask = (true_mask[0] * 255).astype(np.uint8)  # Take first channel
                            pred_mask = (pred_mask[0] * 255).astype(np.uint8)  # Take first channel
                            
                            experiment.log({
                                'validation sample': {
                                    'image': wandb.Image(img),
                                    'true_mask': wandb.Image(true_mask),
                                    'pred_mask': wandb.Image(pred_mask),
                                }
                            })
                            model.train()

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--patch-size', '-p', type=lambda x: None if x.lower() == 'none' else int(x), 
                        default=None, 
                        help='Size of patches to extract (use None to disable)')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Gradient clipping value')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Set up logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(f'Training configuration:')
    logging.info(f'- Patch size: {args.patch_size}')
    logging.info(f'- Scale: {args.scale}')
    logging.info(f'- Batch size: {args.batch_size}')
    logging.info(f'- Learning rate: {args.learning_rate}')

    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(device=device)
    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            img_scale=args.scale,
            patch_size=args.patch_size,
            amp=args.amp,
            gradient_clipping=args.gradient_clipping
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                     'Try reducing the batch size or image scale.')
        raise
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
