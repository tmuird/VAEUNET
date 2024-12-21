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
from utils import metrics
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import IDRIDDataset
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
        img_scale: float = 1,
        patch_size: Optional[int] = None,
        amp: bool = True,
        bilinear: bool = False,
        gradient_clipping: float = 1.0,
):
    # Initialize best_dice at the start
    best_dice = 0.0
    
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
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp, patch_size=patch_size, classes=1, lesion_type='EX')
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_checkpoint}godfather
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # Use appropriate optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=10,     # Reduce patience
        min_lr=1e-6,
        verbose=True,
        factor=0.5
        
    )
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()


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
                images = batch['image']
                true_masks = batch['mask']

                # Move to device and set requires_grad
                images = images.to(device=device, dtype=torch.float32)
                images.requires_grad = True
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        # Combine Focal Loss and Dice Loss
                        focal = metrics.focal_loss(
                            masks_pred, 
                            true_masks.float(),
                            alpha=0.85,    # Higher alpha because exudates are very sparse
                            gamma=2.5      # Higher gamma to focus on hard examples
                        )
                        dice = metrics.dice_loss(
                            torch.sigmoid(masks_pred), 
                            true_masks, 
                            multiclass=False
                        )
                        
                        # Balance between Focal and Dice
                        loss = 0.4 * focal + 0.6 * dice  # More weight on Dice for better boundary detection
                    else:
                        loss = criterion(masks_pred, true_masks)

                

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
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

                        val_metrics = evaluate(model, val_loader, device, amp)
                        # Update scheduler with dice score
                        scheduler.step(val_metrics['dice'])
                        
                        # Log validation metrics to wandb
                        experiment.log({
                            **{f'val/{k}': v for k, v in val_metrics.items()},
                            'epoch': epoch,
                            'step': global_step,
                            'learning_rate': optimizer.param_groups[0]['lr']
                        })
                        with torch.no_grad():
                            # Get a validation batch for visualization
                            val_batch = next(iter(val_loader))
                            val_images = val_batch['image'].to(device=device, dtype=torch.float32)
                            val_masks = val_batch['mask'].to(device=device, dtype=torch.float32)

                            # Get predictions for the validation batch
                            model.eval()  # Ensure model is in eval mode

                            val_masks_pred = model(val_images)
                            val_masks_pred = torch.sigmoid(val_masks_pred)

                        # Log images and masks to wandb
                        if val_images is not None:
                            # Get first image from batch
                            img = val_images[0].cpu().numpy()  # Change to HWC format
                            pred_mask = val_masks_pred[0].cpu().numpy()
                            true_mask = val_masks[0].cpu().numpy()
                            

                            logging.info(f'Mask shapes - Pred: {pred_mask.shape}, True: {true_mask.shape}')
                            logging.info(f'Mask values - Pred: min={pred_mask.min()}, max={pred_mask.max()}, True: min={true_mask.min()}, max={true_mask.max()}')
                                                        # Ensure proper normalization for visualization
                            img = img.transpose(1, 2, 0)  # CHW -> HWC
                            img = (img * 255).astype(np.uint8)
                            true_mask = (true_mask[0] * 255).astype(np.uint8)  # Take first channel
                            pred_mask = (pred_mask[0] * 255).astype(np.uint8)  # Take first channel
                            
                            experiment.log({
                                'validation_images': wandb.Image(
                                    img,  # Original image
                                    masks={
                                        "predictions": {
                                            "mask_data": pred_mask,
                                            "class_labels": {1: "lesion"}
                                        },
                                        "ground_truth": {
                                            "mask_data": true_mask,
                                            "class_labels": {1: "lesion"}
                                        }
                                    }
                                )
                            })
                        
                        model.train()  # Set model back to training mode



                        # Save best model based on dice score
                        if val_metrics['dice'] > best_dice:
                            best_dice = val_metrics['dice']
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            torch.save(state_dict, str(dir_checkpoint / 'best_model.pth'))
                            logging.info(f'Best model saved! Dice score: {val_metrics["dice"]:.4f}')

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
    parser.add_argument('--use-checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing to reduce memory usage')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling')
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

    model = UNet(
        n_channels=3, 
        n_classes=1, 
        bilinear=args.bilinear,
        use_checkpointing=args.use_checkpointing
    )
    model = model.to(memory_format=torch.channels_last)
    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
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
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                     'Try reducing the batch size or image scale.')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            img_scale=args.scale,
            patch_size=args.patch_size,
            amp=args.amp
        )
