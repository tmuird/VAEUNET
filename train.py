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



def collate_patches(batch):
    """Custom collate function to handle patches"""
    return {
        'image': torch.stack([x['image'] for x in batch]),
        'mask': torch.stack([x['mask'] for x in batch]),
        'img_id': [x['img_id'] for x in batch]
    }


def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        patch_size: Optional[int] = None,
        amp: bool = True,
        bilinear: bool = False,
        gradient_clipping: float = 1.0,
        max_images: Optional[int] = None,
):
    # Clear cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info(f"""
Initial GPU Status:
- GPU: {torch.cuda.get_device_name()}
- Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB
- Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB
""")

    # Initialize best_dice at the start
    best_dice = 0.0
    
    # Check if data directories exist
    data_dir = Path('./data')
    if not (data_dir / 'imgs' / 'train').exists():
        raise RuntimeError(f"Training images directory not found at {data_dir / 'imgs' / 'train'}")
    if not (data_dir / 'masks' / 'train').exists():
        raise RuntimeError(f"Training masks directory not found at {data_dir / 'masks' / 'train'}")

    logging.info(f'Loading datasets with patch size: {patch_size} and max images: {max_images}')
    # Load datasets with logging
    train_dataset = IDRIDDataset(base_dir='./data', split='train', 
                                scale=img_scale, patch_size=patch_size,
                                lesion_type='EX',
                                max_images=max_images )
    val_dataset = IDRIDDataset(base_dir='./data', split='val', 
                              scale=img_scale, patch_size=patch_size,
                              lesion_type='EX',
                              max_images=max_images )
    
    logging.info(f'Dataset sizes:')
    logging.info(f'- Training: {len(train_dataset)} images')
    logging.info(f'- Validation: {len(val_dataset)} images')

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args
    )
    val_loader = DataLoader(val_dataset, 
                           shuffle=False, 
                           **loader_args,
                           )

    # Initialize loggingrime
    experiment = wandb.init(project='IDRID-UNET',resume='allow', anonymous='must'
    )
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
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Patch size:      {patch_size}
        Max images:      {max_images}
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

    # At start of training
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logging.info(f"AMP enabled: {amp}")
        
    logging.info(f"Initial VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # After creating dataloaders
    logging.info(f"VRAM after dataloaders: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Before training starts
    logging.info(f"VRAM before training: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # After model to device
    model = model.to(device)
    logging.info(f"VRAM after model to device: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # After optimizer creation
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    logging.info(f"VRAM after optimizer: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # First batch
    for batch in train_loader:
        images = batch['image'].to(device)
        logging.info(f"Image batch shape: {images.shape}")
        logging.info(f"VRAM after first batch: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        break

    # At start of training
    optimizer.zero_grad(set_to_none=True)  # Initial gradient clear

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        current_image=None
        patches_processed=0
        model.train()
        epoch_loss = 0
        image_patches = {}  # Track patches per image
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Track memory before batch

                # Move to GPU and track memory
                images = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)

                # Forward pass with AMP
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        # Adjust Focal Loss parameters for severe imbalance
                        # focal = metrics.focal_loss(
                        #     masks_pred, 
                        #     masks.float(),
                        #     alpha=0.95,    # Increase alpha (weight for positive class) significantly
                        #     gamma=4.0      # Increase gamma to focus more on hard examples
                        # )
                        # dice = metrics.dice_loss(
                        #     torch.sigmoid(masks_pred), 
                        #     masks, 
                        #     multiclass=False
                        # )
                        
                        # Adjust loss weights to emphasize Focal Loss
                        # loss = 0.7 * focal + 0.3 * dice  # Give more weight to Focal Loss
                        loss=criterion(masks_pred,masks)
                    else:
                        loss = criterion(masks_pred, masks)

                mem_after_forward = torch.cuda.memory_allocated()/1e9

                # Backward pass with scaler
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                mem_after_backward = torch.cuda.memory_allocated()/1e9

                # if batch_idx % 1 == 0:  # Log every batch
                #     logging.info(f"""
# Batch {batch_idx} Memory (GB):
# - Before batch: {mem_before:.2f}
# - After transfer: {mem_after_transfer:.2f}
# - After forward: {mem_after_forward:.2f}
# - After backward: {mem_after_backward:.2f}
# Image shape: {images.shape}
# """)
                # Log images every N batches (e.g. every 100 batches)
                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        train_images = batch['image'].to(device=device, dtype=torch.float32)
                        train_masks = batch['mask'].to(device=device, dtype=torch.float32)

                        model.eval()
                        train_masks_pred = model(train_images)
                        train_masks_pred = torch.sigmoid(train_masks_pred)

                        # Select random index
                        idx = random.randint(0, len(train_images) - 1)
                        
                        # Get images and masks
                        img = train_images[idx].cpu().numpy()
                        pred_mask = train_masks_pred[idx].cpu().numpy()
                        true_mask = train_masks[idx].cpu().numpy()

                        # Prepare image for wandb (HWC format, 0-255 range)
                        img = img.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
                        img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        
                        # Prepare masks for wandb (HW format, 0-255 range)
                        true_mask = np.clip(true_mask[0] * 255, 0, 255).astype(np.uint8)  # Remove channel dim and scale
                        pred_mask = np.clip(pred_mask[0] * 255, 0, 255).astype(np.uint8)  # Remove channel dim and scale

                        experiment.log({
                                    'train_images': wandb.Image(
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
                # Clean up
                torch.cuda.empty_cache()
              
                del images, masks, masks_pred
                pbar.update(1)
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = len(train_dataset) // (batch_size * 2)  # Validate twice per epoch
                if division_step > 0 and batch_idx % division_step == 0:
                    val_metrics, val_samples = evaluate(
                        model, 
                        val_loader, 
                        device, 
                        amp,
                        max_samples=4,
                    )
                    # Also log images to wandb
                    for i, (img_np, pred_np, true_np) in enumerate(val_samples):
                        # Convert them to a suitable format for wandb.Image
                        # Typically, original image is [C,H,W], so transpose to [H,W,C]
                        img_vis = (img_np.transpose(1,2,0) * 255).clip(0,255).astype('uint8')

                        # For mask, pick channel 0 if shape is [1,H,W], or if multiple classes, do likewise
                        pred_vis = (pred_np[0] * 255).clip(0,255).astype('uint8')
                        true_vis = (true_np[0] * 255).clip(0,255).astype('uint8')

                        # Log them as a wandb.Image with mask overlays
                        wandb.log({
                            f"val_image_{i}": wandb.Image(
                                img_vis,
                                masks={
                                    "prediction": {
                                        "mask_data": pred_vis,
                                        "class_labels": {1: "Lesion"}
                                    },
                                    "ground_truth": {
                                        "mask_data": true_vis,
                                        "class_labels": {1: "Lesion"}
                                    }
                                }
                            )
                        })
                    # Update scheduler with dice score
                    scheduler.step(val_metrics['dice'])
                    
                    # Log validation metrics to wandb
                    experiment.log({
                        **{f'val/{k}': v for k, v in val_metrics.items()},
                        'epoch': epoch,
                        'step': global_step,
                        'learning_rate': optimizer.param_groups[0]['lr']
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
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of ep8chs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate')
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
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process')
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
            max_images=args.max_images,
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
            max_images=args.max_images,
            amp=args.amp
        )
