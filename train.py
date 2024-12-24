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
    """Custom collate function to handle patches."""
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
    train_dataset = IDRIDDataset(base_dir='./data',
                                 split='train',
                                 scale=img_scale,
                                 patch_size=patch_size,
                                 lesion_type='EX',
                                 max_images=max_images)
    val_dataset = IDRIDDataset(base_dir='./data',
                               split='val',
                               scale=img_scale,
                               patch_size=patch_size,
                               lesion_type='EX',
                               max_images=max_images)
    
    logging.info(f'Dataset sizes:')
    logging.info(f'- Training: {len(train_dataset)} images')
    logging.info(f'- Validation: {len(val_dataset)} images')

    # Create data loaders
    loader_args = dict(
        batch_size=batch_size, 
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory_device='cuda'
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args
    )
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        **loader_args
    )

    # Safely initialize wandb with fallback to offline
    try:
        experiment = wandb.init(
            project='IDRID-UNET',
            resume='allow',
            anonymous='must'
        )
    except wandb.errors.CommError as e:
        logging.warning(f"W&B connection error: {e}. Falling back to offline mode.")
        experiment = wandb.init(
            project='IDRID-UNET',
            resume='allow',
            anonymous='must',
            mode='offline'
        )

    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_checkpoint=save_checkpoint,
            img_scale=img_scale,
            amp=amp,
            patch_size=patch_size,
            classes=1,
            lesion_type='EX'
        )
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
        patience=5,
        min_lr=1e-6,
        verbose=True,
        factor=0.5
    )
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0

    # Move model to device
    model = model.to(device=device)
    
    # Quick shape debug
    for batch in train_loader:
        images = batch['image']
        masks = batch['mask']
        logging.info(f"Image shape: {images.shape}")
        logging.info(f"Mask shape: {masks.shape}")
        break

    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logging.info(f"AMP enabled: {amp}")
        
    # Before training loop
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.preferred_linalg_library('cusolver')

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred, masks.float())
                    else:
                        loss = criterion(masks_pred, masks)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                optimizer.zero_grad(set_to_none=True)

                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += loss.item()
                global_step += 1

                # Safely log train loss to wandb (catch connection errors)
                try:
                    wandb.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                except wandb.errors.Error as e:
                    logging.warning(f"Could not log to W&B (train loss). Error: {e}")

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

                # Periodically evaluate
                division_step = len(train_dataset) // (batch_size * 2)  # Validate 2x per epoch
                if division_step > 0 and batch_idx > 0 and batch_idx % division_step == 0:
                    # Clear memory before validation
                    del images, masks, masks_pred, loss
                    torch.cuda.empty_cache()
                    
                    # Add explicit model mode switching
                    model.eval()
                    
                    val_metrics, val_samples = evaluate(
                        model, 
                        val_loader, 
                        device, 
                        amp,
                        max_samples=4,
                    )

                    # After validation, ensure we switch back to training mode
                    model.train()

                    try:
                        # Log sample images
                        for i, (img_np, pred_np, true_np) in enumerate(val_samples):
                            img_vis = (img_np.transpose(1,2,0) * 255).clip(0,255).astype('uint8')
                            pred_vis = (pred_np[0] * 255).clip(0,255).astype('uint8')
                            true_vis = (true_np[0] * 255).clip(0,255).astype('uint8')
                            wandb.log({
                                f"val_image_{i}": wandb.Image(
                                    img_vis,
                                    masks={
                                        "prediction": {"mask_data": pred_vis, "class_labels": {1: "Lesion"}},
                                        "ground_truth": {"mask_data": true_vis, "class_labels": {1: "Lesion"}}
                                    }
                                )
                            })
                            
                            # Clear sample data after logging
                            del img_vis, pred_vis, true_vis
                            
                        # Log metrics
                        wandb.log({
                            **{f'val/{k}': v for k, v in val_metrics.items()},
                            'epoch': epoch,
                            'step': global_step,
                            'learning_rate': optimizer.param_groups[0]['lr']
                        })
                        
                        # Clear samples after logging
                        del val_samples
                        torch.cuda.empty_cache()
                        
                    except wandb.errors.Error as e:
                        logging.warning(f"Could not log to W&B (validation). Error: {e}")

                    # Update LR based on dice
                    scheduler.step(val_metrics['dice'])
                    
                    # Clear validation metrics
                    del val_metrics
                    torch.cuda.empty_cache()

                # Add memory monitoring (optional, for debugging)
                if torch.cuda.is_available() and batch_idx % 100 == 0:
                    logging.info(f"""
                    Memory Status:
                    - Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB
                    - Reserved:  {torch.cuda.memory_reserved()/1e9:.2f}GB
                    """)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--patch-size', '-p', type=lambda x: None if x.lower() == 'none' else int(x),
                        default=None, help='Size of patches to extract (use None to disable)')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--use-checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing to reduce memory usage')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
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
        # If you previously had some extra keys, remove them:
        state_dict.pop('mask_values', None)
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
        # If you want to attempt re-training with gradient checkpointing or smaller batch size, do so here.
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
