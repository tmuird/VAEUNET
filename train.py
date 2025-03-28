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
import cv2
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Optional
from utils import metrics
import wandb
from evaluate import evaluate
from unet import UNet
from unet.unet_resnet import UNetResNet
from utils.data_loading import IDRIDDataset
from utils.loss import CombinedLoss
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

# Add to imports
from utils.vae_utils import generate_predictions

def collate_patches(batch):
    """Custom collate function to handle patches.
    Improved to better handle full images of the same size.
    """
    # Check if all images are the same shape
    shapes = [x['image'].shape for x in batch]
    
    if len(set(tuple(shape) for shape in shapes)) == 1:
        # All shapes are the same, stack normally
        return {
            'image': torch.stack([x['image'] for x in batch]),
            'mask': torch.stack([x['mask'] for x in batch]),
            'img_id': [x['img_id'] for x in batch]
        }
    else:
        # Different shapes - don't stack, just return list
        return {
            'image': [x['image'] for x in batch],
            'mask': [x['mask'] for x in batch],
            'img_id': [x['img_id'] for x in batch]
        }

def multi_temp_training_step(model, images, true_masks, criterion, 
                        temps=[1.0, 3.0], weight=0.3):
    """Training step with multi-temperature sampling."""
    # Standard prediction
    outputs = model(images)
    if isinstance(outputs, tuple):
        pred = outputs[0]  
    else:
        pred = outputs
    standard_loss = criterion(pred, true_masks)
    
    # Multi-temperature prediction using shared utilities
    multi_temp_loss = 0
    for temp in temps:
        temp_pred = generate_predictions(
            model, images, temperature=temp, num_samples=3
        )
        multi_temp_loss += criterion(temp_pred, true_masks)
    
    multi_temp_loss /= len(temps)
    
    # Combined loss
    total_loss = (1-weight) * standard_loss + weight * multi_temp_loss
    return total_loss, {'standard_loss': standard_loss.item(), 'multi_temp_loss': multi_temp_loss.item()}

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        patch_size: Optional[int] = None,
        amp: bool = False,
        bilinear: bool = False,
        gradient_clipping: float = 1.0,
        max_images: Optional[int] = None,
        gradient_accumulation_steps: int = 2,
        early_stopping_patience: int = 10,
        lesion_type: str = 'EX',
        backbone: str = 'resnet34',
        pretrained: bool = True,
        beta: float = 0.1  # KL weight
):
    # Clear cache at start and set memory limits for better management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Reserve memory fraction to prevent OOM errors
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        logging.info(f"""
Initial GPU Status:
- GPU: {torch.cuda.get_device_name()}
- Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB
- Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB
""")

    # Initialize best_dice at the start
    best_dice = 0.0
    best_val_score = float('-inf')
    no_improvement_count = 0
    
    # Check if data directories exist
    data_dir = Path('./data')
    if not (data_dir / 'imgs' / 'train').exists():
        raise RuntimeError(f"Training images directory not found at {data_dir / 'imgs' / 'train'}")
    if not (data_dir / 'masks' / 'train').exists():
        raise RuntimeError(f"Training masks directory not found at {data_dir / 'masks' / 'train'}")

    logging.info(f'Loading datasets with patch size: {patch_size} and max images: {max_images}')
    try:
        # For full images (patch_size=None), we'll get consistent-sized patches 
        # through padding in the dataset, so no special handling needed here
        train_dataset = IDRIDDataset(base_dir='./data',
                                split='train',
                                scale=img_scale,
                                patch_size=patch_size,
                                lesion_type=lesion_type,
                                max_images=max_images)
        val_dataset = IDRIDDataset(base_dir='./data',
                            split='val',
                            scale=img_scale,
                            patch_size=patch_size,
                            lesion_type=lesion_type,
                            max_images=max_images)
    except ValueError as e:
        logging.error(f"Error creating datasets: {e}")
        logging.error(f"No valid data found for lesion type {lesion_type}. Please check your data directory.")
        return
        
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logging.error(f"Empty dataset found for lesion type {lesion_type}. Please check your data directory.")
        return
    
    logging.info(f'Dataset sizes:')
    logging.info(f'- Training: {len(train_dataset)} images')
    logging.info(f'- Validation: {len(val_dataset)} images')

    # Create data loaders - now we can use the same approach for both patch and full-image modes
    # since full images are padded to a consistent size
    loader_args = dict(
        batch_size=batch_size, 
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory_device='cuda',
        collate_fn=collate_patches  # Add this line to use the custom collate function
    )
    
    # No special handling needed for patch_size=None anymore
    
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
            lesion_type=lesion_type,
            backbone=backbone,
            pretrained=pretrained
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
        Backbone:        {backbone}
        Pretrained:      {pretrained}
    ''')

    # Use combined loss function with balanced weights
    criterion = CombinedLoss()
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
    global_step = 0

    # Move model to device and optimize memory format
    model = model.to(device=device, memory_format=torch.channels_last)
    
    # Quick shape debug
    for batch in train_loader:
        images = batch['image']
        masks = batch['mask']
        logging.info(f"Image shape: {images.shape if not isinstance(images, list) else [img.shape for img in images]}")
        logging.info(f"Mask shape: {masks.shape if not isinstance(masks, list) else [mask.shape for mask in masks]}")
        logging.info(f"Batch size: {len(batch['img_id'])}")
        break

    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logging.info(f"AMP enabled: {amp}")
        
    # Before training loop
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.preferred_linalg_library('cusolver')
            # Set matmul precision to medium for RTX 30xx series (better memory usage)
            torch.set_float32_matmul_precision('medium')
        except:
            logging.warning("Could not set preferred CUDA libraries - using defaults")

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # All tensors will have the same shape now, so no need for special handling
                images = batch['image'].to(device=device, memory_format=torch.channels_last, non_blocking=True)
                masks = batch['mask'].to(device=device, non_blocking=True)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # Get model outputs
                    seg_output, mu, logvar = model(images)
                    
                    # Compute reconstruction loss (dice + BCE)
                    recon_loss = criterion(seg_output, masks)
                    
                    # Compute KL divergence loss
                    kl_loss = 0.5 * torch.sum(
                        torch.exp(logvar) + mu**2 - 1.0 - logvar
                    ) / (mu.size(0) * mu.size(1))  # Normalize by batch size and latent dim
                    
                    # Combine losses
                    loss = recon_loss + beta * kl_loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                grad_scaler.scale(loss).backward()

                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * gradient_accumulation_steps
                global_step += 1

                # Safely log train loss to wandb (catch connection errors)
                try:
                    wandb.log({
                        'train/total_loss': loss.item(),
                        'train/kl_loss': kl_loss.item(),
                        'train/reconstruction_loss': recon_loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                except wandb.errors.Error as e:
                    logging.warning(f"Could not log to W&B (train losses). Error: {e}")

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

                # Calculate steps per epoch and current epoch progress
                steps_per_epoch = len(train_loader)
                current_epoch_step = batch_idx + 1  # +1 because batch_idx is 0-based
                
                # Validate at middle and end of epoch
                if current_epoch_step == steps_per_epoch // 2 or current_epoch_step == steps_per_epoch:
                    # Clear memory before validation
                    for var in ['images', 'masks', 'seg_output', 'loss']:
                        if var in locals():
                            exec(f'del {var}')
                    torch.cuda.empty_cache()
                    
                    validation_point = "mid" if current_epoch_step == steps_per_epoch // 2 else "end"
                    logging.info(f'Running {validation_point}-epoch validation (epoch {epoch}, step {current_epoch_step}/{steps_per_epoch})')
                    model.eval()
                    val_metrics, val_samples = evaluate(model, val_loader, device, amp, max_samples=4)
                    
                    try:
                        # Log
                        for i, (img_np, pred_np, true_np, _) in enumerate(val_samples):
                            # Prepare image for visualization
                            img_vis = img_np.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
                            
                            # Adjust brightness and ensure proper format for W&B media saving
                            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
                            img_vis = (img_vis * 255).clip(0, 255).astype('uint8')
                            
                            # Prepare masks for overlay view
                            pred_overlay = (pred_np[0] > 0.5).astype('uint8')
                            true_overlay = (true_np[0] > 0.5).astype('uint8')
                            
                            # Prepare masks for separate viewing (full range visualization)
                            pred_vis = pred_np[0]  # Get the first channel
                            pred_vis = (pred_vis - pred_vis.min()) / (pred_vis.max() - pred_vis.min() + 1e-8)  # Normalize
                            pred_vis = (pred_vis * 255).astype('uint8')  # Scale to 0-255
                            
                            # Ground truth for separate viewing
                            true_vis = true_np[0] * 255  # Scale to 0-255
                            
                            # Use global step in image names for proper tracking
                            img_name = f"step_{global_step}_sample_{i}"
                            
                            # Log images and metrics together to ensure proper step alignment
                            wandb.log({
                                # Overlay view
                                f"{img_name}_comparison": wandb.Image(
                                    img_vis,
                                    masks={
                                        "predictions": {
                                            "mask_data": pred_overlay,
                                            "class_labels": {1: "Prediction"}
                                        },
                                        "ground_truth": {
                                            "mask_data": true_overlay,
                                            "class_labels": {1: "Ground Truth"}
                                        }
                                    }
                                ),
                                # Separate images for VS Code viewing
                                f"{img_name}_image": wandb.Image(img_vis),
                                f"{img_name}_pred": wandb.Image(pred_vis),
                                f"{img_name}_true": wandb.Image(true_vis),
                                # Include metrics in the same log call
                                **{f'val/{k}': v for k, v in val_metrics.items()},
                                'learning_rate': optimizer.param_groups[0]['lr'],
                                'epoch': epoch,
                                'step': global_step
                            })
                            del img_vis, pred_vis, true_vis, pred_overlay, true_overlay
                    except wandb.errors.Error as e:
                        logging.warning(f"Could not log to W&B (validation). Error: {e}")

                    # Update scheduler based on validation metric
                    val_score = val_metrics['dice']
                    scheduler.step(val_score)
                    
                    # Save best model if we have an improvement
                    if val_score > best_val_score:
                        best_val_score = val_score
                        if save_checkpoint:
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            checkpoint_path = str(dir_checkpoint / f'best_model.pth')
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_val_score': best_val_score,
                                'amp_scaler': grad_scaler.state_dict(),
                                'global_step': global_step
                            }, checkpoint_path)
                            logging.info(f'New best model saved! (Dice: {val_score:.4f})')
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= early_stopping_patience:
                            logging.info(f'Early stopping triggered after {epoch} epochs')
                            return

                    # Cleanup validation data
                    del val_metrics, val_samples
                    torch.cuda.empty_cache()
                    model.train()

                # More aggressive memory cleanup after every few steps
                if batch_idx % 5 == 0:  # Every 5 batches
                    torch.cuda.empty_cache()

        # End of epoch validation and cleanup
        # Safely clean up training variables
        for var in ['images', 'masks', 'seg_output', 'loss']:
            if var in locals():
                exec(f'del {var}')
        torch.cuda.empty_cache()
        
    # Final cleanup 
    torch.cuda.empty_cache()
    return


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--patch-size', '-p', type=lambda x: None if x.lower() == 'none' else int(x),
                        default=None, help='Size of patches to extract (use None to disable)')
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--use-checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing to reduce memory usage')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lesion-type', type=str, default='EX', help='Lesion type')
    parser.add_argument('--model-type', type=str, default='resnet', choices=['basic', 'resnet'],
                    help='Model type: basic (original UNet) or resnet (UNet with ResNet34 encoder)')
    parser.add_argument('--skip', dest='use_skip', action='store_true',
                    help='Enable skip connections in the UNet model (default)')
    parser.add_argument('--no-skip', dest='use_skip', action='store_false',
                    help='Disable skip connections and attention in the UNet model')
    parser.add_argument('--attention', dest='use_attention', action='store_true',
                    help='Enable attention mechanism when skip connections are enabled (default)')
    parser.add_argument('--no-attention', dest='use_attention', action='store_false',
                    help='Disable attention mechanism but keep skip connections enabled')
    parser.set_defaults(use_attention=True, use_skip=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Model initialization based on type
    if args.model_type == 'resnet':
        model = UNetResNet(
            n_channels=3,
            n_classes=1,
            backbone='resnet34',
            pretrained=True,
            use_attention=args.use_attention,
            use_skip=args.use_skip
        )
    else:
        model = UNet(
            n_channels=3,
            n_classes=1,
            bilinear=args.bilinear
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
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience,
            lesion_type=args.lesion_type,
            backbone='resnet34',
            pretrained=True,
            beta=0.1
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
            amp=args.amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience,
            lesion_type=args.lesion_type,
            backbone='resnet34',
            pretrained=True,
            beta=0.1
        )
