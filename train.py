import argparse
import logging
import random  # Added for seeding

import torch

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

import wandb
from evaluate import evaluate
from unet import UNet
from unet.unet_resnet import UNetResNet
from utils.data_loading import IDRIDDataset
from utils.loss import CombinedLoss, MASegmentationLoss, KLAnnealer, kl_with_free_bits
import numpy as np
from torchvision.transforms import Normalize
from utils.vae_utils import generate_predictions, calculate_latent_stats
import datetime

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

# Create normalizer with mean 0 and std 1 for each channel
normalize = Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Ensure deterministic behavior for cuDNN
    # NOTE: Setting deterministic=True and benchmark=False ensures strict reproducibility
    # but can negatively impact performance and may slightly alter convergence.
    # Setting benchmark=True and deterministic=False prioritizes performance.
    # torch.backends.cudnn.deterministic = True # Disabled for performance
    torch.backends.cudnn.benchmark = True  # Enabled for performance
    logging.info(f"Random seed set to {seed}")

def worker_init_fn(worker_id):
    """Sets the seed for DataLoader workers."""
    # Get the main seed and add worker_id to ensure different seeds per worker
    # Using torch.initial_seed() ensures that the seed is derived from the main process seed
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def get_checkpoint_path(
    lesion_type: str,
    model_type: str,
    use_attention: bool,
    img_scale: float,
    patch_size: Optional[int],
    beta: float,
    free_bits: float,
    kl_anneal_epochs: int,
    latent_injection: str = None,
    learning_rate: float = None,
    seed: Optional[int] = None  # Added seed parameter
) -> Path:
    """
    Create a structured checkpoint directory path based on model parameters.
    
    Returns:
        Path: Directory path for saving model checkpoints
    """
    # Format the patch size for the folder name
    patch_str = f"patch{patch_size}" if patch_size is not None else "full_img"
    
    # Format scale with precision appropriate for the value
    if img_scale == int(img_scale):
        scale_str = f"scale{int(img_scale)}"
    else:
        scale_str = f"scale{img_scale:.1f}"
    
    # Create folder components
    attention_str = "attn" if use_attention else "no_attn"
    
    # Create KL parameters string
    kl_str = f"beta{beta:.4f}" if beta > 0 else "noKL"
    if free_bits > 0:
        kl_str += f"_fb{free_bits:.4f}"
    if kl_anneal_epochs > 0:
        kl_str += f"_anneal{kl_anneal_epochs}"
    
    # Add latent injection flag if used
    latent_str = f"_latent{latent_injection}" if latent_injection and latent_injection != "none" else ""
    
    # Add learning rate if specified
    lr_str = f"_lr{learning_rate}" if learning_rate is not None else ""
    
    # Add seed if specified
    seed_str = f"_seed{seed}" if seed is not None else ""
    
    # Combine all components for the directory name
    dir_name = f"{lesion_type}_{model_type}_{attention_str}_{scale_str}_{patch_str}_{kl_str}{latent_str}{lr_str}{seed_str}"
    
    return dir_checkpoint / dir_name

def collate_patches(batch):
    """Custom collate function to handle patches.
    Improved to better handle full images of the same size.
    """
    # Check if all images are the same shape
    shapes = [x['image'].shape for x in batch]

    if len(set(tuple(shape) for shape in shapes)) == 1:
        # All shapes are the same, stack normally
        images = torch.stack([x['image'] for x in batch])
        masks = torch.stack([x['mask'] for x in batch])
        # Apply normalization to standardize images
        images = normalize(images)
        return {
            'image': images,
            'mask': masks,
            'img_id': [x['img_id'] for x in batch]
        }
    else:
        # Different shapes - don't stack, just return list
        return {
            'image': [normalize(x['image']) for x in batch],  # Apply normalization to each image
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
        beta: float = 0.001,  # KL weight
        free_bits: float = 5e-1,  # Added free bits parameter
        kl_anneal_epochs: int = 20,  # Added KL annealing epochs
        seed: Optional[int] = None  # Added seed parameter
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
        collate_fn=collate_patches,  # Add this line to use the custom collate function
        worker_init_fn=worker_init_fn  # Add worker init function for reproducibility
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
            pretrained=pretrained,
            seed=seed  # Log seed
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
        Seed:            {seed}
    ''')

    # Use specialized loss for Microaneurysms
    if lesion_type == 'MA':
        criterion = MASegmentationLoss(class_weight=0.9)
        logging.info(f"Using specialized Microaneurysm loss with class weight 0.9")
    else:
        criterion = CombinedLoss()
    
    # Initialize KL annealer
    kl_annealer = KLAnnealer(kl_start=0.0, kl_end=beta, warmup_epochs=kl_anneal_epochs)
    
    # Use a more gentle learning rate scheduler for microaneurysms
    if lesion_type == 'MA':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # Use ReduceLROnPlateau with more patience and smaller factor
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            patience=8,  # More patience
            min_lr=1e-5,
            verbose=True,
            factor=0.7  # More gentle reduction (0.7 instead of 0.5)
        )
        logging.info(f"Using gentle learning rate schedule for Microaneurysms")
    else:
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
    # Settings for determinism/benchmarking are now handled in set_seed
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

        # Get current KL weight (beta) from annealer
        current_beta = kl_annealer.get_weight(epoch)
        logging.info(f"Epoch {epoch}: KL weight (beta): {current_beta:.6f}")

        # For latent space monitoring
        epoch_mu = []
        epoch_logvar = []

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # All tensors will have the same shape now, so no need for special handling
                images = batch['image'].to(device=device, memory_format=torch.channels_last, non_blocking=True)
                masks = batch['mask'].to(device=device, non_blocking=True)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # Get model outputs
                    seg_output, mu, logvar = model(images)
                    
                    # Store latent variables for monitoring
                    if batch_idx % 5 == 0:  # Sample every 5 batches to avoid memory issues
                        epoch_mu.append(mu.detach().cpu())
                        epoch_logvar.append(logvar.detach().cpu())
                    
                    # Compute reconstruction loss (dice + BCE)
                    recon_loss = criterion(seg_output, masks)
                    
                    # Compute KL divergence loss with free bits
                    kl_loss = kl_with_free_bits(mu, logvar, free_bits=free_bits)
                    
                    # Use current annealed beta value
                    loss = recon_loss + current_beta * kl_loss
                    
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
                        'train/kl_weight': current_beta,
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
                            
                            # Denormalize image (from mean=0, std=1 to 0-1 range)
                            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
                            img_vis = np.clip(img_vis, 0, 1)  # Ensure values are in [0,1]
                            img_vis = (img_vis * 255).clip(0, 255).astype(np.uint8)
                            
                            # Prepare masks for overlay view
                            pred_overlay = (pred_np[0] > 0.5).astype('uint8')
                            true_overlay = (true_np[0] > 0.5).astype('uint8')
                            
                            # Prepare masks for separate viewing (full range visualization)
                            pred_vis = pred_np[0]  # Get the first channel
                            
                            # Check for NaN/Inf values that might occur with standardized inputs
                            pred_vis = np.nan_to_num(pred_vis, nan=0.0, posinf=1.0, neginf=0.0)
                            
                            # Normalize between 0 and 1
                            min_val = pred_vis.min()
                            max_val = pred_vis.max()
                            if max_val > min_val:  # Avoid division by zero
                                pred_vis = (pred_vis - min_val) / (max_val - min_val)
                            else:
                                pred_vis = np.zeros_like(pred_vis)
                                
                            # Now safely convert to uint8
                            pred_vis = (pred_vis * 255).clip(0, 255).astype(np.uint8)  # Scale to 0-255
                            
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
                                # f"{img_name}_image": wandb.Image(img_vis),
                                # f"{img_name}_pred": wandb.Image(pred_vis),
                                # f"{img_name}_true": wandb.Image(true_vis),
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
                            # Create structured checkpoint directory based on parameters
                            model_type = 'resnet' if hasattr(model, 'backbone') else 'basic'
                            use_attention = hasattr(model, 'use_attention') and model.use_attention
                            latent_injection = getattr(model, 'latent_injection', None)
                            
                            # Generate structured checkpoint path
                            checkpoint_dir = get_checkpoint_path(
                                lesion_type=lesion_type,
                                model_type=model_type,
                                use_attention=use_attention,
                                img_scale=img_scale,
                                patch_size=patch_size,
                                beta=beta,
                                free_bits=free_bits,
                                kl_anneal_epochs=kl_anneal_epochs,
                                latent_injection=latent_injection,
                                learning_rate=learning_rate,
                                seed=seed  # Pass seed to checkpoint path function
                            )
                            
                            # Create directory if it doesn't exist
                            checkpoint_dir.mkdir(parents=True, exist_ok=True)
                            logging.info(f"Checkpoint directory: {checkpoint_dir}")
                            # Generate timestamp for filename
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                            
                            # Create checkpoint filename with timestamp, epoch and performance
                            model_filename = f"model_{timestamp}_ep{epoch}_dice{val_score:.4f}.pth"
                            checkpoint_path = checkpoint_dir / model_filename
                            
                            # Also save as best_model.pth in same directory for compatibility
                            best_model_path = checkpoint_dir / "best_model.pth"
                            
                            # Create checkpoint data with comprehensive metadata
                            checkpoint_data = {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_val_score': best_val_score,
                                'amp_scaler': grad_scaler.state_dict(),
                                'global_step': global_step,
                                # Save model parameters for easier identification later
                                'params': {
                                    'lesion_type': lesion_type,
                                    'model_type': model_type,
                                    'use_attention': use_attention,
                                    'img_scale': img_scale,
                                    'patch_size': patch_size,
                                    'beta': beta,
                                    'free_bits': free_bits,
                                    'kl_anneal_epochs': kl_anneal_epochs,
                                    'latent_injection': latent_injection,
                                    'seed': seed  # Save seed in checkpoint
                                }
                            }
                            
                            # Save both the timestamped file and best_model.pth
                            torch.save(checkpoint_data, checkpoint_path)
                            torch.save(checkpoint_data, best_model_path)
                            
                            logging.info(f'New best model with dice score {val_score:.4f} saved')
                            logging.info(f'Also saved as {best_model_path}')
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= early_stopping_patience:
                            logging.info(f'Early stopping triggered after {epoch} epochs')
                            try:
                                wandb.log({"early_stopped": True, "final_epoch": epoch})
                                wandb.finish() 
                            except Exception as e:
                                logging.warning(f"Error finalizing wandb run: {e}")
                            return

                    # Cleanup validation data
                    del val_metrics, val_samples
                    torch.cuda.empty_cache()
                    model.train()

                # More aggressive memory cleanup after every few steps
                if batch_idx % 5 == 0:  # Every 5 batches
                    torch.cuda.empty_cache()

        # End of epoch: analyze latent space
        if epoch_mu:  # If we collected some latent variables
            try:
                # Concatenate all collected latent variables
                all_mu = torch.cat(epoch_mu, dim=0)
                all_logvar = torch.cat(epoch_logvar, dim=0)
                
                # Calculate latent stats
                latent_stats = calculate_latent_stats(all_mu, all_logvar)
                
                # Log to wandb
                wandb.log({
                    'latent/active_dims': latent_stats['active_dims'],
                    'latent/activity_ratio': latent_stats['activity_ratio'],
                    'latent/total_kl': latent_stats['total_kl'],
                    'latent/mean_mu_abs': latent_stats['mean_mu_abs'],
                    'latent/mean_var': latent_stats['mean_var'],
                    'epoch': epoch
                })
                
                # Log to console
                logging.info(f"Latent space stats: Active dims {latent_stats['active_dims']}/{latent_stats['total_dims']} ({latent_stats['activity_ratio']:.2f}), Total KL: {latent_stats['total_kl']:.4f}")
            except Exception as e:
                logging.warning(f"Error calculating latent stats: {e}")
                
            # Clean up
            del epoch_mu, epoch_logvar, all_mu, all_logvar
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
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=6, help='Batch size')
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
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience')
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
    parser.add_argument('--kl-anneal-epochs', type=int, default=20, 
                      help='Number of epochs for KL annealing')
    parser.add_argument('--free-bits', type=float, default=1e-3,
                      help='Free bits parameter to prevent posterior collapse')
    parser.add_argument('--latent-injection', type=str, default='all', 
                        choices=['all', 'first', 'last', 'bottleneck', 'inject_no_bottleneck', 'none'], # Updated choices
                        help='Latent space injection strategy')
    parser.add_argument('--beta', type=float, default=0.001,
                        help='KL weight for the loss function')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')  # Added seed argument
    parser.set_defaults(use_attention=True, use_skip=True)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Set seed as early as possible
    set_seed(args.seed)

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
            use_skip=args.use_skip,
            latent_injection=args.latent_injection  # Add the new parameter
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
            beta=args.beta,
            free_bits=args.free_bits,
            kl_anneal_epochs=args.kl_anneal_epochs,
            seed=args.seed  # Pass seed to train_model
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
            amp=args.amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience,
            lesion_type=args.lesion_type,
            backbone='resnet34',
            pretrained=True,
            beta=args.beta,
            free_bits=args.free_bits,
            kl_anneal_epochs=args.kl_anneal_epochs,
            seed=args.seed  # Pass seed to train_model
        )
