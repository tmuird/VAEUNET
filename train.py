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

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import IDRIDDataset
from utils.dice_score import dice_loss

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
        epochs: int = 100,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        img_scale: float = 0.25,
        amp: bool = True,
        gradient_clipping: float = 1.0,
):
    # Check if data directories exist
    data_dir = Path('./data')
    if not (data_dir / 'imgs' / 'train').exists():
        raise RuntimeError(f"Training images directory not found at {data_dir / 'imgs' / 'train'}")
    if not (data_dir / 'masks' / 'train').exists():
        raise RuntimeError(f"Training masks directory not found at {data_dir / 'masks' / 'train'}")

    # Create datasets
    try:
        train_dataset = IDRIDDataset(base_dir='./data', split='train', scale=img_scale)
        val_dataset = IDRIDDataset(base_dir='./data', split='val', scale=img_scale)
    except Exception as e:
        logging.error(f"Error creating datasets: {e}")
        raise

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    
    # BCE loss for multi-label segmentation
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    # Move model to device and set to appropriate dtype
    model = model.to(device=device)
    
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)

                with torch.autocast(device_type='cuda', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        torch.sigmoid(masks_pred), 
                        true_masks,
                        multiclass=True
                    )

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
                        # Ensure val_score is a dictionary before accessing
                        if isinstance(val_score, dict):
                            scheduler.step(val_score['mean_dice'])
                        else:
                            scheduler.step(val_score)  # Assume it's a single value if not a dict
                        logging.info('Validation Dice score: {}'.format(val_score))
                        
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score['mean_dice'],
                            'MA Dice': val_score['MA_dice'],
                            'HE Dice': val_score['HE_dice'],
                            'EX Dice': val_score['EX_dice'],
                            'SE Dice': val_score['SE_dice'],
                            'OD Dice': val_score['OD_dice'],
                            'images': wandb.Image(
                                images[0].cpu(),
                                caption='Input Image'
                            ),
                            'masks': {
                                'true_MA': wandb.Image(
                                    true_masks[0, 0].float().cpu(),
                                    caption='MA Ground Truth'
                                ),
                                'true_HE': wandb.Image(
                                    true_masks[0, 1].float().cpu(),
                                    caption='HE Ground Truth'
                                ),
                                'true_EX': wandb.Image(
                                    true_masks[0, 2].float().cpu(),
                                    caption='EX Ground Truth'
                                ),
                                'true_SE': wandb.Image(
                                    true_masks[0, 3].float().cpu(),
                                    caption='SE Ground Truth'
                                ),
                                'true_OD': wandb.Image(
                                    true_masks[0, 4].float().cpu(),
                                    caption='OD Ground Truth'
                                ),
                                'pred_MA': wandb.Image(
                                    torch.sigmoid(masks_pred[0, 0]).float().cpu().detach(),
                                    caption='MA Prediction'
                                ),
                                'pred_HE': wandb.Image(
                                    torch.sigmoid(masks_pred[0, 1]).float().cpu().detach(),
                                    caption='HE Prediction'
                                ),
                                'pred_EX': wandb.Image(
                                    torch.sigmoid(masks_pred[0, 2]).float().cpu().detach(),
                                    caption='EX Prediction'
                                ),
                                'pred_SE': wandb.Image(
                                    torch.sigmoid(masks_pred[0, 3]).float().cpu().detach(),
                                    caption='SE Prediction'
                                ),
                                'pred_OD': wandb.Image(
                                    torch.sigmoid(masks_pred[0, 4]).float().cpu().detach(),
                                    caption='OD Prediction'
                                ),
                            },
                            'overlays': {
                                'MA': wandb.Image(
                                    images[0].cpu(),
                                    masks={
                                        "predictions": {
                                            "mask_data": torch.sigmoid(masks_pred[0, 0]).float().cpu().detach().numpy(),
                                            "class_labels": {1: "MA"}
                                        },
                                        "ground_truth": {
                                            "mask_data": true_masks[0, 0].float().cpu().numpy(),
                                            "class_labels": {1: "MA"}
                                        }
                                    },
                                    caption='MA Overlay'
                                ),
                                'EX': wandb.Image(
                                    images[0].cpu(),
                                    masks={
                                        "predictions": {
                                            "mask_data": torch.sigmoid(masks_pred[0, 2]).float().cpu().detach().numpy(),
                                            "class_labels": {1: "EX"}
                                        },
                                        "ground_truth": {
                                            "mask_data": true_masks[0, 2].float().cpu().numpy(),
                                            "class_labels": {1: "EX"}
                                        }
                                    },
                                    caption='EX Overlay'
                                )
                            },
                            'step': global_step,
                            'epoch': epoch
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=6, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=5, bilinear=False)
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
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                     'Try reducing the batch size or image scale.')
        raise
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
