import logging
import os

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
from tqdm import tqdm

from utils.metrics import get_all_metrics

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Create normalizer with mean 0 and std 1 for each channel 
normalize = Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])


@torch.inference_mode()
def evaluate(model, dataloader, device, amp, max_samples=None):
    # Set evaluation mode
    model.eval()

    num_val_batches = len(dataloader)
    metrics_sum = {}
    samples = []

    if max_samples is not None and max_samples < num_val_batches:
        num_val_batches = max_samples

    pbar = tqdm(total=num_val_batches, desc='Validation round', unit='batch', leave=False)

    for i, batch in enumerate(dataloader):
        if i >= num_val_batches:
            break

        if isinstance(batch, dict):
            images = batch['image']
            masks = batch['mask']
        else:
            images, masks = batch

        if isinstance(images, list):
            pbar.update(1)
            continue

        # Move to device and normalize if not already done
        images = images.to(device=device, memory_format=torch.channels_last, non_blocking=True)
        masks = masks.to(device=device, non_blocking=True)

        # Apply normalization if not already applied by the dataloader
        if not isinstance(batch, dict) or 'normalized' not in batch or not batch['normalized']:
            images = normalize(images)

        # Inference
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            # Get model outputs
            outputs = model(images)

            # Handle different output formats
            if isinstance(outputs, tuple):
                mask_pred = outputs[0]  # VAE model returns (seg_output, mu, logvar)
            else:
                mask_pred = outputs

            if mask_pred.shape != masks.shape:
                logging.info(f"Resizing predictions from {mask_pred.shape} to {masks.shape}")
                mask_pred = F.interpolate(
                    mask_pred,
                    size=masks.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Compute metrics
            batch_metrics = get_all_metrics(mask_pred, masks)

            # Store some samples for visualization
            if len(samples) < 4:
                mask_pred_sigmoid = torch.sigmoid(mask_pred)

                for j in range(min(2, images.shape[0])):
                    img_np = images[j].cpu().numpy()
                    pred_np = mask_pred_sigmoid[j].detach().cpu().numpy()
                    mask_np = masks[j].cpu().numpy()

                    # Add to samples list
                    samples.append((img_np, pred_np, mask_np, batch_metrics))

        # Add batch metrics to sum
        for key, value in batch_metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0) + value

        pbar.update(1)

    pbar.close()

    # Compute average metrics
    metrics_avg = {key: value / num_val_batches for key, value in metrics_sum.items()}

    return metrics_avg, samples
