import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import get_all_metrics
import random

def evaluate(model, dataloader, device, amp, max_samples=4):
    """
    Evaluation function for the model that returns:
      - Average metrics (dice, iou, etc.)
      - A small list of sample predictions for wandb logging.
    """
    model.eval()
    num_val_batches = len(dataloader)
    metrics_sum = {
        'dice': 0,
        'iou': 0,
        'precision': 0,
        'recall': 0,
        'specificity': 0,
        'f1': 0,
        'accuracy': 0
    }
    num_val_batches = 0

    # We'll store a few example (image, pred, truth) pairs
    samples = []

    # Track how many samples we've collected
    collected = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch_idx, batch in enumerate(dataloader):
            image, mask_true = batch['image'], batch['mask']
            
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            logits = model(image)
            mask_pred = torch.sigmoid(logits)
            mask_bin = (mask_pred > 0.5).float()

            # compute metrics for this batch
            batch_metrics = get_all_metrics(mask_bin, mask_true)

            for metric in metrics_sum:
                metrics_sum[metric] += batch_metrics[metric]

            # If we still need sample images, collect some from this batch
            if collected < max_samples:
                # We'll pick a random index from this batch
                idx = random.randint(0, len(image) - 1)
                
                # Convert to CPU numpy for wandb logging
                img_np  = image[idx].detach().cpu().numpy()
                pred_np = mask_pred[idx].detach().cpu().numpy()
                true_np = mask_true[idx].detach().cpu().numpy()

                # Add to samples list
                samples.append((img_np, pred_np, true_np))
                collected += 1

            num_val_batches += 1

    model.train()
    
    # Average metrics only over batches actually processed
    metrics_mean = {metric: value / num_val_batches for metric, value in metrics_sum.items()}
    
    return metrics_mean, samples
