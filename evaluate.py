import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import get_all_metrics

def evaluate(model, dataloader, device, amp):
    """Evaluation function for the model"""
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

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            mask_pred = model(image)

            # convert to probabilities
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

            # compute metrics for this batch
            batch_metrics = get_all_metrics(mask_pred,mask_true)
                
                # update metrics sums
            for metric in metrics_sum:
                metrics_sum[metric] += batch_metrics[metric]

    model.train()
    # Calculate mean metrics
    metrics_mean  = {metric: value / num_val_batches for metric, value in metrics_sum.items()}

   
    return metrics_mean
