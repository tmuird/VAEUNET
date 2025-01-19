import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import get_all_metrics, dice_score
from utils import metrics
import random
import os
import logging

# At start of script
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True  # Better performance on RTX 3060
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad()
def evaluate(model, dataloader, device, amp, max_samples=4):
    """
    Evaluation function with optimized memory usage for medical image segmentation
    Args:
        model: UNet model
        dataloader: Validation data loader
        device: Computing device (cuda/cpu)
        amp: Boolean for mixed precision
        max_samples: Number of sample images to return for visualization
    Returns:
        metrics_mean: Dict of averaged metrics
        samples: List of (image, prediction, ground_truth, global_idx) tuples
    """
    model.eval()
    
    # Reserve CUDA memory at start
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some VRAM free

    # Calculate total number of samples
    total_samples = len(dataloader.dataset)
    
    # Pre-select random indices if we're collecting samples
    selected_indices = []
    if max_samples > 0:
        selected_indices = random.sample(range(total_samples), min(max_samples, total_samples))
        selected_indices = set(selected_indices)  # Convert to set for O(1) lookup

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
    samples = [(None, None, None, None)] * max_samples if max_samples > 0 else []
    collected = 0

    # Use tqdm for progress tracking
    with tqdm(total=len(dataloader), desc='Validation', unit='batch') as pbar:
        for batch in dataloader:
            torch.cuda.empty_cache()
            
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                # Ensure consistent dtype across validation
                dtype = torch.float16 if amp else torch.float32
                image = batch['image'].to(device=device, dtype=dtype, non_blocking=True)
                mask_true = batch['mask'].to(device=device, dtype=dtype, non_blocking=True)

                # Compute prediction
                mask_pred = model(image)
                
                # Take only the main output if in training mode (ignoring deep supervision outputs)
                if isinstance(mask_pred, list):
                    mask_pred = mask_pred[0]
                
                mask_pred = torch.sigmoid(mask_pred)

                # Binary prediction for metrics
                mask_bin = (mask_pred > 0.5).float()

                # Compute metrics
                batch_metrics = get_all_metrics(mask_bin, mask_true)
                del mask_bin

                # Update metrics
                for metric in metrics_sum:
                    metrics_sum[metric] += batch_metrics[metric]

                # compute the Dice score
                metrics_sum['dice'] += metrics.dice_score(mask_pred, mask_true)

                # Sample collection with memory optimization
                if max_samples > 0:
                    batch_start_idx = num_val_batches * dataloader.batch_size
                    for i in range(len(image)):
                        global_idx = batch_start_idx + i
                        if global_idx in selected_indices:
                            with torch.cuda.amp.autocast(enabled=False):
                                # Ensure samples are in full precision
                                sample_idx = len([x for x in samples if x[0] is not None])
                                image_np = image[i].detach().cpu().float().numpy()
                                pred_np = mask_pred[i].detach().cpu().float().numpy()
                                true_np = mask_true[i].detach().cpu().float().numpy()
                                logging.info(f"Eval image stats - min: {image_np.min():.3f}, max: {image_np.max():.3f}, mean: {image_np.mean():.3f}")
                                samples[sample_idx] = (
                                    image_np,
                                    pred_np,
                                    true_np,
                                    global_idx
                                )

                # Memory cleanup
                del image, mask_true, mask_pred, batch_metrics
                torch.cuda.empty_cache()

            num_val_batches += 1
            pbar.update(1)

    # Calculate mean metrics
    metrics_mean = {metric: value / num_val_batches for metric, value in metrics_sum.items()}
    
    # Clean up samples list
    if max_samples > 0:
        samples = [s for s in samples if s[0] is not None]

    return metrics_mean, samples
