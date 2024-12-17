import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_scores = torch.zeros(5, device=device)  # One score for each class
    
    with torch.autocast(device_type='cuda', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image = batch['image'].to(device=device, dtype=torch.float32)
            mask_true = batch['mask'].to(device=device, dtype=torch.float32)
            
            # Predict
            mask_pred = net(image)
            mask_pred = torch.sigmoid(mask_pred)
            
            # Calculate Dice score for each class
            for i in range(5):
                dice_scores[i] += dice_coeff(
                    (mask_pred[:, i] > 0.5).float(),
                    mask_true[:, i],
                    reduce_batch_first=False
                )

    net.train()
    
    # Calculate final scores
    dice_scores = dice_scores.cpu() / max(num_val_batches, 1)
    
    # Return scores in dictionary format
    return {
        'mean_dice': dice_scores.mean().item(),
        'MA_dice': dice_scores[0].item(),
        'HE_dice': dice_scores[1].item(),
        'EX_dice': dice_scores[2].item(),
        'SE_dice': dice_scores[3].item(),
        'OD_dice': dice_scores[4].item()
    }