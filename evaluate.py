

import torch
from tqdm import tqdm
import logging
from utils.dice_score import dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0.0
    
    with torch.autocast(device_type='cuda', enabled=amp):
        for batch in dataloader:
            image = batch['image'].to(device=device, dtype=torch.float32)
            mask_true = batch['mask'].to(device=device, dtype=torch.float32)
            
            mask_pred = net(image)
            mask_pred = torch.sigmoid(mask_pred)

            dice_score += dice_coeff((mask_pred > 0.5).float(), mask_true, reduce_batch_first=False)

    net.train()
    mean_dice = dice_score / max(num_val_batches, 1)
    return {
        'mean_dice': mean_dice.item()
    }
