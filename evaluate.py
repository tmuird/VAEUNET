import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from utils.dice_score import dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    
    with torch.autocast(device_type='cuda', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image = batch['image'].to(device=device, dtype=torch.float32)
            mask_true = batch['mask'].to(device=device, dtype=torch.float32)
            
            # Ensure mask has correct dimensions [B, C, H, W]
            if mask_true.dim() > 4:
                mask_true = mask_true.squeeze()
                if mask_true.dim() == 3:
                    mask_true = mask_true.unsqueeze(1)
            
            # Log shapes
            logging.info(f'Shapes - Image: {image.shape}, True mask: {mask_true.shape}')
            
            # Predict
            mask_pred = net(image)
            
            # Ensure prediction matches target shape
            if mask_pred.shape != mask_true.shape:
                logging.warning(f'Shape mismatch - Pred: {mask_pred.shape}, True: {mask_true.shape}')
                mask_pred = mask_pred.squeeze(1) if mask_pred.shape[1] == 1 else mask_pred
            
            # Apply sigmoid
            mask_pred = torch.sigmoid(mask_pred)
            
            # Calculate Dice score
            dice_score += dice_coeff(
                (mask_pred > 0.5).float(),
                mask_true,
                reduce_batch_first=False
            )

    net.train()
    
    # Calculate final score
    final_dice = dice_score / max(num_val_batches, 1)
    
    # Return score in dictionary format for consistency
    return {
        'mean_dice': final_dice.item(),
        'EX_dice': final_dice.item()  # Only EX score since that's all we're predicting
    }