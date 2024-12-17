import torch
from torch import Tensor
import logging


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Add shape logging
    logging.debug(f'Dice calculation - Input shape: {input.shape}, Target shape: {target.shape}')

    # Ensure same shape
    if input.shape != target.shape:
        raise ValueError(f'Shape mismatch in dice_coeff: input {input.shape} vs target {target.shape}')

    if input.dim() > 2 or target.dim() > 2:
        if reduce_batch_first:
            input = input.contiguous().view(input.shape[0], -1)
            target = target.contiguous().view(target.shape[0], -1)
        else:
            input = input.contiguous().view(-1)
            target = target.contiguous().view(-1)

    intersection = (input * target).sum()
    denominator = input.sum() + target.sum()

    if denominator.item() == 0:
        return torch.tensor(1.0).to(input.device)

    return (2.0 * intersection + epsilon) / (denominator + epsilon)


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
