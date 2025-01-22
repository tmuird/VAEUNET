import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from unet.unet_resnet import UNetResNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                use_patches=True,
                patch_size=None,
                overlap=100):
    net.eval()
    
    # Preprocess the full image
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    # Get dimensions
    _, c, h, w = img.shape
    
    # If not using patches or patch_size is None, predict on full image
    if not use_patches or patch_size is None:
        with torch.no_grad():
            mask_pred, _, _ = net(img)  # Unpack only the segmentation output
            mask_pred = torch.sigmoid(mask_pred)
            
            # Resize to original image size if needed
            if scale_factor != 1:
                mask_pred = F.interpolate(mask_pred, (full_img.size[1], full_img.size[0]), mode='bilinear')
            
            # Apply threshold
            mask = (mask_pred > out_threshold).cpu().numpy()
            return mask[0, 0]
    
    # Initialize the full mask and weight map for patch-based prediction
    full_mask = torch.zeros((1, 1, h, w), device=device)
    weight_map = torch.zeros((1, 1, h, w), device=device)
    
    # Create a weight matrix for blending
    y, x = torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size))
    weight = (1 - x.abs()) * (1 - y.abs())
    weight = weight.to(device)[None, None, :, :]
    
    # Calculate steps with overlap
    stride = patch_size - overlap
    
    # Calculate number of patches needed
    n_patches_h = max(1, (h - patch_size + stride) // stride)
    n_patches_w = max(1, (w - patch_size + stride) // stride)
    
    logging.info(f'Processing image of size {h}x{w} with {n_patches_h}x{n_patches_w} patches')
    
    # Process each patch
    with torch.no_grad():
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                top = min(i * stride, h - patch_size)
                left = min(j * stride, w - patch_size)
                
                # Extract patch
                patch = img[:, :, top:top+patch_size, left:left+patch_size]
                
                if patch.shape[-2:] != (patch_size, patch_size):
                    # If patch is smaller than patch_size, pad it
                    temp_patch = torch.zeros((1, c, patch_size, patch_size), device=device)
                    temp_patch[:, :, :patch.shape[-2], :patch.shape[-1]] = patch
                    patch = temp_patch
                
                # Predict on patch
                with torch.no_grad():
                    patch_pred, _, _ = net(patch)  # Unpack only the segmentation output
                    patch_pred = torch.sigmoid(patch_pred)
                
                # Apply weight for blending
                patch_pred = patch_pred * weight
                
                # Add to full mask
                if patch.shape[-2:] != (patch_size, patch_size):
                    # If patch was padded, only take the valid part
                    valid_h, valid_w = patch.shape[-2:]
                    full_mask[:, :, top:top+valid_h, left:left+valid_w] += patch_pred[:, :, :valid_h, :valid_w]
                    weight_map[:, :, top:top+valid_h, left:left+valid_w] += weight[:, :, :valid_h, :valid_w]
                else:
                    full_mask[:, :, top:top+patch_size, left:left+patch_size] += patch_pred
                    weight_map[:, :, top:top+patch_size, left:left+patch_size] += weight
    
    # Average overlapping regions
    full_mask = full_mask / weight_map.clamp(min=1e-8)
    
    # Resize to original image size if needed
    if scale_factor != 1:
        full_mask = F.interpolate(full_mask, (full_img.size[1], full_img.size[0]), mode='bilinear')
    
    # Apply threshold
    mask = (full_mask > out_threshold).cpu().numpy()
    
    return mask[0, 0]


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--no-patches', action='store_true',
                        help='Disable patch-based prediction')
    parser.add_argument('--patch-size', '-p', type=int, default=None,
                        help='Size of patches to use for prediction (None for full image)')
    parser.add_argument('--overlap', type=int, default=100,
                        help='Overlap size between patches')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1,
                        help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNetResNet(n_channels=3, n_classes=args.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    checkpoint = torch.load(args.model, map_location=device)
    mask_values = checkpoint.pop('mask_values', [0, 1])
    net.load_state_dict(checkpoint['model_state_dict'])

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        
        # Log image size
        logging.info(f'Image size: {img.size}')
        
        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=args.scale,
            out_threshold=args.mask_threshold,
            device=device,
            use_patches=not args.no_patches,
            patch_size=args.patch_size,
            overlap=args.overlap
        )

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
