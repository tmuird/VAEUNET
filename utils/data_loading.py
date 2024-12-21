import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional
import random
import os


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class IDRIDDataset(Dataset):
    def __init__(self, base_dir: str, split: str = 'train', scale: float = 0.25, patch_size: Optional[int] = None, lesion_type: str = 'EX'):
        self.scale = scale
        self.patch_size = patch_size
        
        # Setup directories
        self.images_dir = Path(base_dir) / 'imgs' / split
        self.masks_dir = Path(base_dir) / 'masks' / split
        
        # Define the lesion type
        self.lesion_type = lesion_type
        self.class_dir = lesion_type
        
        # Add debug logging
        logging.info(f'Loading {split} dataset from:')
        logging.info(f'- Images: {self.images_dir}')
        logging.info(f'- Masks: {self.masks_dir / self.class_dir}')
        
        # Get image files
        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if file.endswith('.jpg')]
        
        # Verify mask files exist
        for img_id in self.ids:
            mask_file = self.masks_dir / self.class_dir / f"{img_id}_{self.lesion_type}.tif"
            if not mask_file.exists():
                logging.warning(f'No mask found for {img_id}')
            else:
                # Check mask content
                mask = np.array(load_image(mask_file))
                if mask.max() == 0:
                    logging.warning(f'Mask for {img_id} is all zeros')
                else:
                    logging.info(f'Found valid mask for {img_id} with values in range [{mask.min()}, {mask.max()}]')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, scale, is_mask=False):
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
        else:
            w, h = img.size
            
        newW, newH = int(scale * w), int(scale * h)
        
        if not is_mask:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.resize((newW, newH))
            img = np.array(img).transpose((2, 0, 1))
            return img / 255.0
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
            img = img.resize((newW, newH))
            mask = np.array(img)
            # Debug mask values
            #logging.info(f'Mask values after preprocessing: min={mask.min()}, max={mask.max()}, unique={np.unique(mask)}')
            return (mask > 0).astype(np.float32)

    def get_random_patch(self, image, mask):
        """Extract random patch from image and mask, with higher probability around lesions"""
        _, h, w = image.shape
        
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(f"Image size ({h}x{w}) is smaller than patch size ({self.patch_size})")

        # Find lesion coordinates
        lesion_coords = torch.nonzero(mask[0] > 0.5)  # [N, 2] tensor of (y, x) coordinates
        
        if len(lesion_coords) > 0 and random.random() < 0.8:  # 80% chance to sample from lesion areas
            # Randomly select a lesion pixel
            idx = random.randint(0, len(lesion_coords) - 1)
            center_y, center_x = lesion_coords[idx]
            
            # Add some randomness to the center point
            center_y = center_y + random.randint(-self.patch_size//4, self.patch_size//4)
            center_x = center_x + random.randint(-self.patch_size//4, self.patch_size//4)
            
            # Ensure the patch fits within the image
            center_y = torch.clamp(center_y, self.patch_size//2, h - self.patch_size//2)
            center_x = torch.clamp(center_x, self.patch_size//2, w - self.patch_size//2)
            
            # Calculate patch coordinates
            y = center_y - self.patch_size//2
            x = center_x - self.patch_size//2
        else:
            # Random sampling for negative examples (20% of the time)
            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)
        
        # Extract patches
        image_patch = image[:, y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[:, y:y+self.patch_size, x:x+self.patch_size]
        
        # Verify patch contains data
        if image_patch.shape != (3, self.patch_size, self.patch_size):
            logging.error(f"Invalid patch shape: {image_patch.shape}")
            raise ValueError(f"Invalid patch shape: {image_patch.shape}")
        
        # Log patch statistics
        lesion_ratio = (mask_patch > 0.5).float().mean().item()
        if lesion_ratio > 0:
            logging.debug(f"Extracted patch with {lesion_ratio:.1%} lesion pixels")
        
        return image_patch, mask_patch

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Load image
        img_file = self.images_dir / f"{img_id}.jpg"
        img = load_image(img_file)
        
        # Load mask with value checking
        mask_file = self.masks_dir / self.class_dir / f"{img_id}_{self.lesion_type}.tif"
        if mask_file.exists():
            mask = load_image(mask_file)
            mask = np.array(mask)
            # Debug original mask values
            #logging.info(f'Original mask values for {img_id}: min={mask.min()}, max={mask.max()}, unique={np.unique(mask)}')
            if mask.ndim > 2:
                mask = mask[..., 0]
        else:
            w, h = img.size
            mask = np.zeros((h, w), dtype=np.float32)

        # Preprocess
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        # Convert to tensors
        img = torch.as_tensor(img.copy()).float()
        mask = torch.as_tensor(mask.copy()).float()
        
        # Add channel dimension to mask if needed
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        # Only apply patching if patch_size is set
        if self.patch_size is not None:
            img, mask = self.get_random_patch(img, mask)
            
        # Log final shapes
        logging.debug(f'Final shapes - Image: {img.shape}, Mask: {mask.shape}')

        return {
            'image': img.contiguous(),
            'mask': mask.contiguous(),
            'img_id': img_id
        }