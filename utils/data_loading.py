from typing import Optional
import logging
import numpy as np
import torch
from PIL import Image
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_image(filename):
    """Load image/mask and force to RGB (3-channel) or L (1-channel) depending on usage."""
    # Force to RGB so we never get 4-channel RGBA or 1-channel grayscale
    # You can tweak this if you want to keep masks as L, etc.
    try:
        img = Image.open(filename)
        # If you want the image 3-channel:
        img = img.convert('RGB')
        return img
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        raise

def unique_mask_values(idx, mask_dir, mask_suffix):
    """Find unique values in the mask for index `idx`."""
    mask_files = list(mask_dir.glob(idx + mask_suffix + '.*'))
    if not mask_files:
        return []
    mask_file = mask_files[0]
    mask_pil = load_image(mask_file).convert('L')  # force single channel for mask
    mask = np.array(mask_pil, dtype=np.uint8)
    return np.unique(mask)

class BasicDataset(Dataset):
    """Base dataset that loads (image, mask) pairs at a chosen scale."""
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # Collect IDs from the images_dir
        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith('.')
        ]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, check that it is not empty')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            all_uniques = list(tqdm(
                p.imap(
                    partial(unique_mask_values,
                            mask_dir=self.mask_dir,
                            mask_suffix=self.mask_suffix),
                    self.ids
                ),
                total=len(self.ids)
            ))

        # Flatten + unique the mask values
        flattened = [u for sublist in all_uniques for u in sublist]  # flatten
        if flattened:
            self.mask_values = sorted(np.unique(flattened).tolist())
        else:
            self.mask_values = []
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask: bool):
        """Resize image or mask, handle channels and normalization."""
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        if newW < 1 or newH < 1:
            raise ValueError(f'Scale {scale} too small => resulted in {newW}x{newH}')

        # Resizing: NEAREST if mask, BICUBIC if image
        pil_img = pil_img.resize((newW, newH),
                                 resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.array(pil_img, dtype=np.uint8 if is_mask else np.float32)

        if is_mask:
            # Force shape = [H, W]. Convert multiple classes if mask_values is known.
            # If mask is single-channel, shape is (H, W). If your mask is indeed multi-class,
            # you'd do a loop to map values to class indices, but let's assume binary.
            if img.ndim == 3:
                # If it ever comes in as 3D, take a single channel or reduce it
                img = img[..., 0]  # or however you prefer
            # If you have pre-scanned mask_values, you can map them to indices if needed:
            if mask_values:
                # e.g. if you want to map each unique pixel to an integer label
                label_mask = np.zeros((newH, newW), dtype=np.int64)
                for i, val in enumerate(mask_values):
                    label_mask[img == val] = i
                return label_mask
            else:
                # Assume binary
                return (img > 0).astype(np.int64)

        else:
            # Image: force shape to [C, H, W]
            if img.ndim == 2:
                # Grayscale => expand dims => shape [1, H, W]
                img = img[None, ...]  
            else:
                # shape is [H, W, C], we want [C, H, W]
                img = img.transpose((2, 0, 1))  # to [C, H, W]
            # Normalize 0-255 => 0-1 if not already
            if img.max() > 1.0:
                img = img / 255.0
            return img

    def __getitem__(self, idx):
        """Load one (image, mask) pair, scaled."""
        name = self.ids[idx]
        mask_files = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_files  = list(self.images_dir.glob(name + '.*'))

        assert len(img_files) == 1, \
            f'Either no image or multiple images found for the ID {name}: {img_files}'
        assert len(mask_files) == 1, \
            f'Either no mask or multiple masks found for the ID {name}: {mask_files}'

        # Load them (force RGB for image, L for mask)
        mask_pil = load_image(mask_files[0]).convert('L')
        img_pil  = load_image(img_files[0]).convert('RGB')

        # Check same size
        assert img_pil.size == mask_pil.size, \
            f"Image and mask shapes differ for {name}: {img_pil.size} vs {mask_pil.size}"

        # Preprocess
        img_array  = self.preprocess(self.mask_values, img_pil,  self.scale, is_mask=False)
        mask_array = self.preprocess(self.mask_values, mask_pil, self.scale, is_mask=True)

        # Convert to torch Tensors
        img_tensor  = torch.as_tensor(img_array,  dtype=torch.float32).contiguous()
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.long).contiguous()

        return {
            'image': img_tensor,
            'mask':  mask_tensor
        }


class IDRIDDataset(Dataset):
    """Dataset that loads images, masks, then precomputes overlapping patches."""
    def __init__(self, base_dir: str, split: str = 'train', scale: float = 0.25, 
                 patch_size: Optional[int] = None, lesion_type: str = 'EX',
                 max_images: Optional[int] = None):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.base_dir = Path(base_dir)
        self.split = split

        self.images_dir = self.base_dir / 'imgs' / split
        self.masks_dir  = self.base_dir / 'masks' / split

        self.lesion_type = lesion_type
        self.class_dir   = lesion_type

        logging.info(f"Loading {split} dataset from: {self.images_dir}, {self.masks_dir / self.class_dir}")
        # Gather IDs
        self.ids = [
            splitext(file)[0]
            for file in listdir(self.images_dir)
            if file.endswith('.jpg')
        ]
        if max_images is not None:
            self.ids = self.ids[:max_images]

        # Check mask existence
        for img_id in self.ids:
            mask_path = self.masks_dir / self.class_dir / f"{img_id}_{self.lesion_type}.tif"
            if not mask_path.exists():
                logging.warning(f"No mask found for {img_id}")
            else:
                pass  # you can load & check if you want

        # Prepare containers
        self.patches = {}
        self.patch_indices = []
        self.precompute_all_patches()

        # Clear these after patch extraction
        self.full_images = None
        self.full_masks = None

        if split == 'train':
            self.transform = A.Compose([
                # SAFE TRANSFORMATIONS - Keep these
                A.HorizontalFlip(p=0.5),  # Valid since lesions can appear on either side
                A.VerticalFlip(p=0.5),    # Valid since lesions can appear top/bottom
                
                # CAREFUL WITH THESE - Modify parameters
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,   # Reduced from 0.2 - hard exudates are bright yellow
                    contrast_limit=0.1,     # Reduced from 0.2 - preserve lesion contrast
                    p=0.3                   # Reduced probability
                ),
                
                # ADD THESE - Specific for hard exudates
                A.CLAHE(                   # Enhance contrast locally - helps with exudate detection
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=0.5
                ),
                
                # BE CAREFUL WITH ROTATION - Modify parameters
                A.ShiftScaleRotate(
                    shift_limit=0.1,        # Reduced from 0.2 - avoid displacing lesions too much
                    scale_limit=0.1,        # Reduced from 0.2
                    rotate_limit=15,        # Reduced from 30 - small rotations only
                    p=0.3                   # Reduced probability
                ),
                
                # Remove HueSaturationValue as it might alter the characteristic 
                # yellow color of hard exudates
            ])

    def precompute_all_patches(self):
        """Load each image+mask, scale, then slice into patches with balanced sampling."""
        logging.info(f"Precomputing patches for {len(self.ids)} images in split={self.split} ...")
        
        positive_patches = []
        negative_patches = []

        for img_id in tqdm(self.ids, desc="Precomputing patches"):
            # Paths
            img_file  = self.images_dir / f"{img_id}.jpg"
            mask_file = self.masks_dir / self.class_dir / f"{img_id}_{self.lesion_type}.tif"

            # Load & convert
            img_pil  = load_image(img_file).convert('RGB')
            mask_pil = load_image(mask_file).convert('L')

            # Check same size
            if img_pil.size != mask_pil.size:
                logging.warning(f"Mismatch in size for {img_file} vs {mask_file}; skipping.")
                continue

            # Preprocess => numpy => float
            img_array  = self.preprocess(img_pil,  self.scale, is_mask=False)
            mask_array = self.preprocess(mask_pil, self.scale, is_mask=True)

            # Convert to torch
            img_tensor  = torch.as_tensor(img_array,  dtype=torch.float32)
            mask_tensor = torch.as_tensor(mask_array, dtype=torch.float32)

            # Ensure [C, H, W] for image, [1, H, W] for mask
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            if img_tensor.dim() == 2:
                # Rare if grayscale snuck through; ensure shape => [1,H,W]
                img_tensor = img_tensor.unsqueeze(0)

            _, h, w = img_tensor.shape

            # If no patch_size set, store the entire image as a single patch
            if self.patch_size is None:
                self.patches[img_id] = [{
                    'image': img_tensor,
                    'mask':  mask_tensor,
                    'coords': (0, 0),
                    'has_lesion': torch.any(mask_tensor > 0.5)
                }]
                self.patch_indices.append((img_id, 0))
                logging.info(f"{img_id}: 1 patch (full image) with shape {img_tensor.shape}.")
                continue

            if (h < self.patch_size) or (w < self.patch_size):
                # If the scaled image is smaller than patch_size, either skip or pad
                logging.warning(f"{img_id}: scaled to {h}x{w} < patch_size={self.patch_size}; skipping.")
                continue

            stride = self.patch_size // 2
            for y in range(0, h - self.patch_size + 1, stride):
                for x in range(0, w - self.patch_size + 1, stride):
                    img_patch = img_tensor[:, y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch = mask_tensor[:, y:y+self.patch_size, x:x+self.patch_size]
                    has_lesion = torch.any(mask_patch > 0.5)

                    patch_data = {
                        'image': img_patch.contiguous(),
                        'mask': mask_patch.contiguous(),
                        'coords': (y, x),
                        'img_id': img_id,
                        'has_lesion': has_lesion
                    }

                    if has_lesion:
                        positive_patches.append(patch_data)
                    else:
                        negative_patches.append(patch_data)

        # Balance patches
        num_positives = len(positive_patches)
        logging.info(f"Found {num_positives} positive patches and {len(negative_patches)} negative patches")

        # Randomly shuffle negative patches
        random.shuffle(negative_patches)
        
        # Take only as many negatives as we have positives
        negative_patches = negative_patches[:num_positives]
        
        # Combine and shuffle final patches
        all_patches = positive_patches + negative_patches
        random.shuffle(all_patches)

        # Store in class variables
        self.patches = {}
        self.patch_indices = []
        
        for idx, patch in enumerate(all_patches):
            img_id = patch['img_id']
            if img_id not in self.patches:
                self.patches[img_id] = []
            patch_idx = len(self.patches[img_id])
            self.patches[img_id].append(patch)
            self.patch_indices.append((img_id, patch_idx))

        logging.info(f"Final dataset has {len(self.patch_indices)} balanced patches "
                    f"({len(positive_patches)} positive, {len(negative_patches)} negative)")

    def preprocess(self, pil_img: Image.Image, scale: float, is_mask: bool):
        """Resize and convert to np array. For masks => single channel. For images => 3 channels."""
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        if newW < 1 or newH < 1:
            raise ValueError(f"Image {pil_img} scaled too small => {newW}x{newH}.")

        pil_img = pil_img.resize(
            (newW, newH),
            resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        arr = np.array(pil_img)
        # If it's a mask => shape [H,W], threshold => {0 or 1}, etc.
        if is_mask:
            if arr.ndim == 3:
                # If mask is accidentally 3D, take just 1 channel or reduce
                arr = arr[..., 0]
            return (arr > 0).astype(np.float32)
        else:
            # Image => shape => [H,W,3], scale to [0..1] float
            if arr.ndim == 2:
                # grayscale => expand
                arr = np.stack([arr]*3, axis=-1)
            arr = arr.astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # to [C,H,W]
            return arr

    def __getitem__(self, idx):
        """Return a single patch (image, mask)."""
        img_id, patch_idx = self.patch_indices[idx]
        patch = self.patches[img_id][patch_idx]

        return {
            'image': patch['image'],
            'mask':  patch['mask'],
            'img_id': f"{img_id}_y{patch['coords'][0]}_x{patch['coords'][1]}"
        }

    def __len__(self):
        return len(self.patch_indices)

    def shuffle_patches(self):
        random.shuffle(self.patch_indices)
