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
import cv2

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


class IDRIDDataset(Dataset):
    """Dataset that loads images, masks, then precomputes overlapping patches."""
    def __init__(self, base_dir: str, split: str = 'train', scale: float = 0.25, 
                 patch_size: Optional[int] = None, lesion_type: str = 'EX',
                 max_images: Optional[int] = None, skip_border_check: bool = False):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = None  # Initialize transform as None for all splits
        self.skip_border_check = skip_border_check  # New parameter to skip border checks

        self.images_dir = self.base_dir / 'imgs' / split
        self.masks_dir  = self.base_dir / 'masks' / split

        # Calculate stride
        self.stride = self.patch_size // 2 if self.patch_size is not None else None

        # Simple patches directory structure
        self.patches_dir = self.base_dir / 'patches' / split / lesion_type
        # Clear any existing patches
        if self.patches_dir.exists():
            import shutil
            shutil.rmtree(self.patches_dir)
        self.patches_dir.mkdir(parents=True, exist_ok=True)

        self.lesion_type = lesion_type
        self.class_dir = lesion_type

        logging.info(f"Loading {split} dataset from: {self.images_dir}, {self.masks_dir / self.class_dir}")
        # Gather IDs
        self.ids = [
            splitext(file)[0]
            for file in listdir(self.images_dir)
            if file.endswith('.jpg')
        ]
        if max_images is not None:
            self.ids = self.ids[:max_images]

        # Check mask existence and filter out images without masks
        self.ids = [
            img_id for img_id in self.ids
            if (self.masks_dir / self.lesion_type / f"{img_id}_{self.lesion_type}.tif").exists()
        ]
        
        if not self.ids:
            raise RuntimeError(f'No valid image-mask pairs found in {self.images_dir} and {self.masks_dir}')

        logging.info(f"Found {len(self.ids)} valid image-mask pairs")

        # Store patch metadata instead of actual patches
        self.patch_metadata = {}
        self.patch_indices = []
        self.precompute_all_patches()

        # Clear these after patch extraction
        self.full_images = None
        self.full_masks = None

        # Initialize transform for all splits
        if split == 'train':
            self.transform = A.Compose([
                # Geometric Transformations - Safe for medical images
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Advanced Color Augmentations for DR
                A.OneOf([
                    # Multi-scale CLAHE - helps with varying lesion contrasts
                    A.CLAHE(
                        clip_limit=(1.5, 4.0),
                        tile_grid_size=(8, 8),
                        p=1.0
                    ),
                    # Gamma correction - helps with varying illumination
                    A.RandomGamma(
                        gamma_limit=(80, 120),
                        p=1.0
                    ),
                ], p=0.5),
                
                # Careful brightness/contrast adjustments
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=1.0
                    ),
                    # Specific for blood vessel enhancement
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.0,  # No hue changes to preserve lesion colors
                        p=1.0
                    ),
                ], p=0.3),
                
                # Using Affine instead of ShiftScaleRotate as recommended
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.0625, 0.0625),
                    rotate=(-15, 15),
                    mode=cv2.BORDER_CONSTANT,
                    cval=0,
                    p=0.3
                ),
                
                # Gaussian noise with correct parameters
                A.GaussNoise(
                    per_channel=True,
                    p=0.2
                ),
                
                # Blur for motion artifacts
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.2),
                
                # Grid distortion with correct parameters
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.2
                ),
            ])

    def is_valid_patch(self, patch: torch.Tensor, threshold: float = 0.1) -> bool:
        """Check if patch contains too much black border.
        
        Args:
            patch: Tensor of shape [C, H, W] for image or [1, H, W] for mask
            threshold: maximum allowed proportion of black pixels (default: 0.1)
            
        Returns:
            bool: True if patch is valid (contains enough non-border pixels)
        """
        # Skip the check if explicitly requested (for full images)
        if self.skip_border_check:
            return True
            
        if patch.dim() == 3:
            # For RGB patches, consider a pixel black if mean across channels is very dark
            is_black = (patch.mean(dim=0) < 0.1)  # Adjust threshold as needed
            black_ratio = is_black.float().mean()
            return black_ratio.item() <= threshold
        return True

    def precompute_all_patches(self):
        """Load each image+mask, scale, then slice into patches and save to disk one at a time."""
        logging.info(f"Precomputing patches for {len(self.ids)} images in split={self.split} ...")
        
        positive_count = 0
        negative_paths = []  # Store only paths, not actual patches
        patch_index = []

        for img_id in tqdm(self.ids, desc="Processing images and counting patches"):
            # Paths
            img_file  = self.images_dir / f"{img_id}.jpg"
            mask_file = self.masks_dir / self.lesion_type / f"{img_id}_{self.lesion_type}.tif"

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
                img_tensor = img_tensor.unsqueeze(0)

            _, h, w = img_tensor.shape

            # If no patch_size set, store the entire image as a single patch
            if self.patch_size is None:
                # Always include full images when patch_size is None, regardless of borders
                patch_path = self.patches_dir / f"{img_id}_full"
                has_lesion = torch.any(mask_tensor > 0.5)
                torch.save({
                    'image': img_tensor,
                    'mask': mask_tensor,
                    'coords': (0, 0),
                    'has_lesion': has_lesion
                }, patch_path)
                
                metadata = {
                    'path': patch_path,
                    'coords': (0, 0),
                    'has_lesion': has_lesion.item(),
                    'img_id': img_id
                }
                self.patch_metadata[img_id] = [metadata]
                if has_lesion:
                    positive_count += 1
                patch_index.append((img_id, str(patch_path), has_lesion))
                logging.info(f"{img_id}: 1 patch (full image) with shape {img_tensor.shape}.")
                continue

            if (h < self.patch_size) or (w < self.patch_size):
                logging.warning(f"{img_id}: scaled to {h}x{w} < patch_size={self.patch_size}; skipping.")
                continue

            stride = self.patch_size // 2
            patch_count = 0
            for y in range(0, h - self.patch_size + 1, stride):
                for x in range(0, w - self.patch_size + 1, stride):
                    img_patch = img_tensor[:, y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch = mask_tensor[:, y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Skip patches with too much black border
                    if not self.is_valid_patch(img_patch):
                        continue
                        
                    has_lesion = torch.any(mask_patch > 0.5)

                    patch_path = self.patches_dir / f"{img_id}_{patch_count}"
                    torch.save({
                        'image': img_patch.contiguous(),
                        'mask': mask_patch.contiguous(),
                        'coords': (y, x),
                        'has_lesion': has_lesion
                    }, patch_path)
                    
                    if has_lesion:
                        positive_count += 1
                        patch_index.append((img_id, str(patch_path), True))
                    else:
                        negative_paths.append((img_id, str(patch_path)))

                    patch_count += 1

        # Clear tensors to free memory
        del img_tensor, mask_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logging.info(f"Found {positive_count} positive patches and {len(negative_paths)} negative patches")

        if self.patch_size is None:
            # For full images, use all patches directly
            self.patch_indices = patch_index
            logging.info(f"Saved {len(self.patch_indices)} full images to {self.patches_dir}")
        else:
            # For patches, balance positive and negative samples for training only
            if self.split == 'train':
                random.shuffle(negative_paths)
                selected_negative_paths = negative_paths[:positive_count]
                
                # Add selected negative patches to index
                for img_id, path in selected_negative_paths:
                    patch_index.append((img_id, path, False))

                # Remove unselected negative patches to save disk space
                for img_id, path in negative_paths[positive_count:]:
                    try:
                        os.remove(path)
                    except OSError as e:
                        logging.warning(f"Error removing {path}: {e}")

                # Save final patch index
                self.patch_indices = patch_index
                logging.info(f"Saved {len(self.patch_indices)} balanced patches to {self.patches_dir}")
            else:
                # For validation/test, only use positive patches
                self.patch_indices = [(img_id, path, True) for img_id, path, has_lesion in patch_index if has_lesion]
                
                # Remove all negative patches to save disk space
                for img_id, path in negative_paths:
                    try:
                        os.remove(path)
                    except OSError as e:
                        logging.warning(f"Error removing {path}: {e}")
                
                logging.info(f"Saved {len(self.patch_indices)} positive patches for validation to {self.patches_dir}")

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
        """Return a single patch (image, mask) loaded from disk. If patch_size is None, returns full image."""
        if self.patch_size is None:
            img_id = self.patch_indices[idx][0]
            patch_data = torch.load(self.patches_dir / f"{img_id}_full")
        else:
            img_id, patch_path, _ = self.patch_indices[idx]
            patch_data = torch.load(patch_path)
        
        # Get image and mask from patch data
        image = patch_data['image']
        mask = patch_data['mask']
        
        sample = {
            'image': image,
            'mask': mask,
            'img_id': img_id
        }

        if self.transform is not None:
            # Convert to numpy for albumentations (expecting HWC format)
            image_np = sample['image'].permute(1, 2, 0).numpy()
            mask_np = sample['mask'].permute(1, 2, 0).numpy()
            
            transformed = self.transform(image=image_np, mask=mask_np)
            
            # Convert back to torch tensors (CHW format)
            sample['image'] = torch.from_numpy(transformed['image']).permute(2, 0, 1)
            sample['mask'] = torch.from_numpy(transformed['mask']).permute(2, 0, 1)

        return sample

    def __len__(self):
        return len(self.patch_indices)
