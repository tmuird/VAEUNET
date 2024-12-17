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
            logging.info(f'Mask values after preprocessing: min={mask.min()}, max={mask.max()}, unique={np.unique(mask)}')
            return (mask > 0).astype(np.float32)

    def extract_patch(self, img: torch.Tensor, mask: torch.Tensor) -> tuple:
        """Extract a random patch from image and mask."""
        if self.patch_size is None:
            logging.debug('Patch extraction disabled, using full images')
            return img, mask
            
        logging.debug(f'Extracting patch of size {self.patch_size}')
        _, h, w = img.shape
        max_x = w - self.patch_size
        max_y = h - self.patch_size
        
        if max_x > 0 and max_y > 0:
            x = torch.randint(0, max_x, (1,)).item()
            y = torch.randint(0, max_y, (1,)).item()
            
            img = img[:, y:y+self.patch_size, x:x+self.patch_size]
            mask = mask[:, y:y+self.patch_size, x:x+self.patch_size]
            logging.debug(f'Extracted patch at position ({x}, {y})')
        else:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            img = F.pad(img, (0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_w, 0, pad_h))
            logging.debug(f'Padded image to match patch size')
            
        return img, mask

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
            logging.info(f'Original mask values for {img_id}: min={mask.min()}, max={mask.max()}, unique={np.unique(mask)}')
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
            img, mask = self.extract_patch(img, mask)
            
        # Log final shapes
        logging.debug(f'Final shapes - Image: {img.shape}, Mask: {mask.shape}')

        return {
            'image': img.contiguous(),
            'mask': mask.contiguous(),
            'img_id': img_id
        }
