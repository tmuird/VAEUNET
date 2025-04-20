import logging
import os
import random
from os import listdir
from os.path import splitext
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    """Load image/mask and force to RGB (3-channel) or L (1-channel) depending on usage."""
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
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = None
        self.skip_border_check = skip_border_check
        self.is_full_image = patch_size is None

        self.images_dir = self.base_dir / 'imgs' / split
        self.masks_dir = self.base_dir / 'masks' / split

        # Gather IDs first to determine max dimensions if needed
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
            if (self.masks_dir / lesion_type / f"{img_id}_{lesion_type}.tif").exists()
        ]

        if not self.ids:
            raise RuntimeError(f'No valid image-mask pairs found in {self.images_dir} and {self.masks_dir}')

        logging.info(f"Found {len(self.ids)} valid image-mask pairs")

        # If in full image mode, determine the maximum dimensions
        if self.is_full_image:
            # Find max dimensions to use as patch size
            logging.info("Full image mode: Finding maximum dimensions of all images...")
            max_width, max_height = self.find_max_dimensions()

            # Set patch size to the maximum dimension (making square patches)
            self.patch_size = max(max_width, max_height)
            logging.info(f"Using patch size of {self.patch_size} for full images")
        else:
            self.patch_size = patch_size

        # Calculate stride - for full images, we use the full patch size
        self.stride = self.patch_size // 2 if not self.is_full_image else self.patch_size

        # Simple patches directory structure
        self.patches_dir = self.base_dir / 'patches' / split / lesion_type
        # Clear any existing patches
        if self.patches_dir.exists():
            import shutil
            shutil.rmtree(self.patches_dir)
        self.patches_dir.mkdir(parents=True, exist_ok=True)

        self.lesion_type = lesion_type
        self.class_dir = lesion_type

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
                # Geometric Transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

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

                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=1.0
                    ),

                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.0,
                        p=1.0
                    ),
                ], p=0.3),

                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.0625, 0.0625),
                    rotate=(-15, 15),
                    mode=cv2.BORDER_CONSTANT,
                    cval=0,
                    p=0.3
                ),

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

    def find_max_dimensions(self):
        """Find the maximum dimensions of all images in the dataset."""
        max_width, max_height = 0, 0
        circle_diameters = []

        for img_id in tqdm(self.ids, desc="Finding fundus circle dimensions"):
            img_file = self.images_dir / f"{img_id}.jpg"
            try:
                with Image.open(img_file) as img:
                    # Convert to numpy for processing
                    img_np = np.array(img)
                    # Detect the circular fundus region
                    diameter = self.detect_fundus_diameter(img_np, return_center=False)
                    if diameter is not None:  # Check for None to avoid errors
                        circle_diameters.append(float(diameter))  # Convert to float explicitly

                        # Scale the diameter
                        scaled_diameter = int(diameter * self.scale)
                        # The maximum dimension is the diameter of the circle
                        max_width = max(max_width, scaled_diameter)
                        max_height = max(max_height, scaled_diameter)
            except Exception as e:
                logging.warning(f"Couldn't process {img_file}: {e}")

                if "multiply" in str(e):
                    logging.error(f"Debug multiplication error: {e}, value type: {type(diameter)}")

        if circle_diameters:

            circle_diameters = [float(d) for d in circle_diameters if isinstance(d, (int, float))]
            typical_diameter = int(np.percentile(circle_diameters, 95) * self.scale)
            logging.info(f"Typical fundus diameter (95th percentile): {typical_diameter}")
            max_width = max_height = typical_diameter
        else:

            max_width = max_height = 694
            logging.warning("No valid fundus diameters detected, using fallback size")

        logging.info(f"Using dimensions: {max_width}x{max_height} for full image crops")
        return max_width, max_height

    def detect_fundus_diameter(self, image, return_center=True):
        """Detect the circular fundus region and return its diameter and center if requested. """
        try:
            # Convert to grayscale if RGB
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            if gray.dtype != np.uint8:
                # Scale to 0-255 range if float
                if gray.dtype == np.float32 or gray.dtype == np.float64:
                    gray = (gray * 255).astype(np.uint8)
                else:
                    gray = gray.astype(np.uint8)

            gray = cv2.medianBlur(gray, 5)

            # Threshold to separate fundus from background
            _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

            thresh = thresh.astype(np.uint8)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest contour (the fundus circle)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # Find the minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)

                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    # Fallback to the enclosing circle's center
                    center_x, center_y = int(x), int(y)

                diameter = float(radius * 2)  # Ensure diameter is a float

                # Return based on requested format
                if return_center:
                    return diameter, (center_x, center_y)
                else:
                    return diameter
            else:

                h, w = gray.shape[:2]
                diameter = float(min(h, w))

                if return_center:
                    return diameter, (w // 2, h // 2)
                else:
                    return diameter
        except Exception as e:
            logging.error(f"Error in detect_fundus_diameter: {e}")
            # Return None so caller can handle the error
            if return_center:
                return None, (None, None)
            else:
                return None

    def is_valid_patch(self, patch: torch.Tensor, threshold: float = 0.1) -> bool:
        """Check if patch contains too much black border. """

        if self.skip_border_check or self.is_full_image:
            return True

        if self.split == 'test':
            threshold = 0.5

        if patch.dim() == 3:
            is_black = (patch.mean(dim=0) < 0.1)
            black_ratio = is_black.float().mean()
            return black_ratio.item() <= threshold
        return True

    def precompute_all_patches(self):
        """Load each image+mask, scale, then slice into patches and save to disk one at a time."""
        logging.info(f"Precomputing patches for {len(self.ids)} images in split={self.split} ...")

        positive_count = 0
        negative_paths = []
        patch_index = []

        for img_id in tqdm(self.ids, desc="Processing images and extracting patches"):
            # Free memory proactively between images
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            img_file = self.images_dir / f"{img_id}.jpg"
            mask_file = self.masks_dir / self.lesion_type / f"{img_id}_{self.lesion_type}.tif"

            img_pil = load_image(img_file).convert('RGB')
            mask_pil = load_image(mask_file).convert('L')

            if img_pil.size != mask_pil.size:
                logging.warning(f"Mismatch in size for {img_file} vs {mask_file}; skipping.")
                continue

            img_array = self.preprocess(img_pil, self.scale, is_mask=False)
            mask_array = self.preprocess(mask_pil, self.scale, is_mask=True)

            img_tensor = torch.as_tensor(img_array, dtype=torch.float32)
            mask_tensor = torch.as_tensor(mask_array, dtype=torch.float32)

            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)

            _, h, w = img_tensor.shape

            if self.is_full_image:
                try:
                    # Crop to the standard size that contains the fundus
                    cropped_img, cropped_mask = self.crop_to_fundus(img_tensor, mask_tensor, img_array)

                    # Save as a single patch
                    patch_path = self.patches_dir / f"{img_id}_full"
                    has_lesion = torch.any(mask_tensor > 0.5)

                    torch.save({
                        'image': cropped_img.contiguous(),
                        'mask': cropped_mask.contiguous(),
                        'coords': (0, 0),
                        'has_lesion': has_lesion,
                        'original_shape': img_tensor.shape[1:]
                    }, patch_path)

                    if has_lesion:
                        positive_count += 1
                    patch_index.append((img_id, str(patch_path), has_lesion))
                    logging.info(f"{img_id}: 1 patch (cropped fundus) with shape {cropped_img.shape}")

                    del cropped_img, cropped_mask

                except Exception as e:
                    logging.error(f"Error processing full image {img_id}: {e}")
                    continue

            # If this is regular patch mode (not full image)
            if (h < self.patch_size) or (w < self.patch_size):
                logging.warning(f"{img_id}: scaled to {h}x{w} < patch_size={self.patch_size}; skipping.")
                continue

            stride = self.patch_size // 2
            patch_count = 0
            for y in range(0, h - self.patch_size + 1, stride):
                for x in range(0, w - self.patch_size + 1, stride):
                    img_patch = img_tensor[:, y:y + self.patch_size, x:x + self.patch_size]
                    mask_patch = mask_tensor[:, y:y + self.patch_size, x:x + self.patch_size]

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

        del img_tensor, mask_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logging.info(f"Found {positive_count} positive patches and {len(negative_paths)} negative patches")

        if self.is_full_image:
            # For full images, use all images directly
            self.patch_indices = patch_index
            logging.info(f"Saved {len(self.patch_indices)} uniformly-sized full images to {self.patches_dir}")
        else:

            if self.split == 'test' and not patch_index and not negative_paths:
                logging.warning(f"No patches found for test set with patch_size={self.patch_size}, scale={self.scale}")
                logging.warning(f"You may need to decrease patch size, increase scale, or disable border check")

            # Regular patch handling code
            if self.split == 'train':
                # For training, balance positive and negative patches
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
                # For validation and test, keep all patches to ensure full coverage
                self.patch_indices = patch_index + [(img_id, path, False) for img_id, path in negative_paths]

                logging.info(f"Saved {len(self.patch_indices)} patches for {self.split} to {self.patches_dir}")

                # For test specifically, ensure we have at least one patch per image
                if self.split == 'test' and len(self.patch_indices) == 0 and positive_count == 0:
                    # If no positive patches, use some negative patches
                    self.patch_indices = [(img_id, path, False) for img_id, path in
                                          negative_paths[:min(10, len(negative_paths))]]
                    logging.info(
                        f"Using {len(self.patch_indices)} negative patches for testing due to lack of positive patches")

    def crop_to_fundus(self, image_tensor, mask_tensor, image_array):
        """Crop image and mask to a square that contains the circular fundus region."""
        _, h, w = image_tensor.shape

        try:
            # Convert to the expected format before passing to detect_fundus_diameter
            if image_array.shape[0] == 3 and len(image_array.shape) == 3:
                image_np = image_array.transpose(1, 2, 0)
            else:
                image_np = image_array
            result = self.detect_fundus_diameter(image_np, return_center=True)
            if result[0] is None:  # If detection failed
                # Use fallback values
                diameter = min(h, w)
                center_x, center_y = w // 2, h // 2
                logging.warning(
                    f"Fundus detection failed, using fallback center=({center_x},{center_y}) and diameter={diameter}")
            else:
                diameter, (center_x, center_y) = result

            # Rest of the function remains the same...
            # Calculate the bounding square size (exact diameter, not rounded)
            square_size = int(np.ceil(diameter))

            # Calculate crop boundaries centered on the detected fundus center
            half_size = square_size // 2
            top = max(0, center_y - half_size)
            bottom = min(h, center_y + half_size + (square_size % 2))  # Add 1 if odd size
            left = max(0, center_x - half_size)
            right = min(w, center_x + half_size + (square_size % 2))  # Add 1 if odd size

            # Adjust if we're at image edges while maintaining square shape
            if top == 0:
                bottom = min(h, square_size)
            if left == 0:
                right = min(w, square_size)
            if bottom == h:
                top = max(0, h - square_size)
            if right == w:
                left = max(0, w - square_size)

            # Final check to ensure our crop is square
            actual_height = bottom - top
            actual_width = right - left

            if actual_height != actual_width:
                # Ensure dimensions match by adjusting to the smaller dimension
                new_size = min(actual_height, actual_width)

                # Recenter the crop around the center point
                if actual_height > new_size:
                    diff = actual_height - new_size
                    top += diff // 2
                    bottom = top + new_size
                else:
                    diff = actual_width - new_size
                    left += diff // 2
                    right = left + new_size

            # Perform the crop
            cropped_image = image_tensor[:, top:bottom, left:right]
            cropped_mask = mask_tensor[:, top:bottom, left:right]

            # Verify the crop is square
            assert cropped_image.shape[1] == cropped_image.shape[2], "Crop is not square!"

            # If we need to resize to a consistent size
            if self.patch_size != cropped_image.shape[1]:
                # Use memory-efficient approach for resizing
                with torch.no_grad():
                    resized_image = torch.nn.functional.interpolate(
                        cropped_image.unsqueeze(0),
                        size=(self.patch_size, self.patch_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                    resized_mask = torch.nn.functional.interpolate(
                        cropped_mask.unsqueeze(0),
                        size=(self.patch_size, self.patch_size),
                        mode='nearest'
                    ).squeeze(0)

                    # Free original tensors to save memory
                    del cropped_image, cropped_mask
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    return resized_image, resized_mask

            return cropped_image, cropped_mask

        except Exception as e:
            logging.error(f"Error in crop_to_fundus: {e}")
            # Fallback to center crop if any error occurs
            square_size = min(h, w)
            top = (h - square_size) // 2
            bottom = top + square_size
            left = (w - square_size) // 2
            right = left + square_size

        # Perform the crop
        cropped_image = image_tensor[:, top:bottom, left:right]
        cropped_mask = mask_tensor[:, top:bottom, left:right]

        # Verify the crop is square
        assert cropped_image.shape[1] == cropped_image.shape[2], "Crop is not square!"

        # If we need to resize to a consistent size
        if self.patch_size != cropped_image.shape[1]:
            # Use memory-efficient approach for resizing
            with torch.no_grad():
                resized_image = torch.nn.functional.interpolate(
                    cropped_image.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                resized_mask = torch.nn.functional.interpolate(
                    cropped_mask.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode='nearest'
                ).squeeze(0)

                # Free original tensors to save memory
                del cropped_image, cropped_mask
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                return resized_image, resized_mask

        return cropped_image, cropped_mask

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
        if is_mask:
            if arr.ndim == 3:
                arr = arr[..., 0]
            return (arr > 0).astype(np.float32)
        else:
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = arr.astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            return arr

    def __getitem__(self, idx):
        """Return a single patch (image, mask) loaded from disk. If patch_size is None, returns full image."""
        if self.patch_size is None:
            img_id = self.patch_indices[idx][0]
            patch_data = torch.load(self.patches_dir / f"{img_id}_full", weights_only=True)
        else:
            img_id, patch_path, _ = self.patch_indices[idx]
            patch_data = torch.load(patch_path, weights_only=True)

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
