import argparse
import os
import logging
import sys
import subprocess
import torch
import numpy as np
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# Print what directory we're in
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

# Install dependencies directly (bypassing requirements.txt)
packages = [
    "timm==0.4.12",
    "albumentations==1.0.3", 
    "opencv-python-headless==4.5.3.56",
    "tensorboard==2.6.0",
    "scipy==1.7.1",
    "scikit-learn==0.24.2",
    "scikit-image==0.18.3", 
    "matplotlib==3.5.3",
    "tqdm==4.62.0",
    "pillow==8.3.1",
    "torchmetrics==0.7.3"
]


# Install packages
print("Installing packages directly:")
for package in packages:
    print(f"Installing {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# Create a patch for the missing dice function import
print("Creating patch for torchmetrics...")
# Create a patch for utils/metrics.py
metrics_path = os.path.join('src', 'utils', 'metrics.py')
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics_content = f.read()
    
    # Replace the problematic import
    patched_content = metrics_content.replace(
        'from torchmetrics.functional import dice', 
        'from torchmetrics.functional.classification import dice_score as dice'
    )
    
    # Write back the patched file
    with open(metrics_path, 'w') as f:
        f.write(patched_content)
    
    print(f"Patched {metrics_path} for correct torchmetrics import")
else:
    print(f"Warning: Could not find {metrics_path} to patch")


# Add src directory to path for imports
sys.path.append('/opt/ml/code/src')

# Import model and utilities from src
from src.unet.unet_resnet import UNetResNet
from src.utils.data_loading import IDRIDDataset
from src.utils.loss import CombinedLoss
from src.evaluate import evaluate

# SageMaker paths
TRAINING_DIR = '/opt/ml/input/data/training'
VALIDATION_DIR = '/opt/ml/input/data/validation'
MODEL_DIR = '/opt/ml/model'
OUTPUT_DIR = '/opt/ml/output'

def collate_patches(batch):
    """Custom collate function to handle patches.
    Improved to better handle full images of the same size.
    """
    # Check if all images are the same shape
    shapes = [x['image'].shape for x in batch]
    
    if len(set(tuple(shape) for shape in shapes)) == 1:
        # All shapes are the same, stack normally
        return {
            'image': torch.stack([x['image'] for x in batch]),
            'mask': torch.stack([x['mask'] for x in batch]),
            'img_id': [x['img_id'] for x in batch]
        }
    else:
        # Different shapes - don't stack, just return list
        return {
            'image': [x['image'] for x in batch],
            'mask': [x['mask'] for x in batch],
            'img_id': [x['img_id'] for x in batch]
        }

def train(args):
    """Training function adapted for SageMaker environment"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Display GPU information if available
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name()}')
        logging.info(f'Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        logging.info(f'Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB')
        
        # Set memory limits for better management
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Create model
    logging.info('Initializing model...')
    model = UNetResNet(
        n_channels=3,
        n_classes=1,
        backbone='resnet34',
        pretrained=True,
        use_attention=args.use_attention,
        use_skip=True
    )
    model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    
    # Add this debugging code right before creating the datasets

    logging.info(f"Checking data directories:")
    logging.info(f"TRAINING_DIR contents: {os.listdir(TRAINING_DIR)}")
    try:
        # Try to list subdirectories to understand the structure
        for item in os.listdir(TRAINING_DIR):
            item_path = os.path.join(TRAINING_DIR, item)
            if os.path.isdir(item_path):
                logging.info(f"  - {item}/ contents: {os.listdir(item_path)}")
    except Exception as e:
        logging.error(f"Error listing directories: {e}")

    # Adjust IDRIDDataset for SageMaker paths
    class SageMakerDatasetAdapter(IDRIDDataset):
        def __init__(self, base_dir, split, scale, patch_size, lesion_type):
            self.base_dir = base_dir
            self.split = split
            self.scale = scale
            self.patch_size = patch_size
            self.lesion_type = lesion_type
            self.transform = None
            self.patch_indices = []  # Initialize this attribute that __len__ needs
            self.skip_border_check = True  # Skip border checks
            
            # Check if we have the expected structure
            imgs_dir_path = os.path.join(base_dir, 'imgs', split)
            
            if os.path.exists(imgs_dir_path) and split:
                # Original directory structure exists
                logging.info(f"Using original directory structure with split '{split}'")
                super().__init__(base_dir, split, scale, patch_size, lesion_type)
            else:
                # Use the flattened SageMaker structure
                logging.info(f"Using adapted directory structure for SageMaker")
                
                # Set up paths
                self.images_dir = os.path.join(base_dir, 'imgs')
                self.masks_dir = os.path.join(base_dir, 'masks', lesion_type)
                
                # Initialize base attributes needed by parent class
                self.ids = []
                
                # Find all images
                if os.path.exists(self.images_dir):
                    for file in os.listdir(self.images_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_id = os.path.splitext(file)[0]
                            mask_path = os.path.join(self.masks_dir, f"{img_id}_{lesion_type}.tif")
                            
                            # Only add if mask exists
                            if os.path.exists(mask_path):
                                self.ids.append(img_id)
                                # Create a direct mapping from index to image ID
                                self.patch_indices.append((img_id, None, True))
                
                logging.info(f"Found {len(self.ids)} images in {self.images_dir}")
                
                # Initialize transforms directly instead of calling parent method
                import albumentations as A
                from albumentations.pytorch import ToTensorV2
                
                if split == 'train':
                    self.transform = A.Compose([
                        A.Resize(height=int(2848 * self.scale), width=int(4288 * self.scale)),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])
                else:
                    self.transform = A.Compose([
                        A.Resize(height=int(2848 * self.scale), width=int(4288 * self.scale)),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])

        def __getitem__(self, idx):
            """Custom implementation that loads images directly instead of from patches"""
            img_id = self.ids[idx]
            
            # Load image and mask
            img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
            mask_path = os.path.join(self.masks_dir, f"{img_id}_{self.lesion_type}.tif")
            
            # Load and process image
            from PIL import Image
            import numpy as np
            
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Convert to numpy arrays
            image_np = np.array(image)
            mask_np = np.array(mask)
            mask_np = (mask_np > 0).astype(np.float32)  # Binarize mask
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image_np, mask=mask_np)
                image_tensor = transformed['image']
                mask_tensor = transformed['mask']
            else:
                # Manual conversion as fallback
                image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1).astype(np.float32) / 255.0)
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # Add channel dimension
            
            # Ensure mask is right shape
            if len(mask_tensor.shape) == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'img_id': img_id
            }

    # Replace IDRIDDataset with our adapter in the dataset creation
    train_dataset = SageMakerDatasetAdapter(
        base_dir=TRAINING_DIR,
        split='train',
        scale=args.scale,
        patch_size=args.patch_size,
        lesion_type=args.lesion_type
    )
    
    val_dataset = SageMakerDatasetAdapter(
        base_dir=VALIDATION_DIR,
        split='val',
        scale=args.scale,
        patch_size=args.patch_size,
        lesion_type=args.lesion_type
    )
    
    logging.info(f'Dataset sizes:')
    logging.info(f'- Training: {len(train_dataset)} images')
    logging.info(f'- Validation: {len(val_dataset)} images')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_patches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_patches
    )
    
    # Calculate effective batch size and adjusted learning rate
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    learning_rate = args.learning_rate
    
    # Scale learning rate based on batch size and patch size
    if args.batch_size >= 8:
        learning_rate *= 2.0
        
    if args.patch_size == 512 and args.scale == 1.0:
        learning_rate *= 1.5
        weight_decay = 5e-6  # Reduced for high-resolution
    else:
        weight_decay = 1e-5
    
    logging.info(f'Adjusted learning rate: {learning_rate:.6f}')
    logging.info(f'Adjusted weight decay: {weight_decay:.6f}')
    
    # Initialize loss function, optimizer, and scheduler
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # Enable gradient scaler for mixed precision
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # Initialize variables for training loop
    best_dice = 0.0
    best_val_score = float('-inf')
    no_improvement_count = 0
    global_step = 0
    
    # Warmup epochs based on resolution
    warmup_epochs = 3 if args.patch_size == 512 and args.scale == 1.0 else 0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Apply warmup schedule if needed
                if epoch <= warmup_epochs:
                    # Linear warmup from 20% to 100% of learning rate
                    progress = (epoch - 1 + batch_idx/len(train_loader)) / warmup_epochs
                    warmup_factor = 0.2 + 0.8 * progress
                    current_lr = learning_rate * warmup_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                
                # Get images and masks
                images = batch['image']
                true_masks = batch['mask']
                
                # Handle different shapes or convert to appropriate format
                if isinstance(images, list):
                    # Process each image individually if they have different sizes
                    batch_loss = 0
                    for i in range(len(images)):
                        img = images[i].unsqueeze(0).to(device=device, dtype=torch.float32)
                        mask = true_masks[i].unsqueeze(0).to(device=device, dtype=torch.float32)
                        
                        with torch.cuda.amp.autocast(enabled=args.amp):
                            outputs = model(img)
                            # Check if the model output is a tuple and handle accordingly
                            if isinstance(outputs, tuple):
                                pred = outputs[0]  # In VAE models, first output is usually the reconstruction
                            else:
                                pred = outputs
                            
                            loss = criterion(pred, mask) / args.gradient_accumulation_steps
                        
                        grad_scaler.scale(loss).backward()
                        batch_loss += loss.item()
                else:
                    # Process batch normally
                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                    
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        outputs = model(images)
                        # Check if the model output is a tuple and handle accordingly
                        if isinstance(outputs, tuple):
                            pred = outputs[0]  # In VAE models, first output is usually the reconstruction
                        else:
                            pred = outputs
                        
                        loss = criterion(pred, true_masks) / args.gradient_accumulation_steps
                    
                    grad_scaler.scale(loss).backward()
                    batch_loss = loss.item()
                
                # Update weights after accumulating gradients
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if args.gradient_clipping > 0:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                    
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                # Update metrics
                epoch_loss += batch_loss
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix(**{'loss (batch)': batch_loss})
                pbar.update()
        
        # Calculate average loss for epoch
        epoch_loss = epoch_loss / len(train_loader)
        logging.info(f'Epoch {epoch} - Average loss: {epoch_loss:.4f}')
        
        # Evaluate on validation set
        val_metrics, val_samples = evaluate(model, val_loader, device, args.amp)
        val_dice = val_metrics['dice']  # Correctly access the metrics dictionary
        val_score = val_dice  # Use dice as our primary metric

        logging.info(f'Validation Dice score: {val_dice:.4f}')
        
        # Update learning rate scheduler
        scheduler.step(val_score)
        
        # Check for improvement
        if val_score > best_val_score:
            best_val_score = val_score
            best_dice = val_dice
            no_improvement_count = 0
            
            # Save best model
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, f'best_model_{args.lesion_type}.pth')
            )
            logging.info(f'Saved new best model with dice score: {best_dice:.4f}')
        else:
            no_improvement_count += 1
            logging.info(f'No improvement for {no_improvement_count} epochs')
            
            if no_improvement_count >= args.early_stopping_patience:
                logging.info(f'Early stopping after {epoch} epochs')
                break
        
    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, f'final_model_{args.lesion_type}.pth')
    )
    logging.info(f'Training completed. Best dice score: {best_dice:.4f}')
    
    return {
        'best_dice': float(best_dice),
        'best_val_score': float(best_val_score)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--scale', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--patch-size', type=int, default=512, help='Size of patches to extract')
    parser.add_argument('--amp', type=lambda x: x.lower() == 'true', default=True, help='Use mixed precision')
    parser.add_argument('--use-attention', type=lambda x: x.lower() == 'true', default=True, help='Use attention in model')
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lesion-type', type=str, default='EX', help='Lesion type')

    args = parser.parse_args()
    train(args)