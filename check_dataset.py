import matplotlib.pyplot as plt
from utils.data_loading import IDRIDDataset
import logging

logging.basicConfig(level=logging.INFO)

def check_masks():
    # Initialize dataset
    dataset = IDRIDDataset(base_dir='data', split='val')
    
    # Get first item
    sample = dataset[0]
    
    # Print basic information
    print(f"Image ID: {sample['img_id']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    
    # Visualize each mask channel
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot original image
    axes[0].imshow(sample['image'].numpy().transpose(1, 2, 0))
    axes[0].set_title('Original Image')
    
    # Plot each mask channel
    for i, class_name in enumerate(dataset.classes):
        axes[i+1].imshow(sample['mask'][i], cmap='gray')
        axes[i+1].set_title(f'Mask: {class_name}')
        
        # Print sum of mask pixels to verify it's not empty
        print(f"Sum of pixels in {class_name} mask: {sample['mask'][i].sum()}")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    check_masks()
    