import boto3
import os
from pathlib import Path
import argparse

def upload_data_to_s3(data_dir, bucket, prefix):
    """Upload training and validation data to S3"""
    s3 = boto3.client('s3')
    data_dir = Path(data_dir)
    
    # Prepare data directories
    for split in ['train', 'val']:
        # Upload images
        img_dir = data_dir / 'imgs' / split
        if img_dir.exists():
            print(f"Uploading {split} images...")
            for img_file in img_dir.glob('*.jpg'):
                key = f"{prefix}/{split}/imgs/{img_file.name}"
                print(f"  Uploading {img_file} to s3://{bucket}/{key}")
                s3.upload_file(str(img_file), bucket, key)
        
        # Upload masks for each lesion type
        for lesion in ['EX', 'HE', 'MA', 'OD']:
            mask_dir = data_dir / 'masks' / split / lesion
            if mask_dir.exists():
                print(f"Uploading {split} masks for {lesion}...")
                for mask_file in mask_dir.glob('*.tif'):
                    key = f"{prefix}/{split}/masks/{lesion}/{mask_file.name}"
                    print(f"  Uploading {mask_file} to s3://{bucket}/{key}")
                    s3.upload_file(str(mask_file), bucket, key)
    
    print("Data upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload IDRID dataset to S3')
    parser.add_argument('--data-dir', type=str, required=True, 
                        help='Local directory with your IDRID dataset')
    parser.add_argument('--bucket', type=str, required=True,
                        help='S3 bucket name')
    parser.add_argument('--prefix', type=str, default='data',
                        help='S3 prefix for your data')
    
    args = parser.parse_args()
    upload_data_to_s3(args.data_dir, args.bucket, args.prefix)
