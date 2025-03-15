import boto3
import os
import argparse

def upload_directory_to_s3(local_directory, bucket, s3_prefix):
    """Upload directory contents to S3 bucket"""
    s3_client = boto3.client('s3')
    
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_prefix, relative_path)
            
            print(f"Uploading {local_path} to s3://{bucket}/{s3_path}")
            s3_client.upload_file(local_path, bucket, s3_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Local data directory')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--s3-prefix', type=str, required=True, help='S3 prefix for data')
    args = parser.parse_args()
    
    upload_directory_to_s3(args.data_dir, args.bucket, args.s3_prefix)
