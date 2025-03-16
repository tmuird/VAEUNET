import boto3
import sagemaker
import argparse
from sagemaker.pytorch import PyTorch

def launch_training(args):
    # Explicitly create boto session with region
    boto_session = boto3.Session(profile_name=args.profile, region_name=args.region)
    print(f"Using AWS region: {args.region}")
    
    # Create SageMaker session with explicit boto session
    session = sagemaker.Session(boto_session=boto_session)
    
    # Get account ID for constructing the role ARN
    sts = boto_session.client('sts')
    account_id = sts.get_caller_identity()['Account']
    
    # Set the role ARN explicitly
    role_arn = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    print(f"Using execution role: {role_arn}")
    
    # Configure estimator with explicit session
# Update the estimator configuration section to include:

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='code',
        role=role_arn,
        framework_version='1.9.1',
        py_version='py38',
        instance_count=1,
        instance_type=args.instance_type,
        hyperparameters={
            'batch-size': args.batch_size,
            'learning-rate': args.learning_rate,
            'epochs': args.epochs,
            'scale': args.scale,
            'patch-size': args.patch_size,
            'lesion-type': args.lesion_type,
            'use-attention': args.use_attention,
            'amp': args.amp
        },
        max_run=args.max_run,
        sagemaker_session=session,
        # Remove or comment out this line:
        # dependencies=['requirements.txt'],  
        disable_profiler=True
    )
    
    # Define data channels
    data_channels = {
        'training': f's3://{args.bucket}/{args.prefix}/train',
        'validation': f's3://{args.bucket}/{args.prefix}/val'
    }
    
    print(f"Data channels configured:")
    print(f"  Training: {data_channels['training']}")
    print(f"  Validation: {data_channels['validation']}")
    
    # Launch the training job
    print(f"\nLaunching training job: {args.job_name}")
    estimator.fit(data_channels, job_name=args.job_name, wait=args.wait)
    print(f"Training job {args.job_name} submitted!")
    
    return estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch SageMaker training job')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--prefix', type=str, default='vaeunet-data', help='S3 prefix for data')
    parser.add_argument('--job-name', type=str, required=True, help='Training job name')
    parser.add_argument('--instance-type', type=str, default='ml.p3.2xlarge', 
                        help='Training instance type')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--scale', type=float, default=1.0, help='Image scale factor')
    parser.add_argument('--patch-size', type=int, default=512, help='Patch size')
    parser.add_argument('--lesion-type', type=str, default='EX', help='Lesion type')
    parser.add_argument('--use-attention', action='store_true', help='Use attention mechanism')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--max-run', type=int, default=86400, help='Max runtime in seconds')
    parser.add_argument('--wait', action='store_true', help='Wait for job to complete')
    parser.add_argument('--profile', type=str, default='sagemaker', help='AWS profile name')
    parser.add_argument('--region', type=str, default='eu-west-2', help='AWS region')
    
    args = parser.parse_args()
    
    # Execute training
    launch_training(args)