import sagemaker
from sagemaker.pytorch import PyTorch
import argparse

def launch_training_job(args):
    # Initialize SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role() if args.role is None else args.role
    
    # Configure PyTorch estimator
    pytorch_estimator = PyTorch(
        entry_point='train.py',
        source_dir='code',
        role=role,
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
            'use-attention': args.use_attention
        },
        max_run=args.max_run
    )
    
    # Start training job
    data_channels = {
        'training': f's3://{args.bucket}/{args.data_prefix}/train',
        'validation': f's3://{args.bucket}/{args.data_prefix}/val'
    }
    
    pytorch_estimator.fit(data_channels, wait=args.wait)
    
    return pytorch_estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket for data and model')
    parser.add_argument('--data-prefix', type=str, required=True, help='S3 prefix for data')
    parser.add_argument('--role', type=str, help='SageMaker IAM role')
    parser.add_argument('--instance-type', type=str, default='ml.p3.2xlarge', 
                        help='Training instance type (default: ml.p3.2xlarge)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--lesion-type', type=str, default='EX')
    parser.add_argument('--use-attention', type=bool, default=True)
    parser.add_argument('--max-run', type=int, default=86400, help='Max seconds for training job')
    parser.add_argument('--wait', action='store_true', help='Wait for job completion')
    
    args = parser.parse_args()
    launch_training_job(args)
