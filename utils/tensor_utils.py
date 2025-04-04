import torch
import numpy as np
import pandas as pd

def to_python_scalar(value):
    """Convert torch tensors, numpy arrays, or other numeric types to Python scalars."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:  # Single-element tensor
            return value.item()
        elif value.dim() == 0:  # 0-d tensor
            return value.item()
        else:  # Multi-element tensor
            return value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        if value.size == 1:  # Single-element array
            return value.item()
        else:
            return value
    elif hasattr(value, 'item'):  # Objects with .item() method
        return value.item()
    else:
        return value  # Already a Python scalar or other type

def ensure_dict_python_scalars(d):
    """Recursively convert all torch tensors and numpy arrays in a dict to Python scalars."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = ensure_dict_python_scalars(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [to_python_scalar(x) for x in v]
        else:
            result[k] = to_python_scalar(v)
    return result

def fix_dataframe_tensors(df):
    """Convert any tensors in a DataFrame to Python scalars."""
    # Copy the DataFrame to avoid modifying the original
    fixed_df = df.copy()
    
    # Process each column
    for col in fixed_df.columns:
        # Special handling for img_id column
        if col == 'img_id':
            fixed_df[col] = fixed_df[col].astype(str)
            continue
            
        # For other columns, convert to numeric when possible
        try:
            # Try simple conversion first
            fixed_df[col] = pd.to_numeric(fixed_df[col], errors='coerce')
        except:
            # If that fails, try more detailed conversion
            if fixed_df[col].dtype == 'object':
                # Check if column contains tensors or arrays
                fixed_df[col] = fixed_df[col].apply(
                    lambda x: to_python_scalar(x) if isinstance(x, (torch.Tensor, np.ndarray)) else x
                )
                # Now try numeric conversion again
                try:
                    fixed_df[col] = pd.to_numeric(fixed_df[col], errors='coerce')
                except:
                    pass
    
    return fixed_df

def ensure_numeric_dataframe(df, exclude_cols=None):
    """Ensure all columns except those excluded are numeric type."""
    if exclude_cols is None:
        exclude_cols = []
        
    result = df.copy()
    
    for col in result.columns:
        if col not in exclude_cols:
            # Try to convert to numeric
            try:
                result[col] = pd.to_numeric(result[col], errors='coerce')
            except Exception as e:
                print(f"Warning: could not convert column {col} to numeric. Error: {e}")
    
    return result
