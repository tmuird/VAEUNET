import torch
import numpy as np
import pandas as pd
import logging

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
    """Convert all torch tensors and numpy values in a dictionary to Python scalar types.
    This is important for proper serialization to CSV and other formats.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            # Handle different tensor shapes
            if v.numel() == 1:  # Single value tensor
                result[k] = v.item()  # Convert to Python scalar
            else:
                # For multi-value tensors, convert to list of Python scalars
                result[k] = v.detach().cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            # Handle numpy arrays
            if v.size == 1:  # Single value array
                result[k] = v.item()  # Convert to Python scalar
            else:
                result[k] = v.tolist()  # Convert to list
        elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
            # Handle numpy scalar types
            result[k] = v.item()  # Convert to Python scalar
        else:
            # Keep other types as is
            result[k] = v
    return result

def fix_dataframe_tensors(df):
    """Fix a pandas DataFrame with tensor or numpy values.
    
    Args:
        df: A pandas DataFrame that might contain torch tensors or numpy arrays
        
    Returns:
        A pandas DataFrame with all tensors and numpy arrays converted to Python types
    """
    # Create a copy to avoid modifying the original
    fixed_df = df.copy()
    
    for col in fixed_df.columns:
        # Check if this column contains any tensor or numpy values
        if any(isinstance(x, (torch.Tensor, np.ndarray, np.float32, np.float64, np.int32, np.int64)) 
               for x in fixed_df[col] if x is not None):
            
            # Create a new column with converted values
            try:
                fixed_df[col] = fixed_df[col].apply(
                    lambda x: x.item() if isinstance(x, (torch.Tensor, np.ndarray)) and 
                                         (hasattr(x, 'numel') and x.numel() == 1 or 
                                          hasattr(x, 'size') and x.size == 1) 
                             else x.detach().cpu().numpy().tolist() if isinstance(x, torch.Tensor)
                             else x.tolist() if isinstance(x, np.ndarray)
                             else x.item() if isinstance(x, (np.float32, np.float64, np.int32, np.int64))
                             else x
                )
            except Exception as e:
                logging.warning(f"Could not convert column {col}: {e}")
                # Try a more direct approach for problematic columns
                fixed_values = []
                for x in fixed_df[col]:
                    if isinstance(x, (torch.Tensor, np.ndarray)):
                        if hasattr(x, 'numel') and x.numel() == 1 or hasattr(x, 'size') and x.size == 1:
                            fixed_values.append(float(x.item()))
                        else:
                            try:
                                if isinstance(x, torch.Tensor):
                                    fixed_values.append(float(x.detach().cpu().numpy().tolist()))
                                else:
                                    fixed_values.append(float(x.tolist()))
                            except:
                                fixed_values.append(None)
                    elif isinstance(x, (np.float32, np.float64, np.int32, np.int64)):
                        fixed_values.append(float(x.item()))
                    else:
                        fixed_values.append(x)
                fixed_df[col] = fixed_values
    
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
