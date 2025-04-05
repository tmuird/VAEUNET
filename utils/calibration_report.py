import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def interpret_calibration_results(ece_value, bin_accs=None, bin_confs=None):
    """Interpret ECE values and provide insights on model calibration quality.
    
    Args:
        ece_value: Expected Calibration Error value (float)
        bin_accs: Optional array of accuracies per bin
        bin_confs: Optional array of confidences per bin
        
    Returns:
        dict: Dictionary containing interpretation and recommendations
    """
    # Interpret ECE value
    if ece_value < 0.01:
        quality = "Excellent"
        description = (
            "Your model demonstrates exceptional calibration quality. This means the "
            "predicted probabilities almost perfectly match the actual frequencies "
            "observed in the data. For medical applications, this level of calibration "
            "is ideal as clinicians can trust the confidence levels provided by the model."
        )
        recommendation = (
            "No calibration adjustment needed. The uncertainty estimates from this model "
            "can be trusted with high confidence. This exceptional calibration may be "
            "due to the VAE architecture's natural ability to represent uncertainty "
            "and/or your temperature scaling approach. Consider documenting your "
            "methodology as this level of calibration is better than many published models."
        )
    elif ece_value < 0.05:
        quality = "Good"
        description = (
            "Your model shows good calibration. The predicted probabilities are generally "
            "reliable indicators of true frequencies, with only minor discrepancies."
        )
        recommendation = (
            "Consider light temperature scaling (T≈1.1-1.2) if you notice slight "
            "overconfidence, or T≈0.8-0.9 for slight underconfidence."
        )
    elif ece_value < 0.15:
        quality = "Fair"
        description = (
            "Your model has fair calibration quality. There are noticeable discrepancies "
            "between predicted probabilities and actual frequencies."
        )
        recommendation = (
            "Apply temperature scaling with T>1.0 for overconfidence issues or "
            "T<1.0 for underconfidence. Consider retraining with focal loss or "
            "label smoothing."
        )
    else:
        quality = "Poor"
        description = (
            "Your model shows poor calibration. The predicted probabilities significantly "
            "differ from actual frequencies, making confidence estimates unreliable."
        )
        recommendation = (
            "Apply more aggressive calibration techniques such as Platt scaling or "
            "isotonic regression. Consider architecture changes that better handle "
            "uncertainty, such as ensemble methods."
        )
    
    # Calculate calibration direction if bin data is provided
    direction = "Not determined (bin data not provided)"
    if bin_accs is not None and bin_confs is not None and len(bin_accs) == len(bin_confs):
        # Calculate weighted difference between accuracy and confidence
        diffs = bin_accs - bin_confs
        if np.mean(diffs) > 0.01:
            direction = "Underconfidence (model predicts lower probabilities than actual frequencies)"
        elif np.mean(diffs) < -0.01:
            direction = "Overconfidence (model predicts higher probabilities than actual frequencies)"
        else:
            direction = "Well-balanced (no systematic bias in either direction)"
    
    return {
        "ece_value": ece_value,
        "quality": quality,
        "description": description,
        "recommendation": recommendation,
        "calibration_direction": direction
    }

def generate_calibration_report(ece_values, model_name, output_dir=None):
    """Generate a comprehensive report from multiple ECE values.
    
    Args:
        ece_values: List or array of ECE values from different images/models
        model_name: Name of the model for report titles
        output_dir: Optional directory to save report visualizations
    
    Returns:
        DataFrame with summary statistics
    """
    # Create summary statistics
    ece_array = np.array(ece_values)
    summary = {
        "model": model_name,
        "mean_ece": np.mean(ece_array),
        "median_ece": np.median(ece_array),
        "std_ece": np.std(ece_array),
        "min_ece": np.min(ece_array),
        "max_ece": np.max(ece_array),
        "percent_excellent": np.mean(ece_array < 0.01) * 100,
        "percent_good": np.mean((ece_array >= 0.01) & (ece_array < 0.05)) * 100,
        "percent_fair": np.mean((ece_array >= 0.05) & (ece_array < 0.15)) * 100,
        "percent_poor": np.mean(ece_array >= 0.15) * 100
    }
    
    # Get overall interpretation
    interpretation = interpret_calibration_results(summary["mean_ece"])
    summary.update({
        "quality": interpretation["quality"],
        "description": interpretation["description"],
        "recommendation": interpretation["recommendation"]
    })
    
    # Create visualizations if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create histogram with quality thresholds
        plt.figure(figsize=(10, 6))
        sns.histplot(ece_array, kde=True, bins=20)
        
        # Add vertical lines for quality thresholds
        plt.axvline(0.01, color='green', linestyle='--', label='Excellent (< 0.01)')
        plt.axvline(0.05, color='orange', linestyle='--', label='Good (< 0.05)')
        plt.axvline(0.15, color='red', linestyle='--', label='Fair (< 0.15)')
        
        # Add mean ECE line
        plt.axvline(summary["mean_ece"], color='blue', linestyle='-', 
                   label=f'Mean ECE: {summary["mean_ece"]:.4f}')
        
        plt.title(f'Distribution of ECE Values - {model_name}', fontsize=14)
        plt.xlabel('Expected Calibration Error (ECE)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "ece_distribution.png", dpi=300)
        plt.close()
        
        # Create summary table as image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = [
            ["Metric", "Value"],
            ["Mean ECE", f"{summary['mean_ece']:.4f}"],
            ["Median ECE", f"{summary['median_ece']:.4f}"],
            ["Standard Deviation", f"{summary['std_ece']:.4f}"],
            ["Min ECE", f"{summary['min_ece']:.4f}"],
            ["Max ECE", f"{summary['max_ece']:.4f}"],
            ["Excellent Calibration %", f"{summary['percent_excellent']:.1f}%"],
            ["Good Calibration %", f"{summary['percent_good']:.1f}%"],
            ["Fair Calibration %", f"{summary['percent_fair']:.1f}%"],
            ["Poor Calibration %", f"{summary['percent_poor']:.1f}%"],
            ["Overall Quality", summary['quality']],
        ]
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Highlight quality cell based on value
        quality_cell = table[(len(table_data)-1, 1)]
        if summary['quality'] == "Excellent":
            quality_cell.set_facecolor('lightgreen')
        elif summary['quality'] == "Good":
            quality_cell.set_facecolor('palegreen')
        elif summary['quality'] == "Fair":
            quality_cell.set_facecolor('khaki')
        else:
            quality_cell.set_facecolor('salmon')
            
        plt.title(f'Calibration Quality Summary - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / "calibration_summary.png", dpi=300)
        plt.close()
    
    return pd.DataFrame([summary])
