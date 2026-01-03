"""
Linearity of Log-Odds Check for Logistic Regression

This module checks whether continuous features have a linear relationship 
with log-odds of the target — a key assumption for logistic regression.

Usage:
    from src.linearity_check import check_linearity
    
    results = check_linearity(
        df=my_data,
        features=['Amount', 'Age', 'Balance'],
        target_col='Success'
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from scipy import stats


def check_linearity(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    n_bins: int = 10,
    figsize_per_plot: Tuple[int, int] = (6, 5),
    cols_per_row: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Check linearity of log-odds for continuous features.
    
    For logistic regression, we assume continuous predictors have a linear
    relationship with the log-odds of the outcome. This function bins each
    feature into quantiles, calculates log-odds per bin, and visualizes
    whether the relationship is approximately linear.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    features : list of str
        List of continuous feature names to check
    target_col : str
        Binary target variable (0/1)
    n_bins : int, default=10
        Number of quantile bins (deciles by default)
    figsize_per_plot : tuple, default=(6, 5)
        Size of each subplot (width, height)
    cols_per_row : int, default=3
        Number of plots per row
    verbose : bool, default=True
        Print interpretation guide
    
    Returns
    -------
    pd.DataFrame
        Results table with columns:
        - Feature: feature name
        - R_squared: R² of linear fit (closer to 1 = more linear)
        - Correlation: Pearson correlation coefficient
        - P_value: statistical significance
        - Assessment: ✅ Linear / ⚠️ Moderate / ❌ Non-linear
    
    Examples
    --------
    >>> results = check_linearity(
    ...     df=loan_data,
    ...     features=['Amount', 'Age', 'Balance'],
    ...     target_col='Success',
    ...     n_bins=10
    ... )
    >>> print(results)
    """
    # Validate inputs
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("No valid features found in DataFrame")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Calculate grid dimensions
    n_features = len(features)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(
        n_rows, cols_per_row, 
        figsize=(figsize_per_plot[0] * cols_per_row, figsize_per_plot[1] * n_rows)
    )
    
    # Flatten axes for easier iteration
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    results = []
    
    for i, col in enumerate(features):
        ax = axes[i]
        
        # Create binned column
        try:
            binned = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            # If too few unique values, use available unique values
            binned = pd.cut(df[col], bins=min(n_bins, df[col].nunique()), labels=False)
        
        # Calculate stats per bin
        temp_df = df[[col, target_col]].copy()
        temp_df['bin'] = binned
        
        binned_stats = temp_df.groupby('bin').agg({
            col: 'mean',
            target_col: ['mean', 'sum', 'count']
        })
        binned_stats.columns = ['mean_value', 'success_rate', 'success_count', 'total_count']
        binned_stats = binned_stats.dropna()
        
        # Calculate log-odds (with smoothing to avoid log(0))
        eps = 0.001
        binned_stats['log_odds'] = np.log(
            (binned_stats['success_rate'] + eps) / 
            (1 - binned_stats['success_rate'] + eps)
        )
        
        # Fit linear trend
        x = binned_stats['mean_value'].values
        y = binned_stats['log_odds'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Plot
        ax.scatter(x, y, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Trend line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2, label=f'R²={r_squared:.3f}')
        
        ax.set_xlabel(f'{col}', fontsize=10)
        ax.set_ylabel('Log-Odds', fontsize=10)
        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Assess linearity
        if r_squared >= 0.7:
            assessment = "✅ Linear"
        elif r_squared >= 0.4:
            assessment = "⚠️ Moderate"
        else:
            assessment = "❌ Non-linear"
        
        results.append({
            'Feature': col,
            'R_squared': round(r_squared, 4),
            'Correlation': round(r_value, 4),
            'P_value': round(p_value, 4),
            'Assessment': assessment
        })
    
    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Linearity of Log-Odds Check', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("\nLinearity Assessment Results")
        print("=" * 60)
        print(results_df.to_string(index=False))
        print("\n" + "=" * 60)
        print("Interpretation Guide:")
        print("  R² ≥ 0.70  → ✅ Linear (assumption satisfied)")
        print("  R² 0.40-0.70 → ⚠️ Moderate (consider transformation)")
        print("  R² < 0.40  → ❌ Non-linear (transform or bin the feature)")
        print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 1000
    
    # Create features with different relationships to log-odds
    x_linear = np.random.normal(50, 15, n)  # Linear relationship
    x_nonlinear = np.random.uniform(0, 100, n)  # Non-linear
    
    # Generate target based on linear relationship with x_linear
    prob = 1 / (1 + np.exp(-(0.05 * x_linear - 2.5)))
    target = (np.random.random(n) < prob).astype(int)
    
    test_df = pd.DataFrame({
        'Linear_Feature': x_linear,
        'Random_Feature': x_nonlinear,
        'Success': target
    })
    
    print("Testing check_linearity()...")
    results = check_linearity(
        df=test_df,
        features=['Linear_Feature', 'Random_Feature'],
        target_col='Success',
        n_bins=10
    )

