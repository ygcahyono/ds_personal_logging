"""
Outlier Analysis and Removal Module

This module provides functions to:
1. Analyze feature distributions (min, max, percentiles, outliers)
2. Selectively remove outliers from specific columns using z-score method

Usage:
    from src.outlier_analysis import describe_features, remove_outliers
    
    # Analyze distributions
    stats_df = describe_features(df, columns=['Amount', 'Balance'])
    
    # Remove outliers from specific column only
    clean_df, mask = remove_outliers(df, column='Balance', z_threshold=3)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union


def describe_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    percentiles: List[float] = [90, 95, 99, 99.5],
    z_threshold: float = 3.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate detailed distribution statistics for numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list of str, optional
        Columns to analyze. If None, analyzes all numeric columns.
    percentiles : list of float, default=[90, 95, 99, 99.5]
        Percentiles to calculate (values between 0-100)
    z_threshold : float, default=3.0
        Z-score threshold for counting outliers
    verbose : bool, default=True
        Print formatted output
    
    Returns
    -------
    pd.DataFrame
        Statistics table with columns:
        - Feature, Min, Max, Mean, Median, Std
        - Percentile columns (P90, P95, etc.)
        - Z_Outliers (count), Z_Outliers_Pct (percentage)
    
    Examples
    --------
    >>> stats = describe_features(df, columns=['Amount', 'Balance'])
    >>> stats = describe_features(df, percentiles=[50, 75, 90, 99])
    """
    # Select columns
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    else:
        # Validate columns exist and are numeric
        columns = [c for c in columns if c in df.columns]
        non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns: {non_numeric}")
    
    if not columns:
        raise ValueError("No valid numeric columns to analyze")
    
    results = []
    
    for col in columns:
        data = df[col].dropna()
        
        # Basic stats
        stats = {
            'Feature': col,
            'Count': len(data),
            'Min': data.min(),
            'Max': data.max(),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std': data.std()
        }
        
        # Percentiles
        for p in percentiles:
            stats[f'P{int(p)}'] = data.quantile(p / 100)
        
        # Z-score outliers
        z_scores = (data - data.mean()).abs() / data.std()
        outlier_count = (z_scores > z_threshold).sum()
        stats['Z_Outliers'] = outlier_count
        stats['Z_Outliers_Pct'] = (outlier_count / len(data)) * 100
        
        results.append(stats)
    
    stats_df = pd.DataFrame(results)
    
    if verbose:
        print("Feature Distribution Analysis")
        print("=" * 80)
        print(f"Z-score threshold: {z_threshold} SD")
        print("=" * 80)
        
        for col in columns:
            row = stats_df[stats_df['Feature'] == col].iloc[0]
            print(f"\n📊 {col}")
            print("-" * 50)
            print(f"   Min:    {row['Min']:,.2f}")
            print(f"   Max:    {row['Max']:,.2f}")
            print(f"   Mean:   {row['Mean']:,.2f}")
            print(f"   Median: {row['Median']:,.2f}")
            print(f"   Std:    {row['Std']:,.2f}")
            
            print(f"\n   Percentiles:")
            for p in percentiles:
                print(f"   {int(p)}th: {row[f'P{int(p)}']:,.2f}")
            
            print(f"\n   Z-score outliers (>{z_threshold} SD): "
                  f"{int(row['Z_Outliers'])} ({row['Z_Outliers_Pct']:.2f}%)")
    
    return stats_df


def plot_distributions(
    df: pd.DataFrame,
    columns: List[str],
    figsize_per_col: Tuple[int, int] = (5, 4),
    show_percentile: float = 99
) -> None:
    """
    Plot histograms and boxplots for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list of str
        Columns to plot
    figsize_per_col : tuple, default=(5, 4)
        Figure size per column (width, height)
    show_percentile : float, default=99
        Percentile line to show on histogram
    """
    columns = [c for c in columns if c in df.columns]
    n_cols = len(columns)
    
    if n_cols == 0:
        print("No valid columns to plot")
        return
    
    # Histograms
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize_per_col[0] * n_cols, figsize_per_col[1]))
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        data = df[col].dropna()
        axes[i].hist(data, bins=50, edgecolor='black', alpha=0.7)
        pctl_val = data.quantile(show_percentile / 100)
        axes[i].axvline(pctl_val, color='red', linestyle='--', 
                        label=f'{int(show_percentile)}th pctl')
        axes[i].set_title(f'{col}\n(Max: {data.max():,.0f})')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.suptitle('Distribution of Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Box plots
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize_per_col[0] * n_cols, figsize_per_col[1]))
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(col)
        axes[i].set_ylabel('Value')
    
    plt.suptitle('Box Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    z_threshold: float = 3.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers from a SINGLE column using z-score method.
    
    Use this when you want to selectively remove outliers from specific 
    columns rather than all columns at once.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Single column name to remove outliers from
    z_threshold : float, default=3.0
        Z-score threshold (removes rows where |z| > threshold)
    verbose : bool, default=True
        Print removal summary
    
    Returns
    -------
    tuple of (cleaned_df, outlier_mask)
        - cleaned_df: DataFrame with outliers removed
        - outlier_mask: Boolean Series (True = outlier row)
    
    Examples
    --------
    >>> # Remove outliers from only one column
    >>> clean_df, mask = remove_outliers(df, column='Balance', z_threshold=3)
    >>> print(f"Removed {mask.sum()} rows")
    
    >>> # Chain multiple removals
    >>> df_clean, _ = remove_outliers(df, 'Amount', z_threshold=3)
    >>> df_clean, _ = remove_outliers(df_clean, 'Balance', z_threshold=2.5)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    data = df[column]
    mean = data.mean()
    std = data.std()
    
    # Calculate z-scores
    z_scores = (data - mean).abs() / std
    
    # Identify outliers
    outlier_mask = z_scores > z_threshold
    outlier_count = outlier_mask.sum()
    
    # Remove outliers
    cleaned_df = df[~outlier_mask].copy()
    
    if verbose:
        print(f"Outlier Removal: {column}")
        print("=" * 50)
        print(f"   Method: Z-score > {z_threshold} SD")
        print(f"   Mean: {mean:,.2f}")
        print(f"   Std:  {std:,.2f}")
        print(f"   Threshold: |value - mean| > {z_threshold * std:,.2f}")
        print("-" * 50)
        print(f"   Original rows:  {len(df):,}")
        print(f"   Outliers found: {outlier_count:,} ({outlier_count/len(df)*100:.2f}%)")
        print(f"   Remaining rows: {len(cleaned_df):,}")
        
        if outlier_count > 0:
            outlier_values = data[outlier_mask]
            print(f"\n   Outlier range: {outlier_values.min():,.2f} to {outlier_values.max():,.2f}")
    
    return cleaned_df, outlier_mask


def remove_outliers_multi(
    df: pd.DataFrame,
    columns_config: dict,
    verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove outliers from multiple columns with different thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns_config : dict
        Dictionary mapping column names to z-score thresholds
        e.g., {'Balance': 3.0, 'Amount': 2.5}
    verbose : bool, default=True
        Print removal summary
    
    Returns
    -------
    tuple of (cleaned_df, removal_stats)
        - cleaned_df: DataFrame with all specified outliers removed
        - removal_stats: Dict with removal counts per column
    
    Examples
    --------
    >>> config = {
    ...     'ALL_SumCurrentOutstandingBal': 3.0,
    ...     'Amount': 2.5
    ... }
    >>> clean_df, stats = remove_outliers_multi(df, config)
    """
    cleaned_df = df.copy()
    removal_stats = {}
    
    if verbose:
        print("Multi-Column Outlier Removal")
        print("=" * 60)
    
    for column, threshold in columns_config.items():
        if column not in cleaned_df.columns:
            if verbose:
                print(f"⚠️ Skipping '{column}' - not found in DataFrame")
            continue
        
        original_len = len(cleaned_df)
        cleaned_df, mask = remove_outliers(
            cleaned_df, 
            column=column, 
            z_threshold=threshold, 
            verbose=verbose
        )
        removed = original_len - len(cleaned_df)
        removal_stats[column] = {
            'threshold': threshold,
            'removed': removed,
            'pct': (removed / original_len) * 100 if original_len > 0 else 0
        }
        
        if verbose:
            print()
    
    if verbose:
        print("=" * 60)
        print("📊 SUMMARY")
        print(f"   Original rows: {len(df):,}")
        print(f"   Final rows:    {len(cleaned_df):,}")
        print(f"   Total removed: {len(df) - len(cleaned_df):,} "
              f"({(len(df) - len(cleaned_df))/len(df)*100:.2f}%)")
    
    return cleaned_df, removal_stats


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    np.random.seed(42)
    n = 1000
    
    # Create test data with outliers
    test_df = pd.DataFrame({
        'Amount': np.concatenate([
            np.random.normal(5000, 1000, n-5),
            [50000, 60000, 70000, 80000, 100000]  # Outliers
        ]),
        'Balance': np.concatenate([
            np.random.normal(10000, 3000, n-3),
            [200000, 300000, 500000]  # Outliers
        ]),
        'Age': np.random.normal(35, 10, n)  # No extreme outliers
    })
    
    print("=" * 60)
    print("TEST: describe_features()")
    print("=" * 60)
    stats = describe_features(test_df, columns=['Amount', 'Balance', 'Age'])
    
    print("\n" + "=" * 60)
    print("TEST: remove_outliers() - Single column")
    print("=" * 60)
    clean_df, mask = remove_outliers(test_df, column='Balance', z_threshold=3)
    
    print("\n" + "=" * 60)
    print("TEST: remove_outliers_multi()")
    print("=" * 60)
    config = {'Amount': 3.0, 'Balance': 3.0}
    final_df, stats = remove_outliers_multi(test_df, config)

