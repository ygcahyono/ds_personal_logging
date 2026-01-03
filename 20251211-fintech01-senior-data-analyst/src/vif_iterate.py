"""
Iterative VIF (Variance Inflation Factor) Removal Module

This module provides functions to detect and iteratively remove features 
with high multicollinearity from a DataFrame, recalculating VIF after 
each removal to ensure accurate assessment.

Usage:
    from src.vif_iterate import iterative_vif_removal
    
    cleaned_df, removed, vif_table = iterative_vif_removal(
        df=my_dataframe,
        target_col='Success',
        id_cols=['UID'],
        vif_threshold=10
    )
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple, Optional


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VIF for all features in a DataFrame.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric columns only)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with Feature names and VIF values, sorted descending
    """
    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def vif_status(vif: float, threshold: float = 10) -> str:
    """
    Return status label based on VIF value.
    
    Parameters
    ----------
    vif : float
        VIF value
    threshold : float
        VIF threshold for "High" status
    
    Returns
    -------
    str
        Status label with emoji
    """
    if vif > threshold:
        return "❌ High - Remove"
    elif vif > 5:
        return "⚠️ Moderate"
    else:
        return "✅ OK"


def iterative_vif_removal(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
    vif_threshold: float = 10.0,
    verbose: bool = True,
    max_iterations: int = 100
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Iteratively remove features with high VIF, one at a time.
    
    VIF values change when features are removed, so this function removes 
    the highest VIF feature, recalculates, and repeats until all features 
    are below the threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features
    target_col : str, optional
        Name of target/response variable to exclude from VIF calculation
    id_cols : list of str, optional
        List of ID columns to exclude (e.g., ['UID', 'customer_id'])
    vif_threshold : float, default=10.0
        VIF threshold above which features are removed
    verbose : bool, default=True
        Print iteration details
    max_iterations : int, default=100
        Safety limit to prevent infinite loops
    
    Returns
    -------
    tuple of (final_features, removed_features, vif_table)
        - final_features: List of feature names that passed VIF check
        - removed_features: List of feature names removed (in order)
        - vif_table: Final VIF DataFrame with Status column
    
    Examples
    --------
    >>> final_features, removed, vif_df = iterative_vif_removal(
    ...     df=loan_data,
    ...     target_col='Success',
    ...     id_cols=['UID'],
    ...     vif_threshold=10
    ... )
    >>> print(f"Removed {len(removed)} features: {removed}")
    >>> X_clean = loan_data[final_features]
    """
    # Determine columns to exclude
    exclude_cols = []
    if target_col:
        exclude_cols.append(target_col)
    if id_cols:
        exclude_cols.extend(id_cols)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Validate: ensure all feature columns are numeric
    non_numeric = df[feature_cols].select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns found: {non_numeric}. "
            "Encode categorical variables before running VIF analysis."
        )
    
    removed_features = []
    iteration = 0
    
    if verbose:
        print("Iterative VIF Analysis - Removing One Feature at a Time")
        print("=" * 70)
        print(f"Threshold: VIF > {vif_threshold}")
        print(f"Starting features: {len(feature_cols)}")
        print("=" * 70)
    
    while iteration < max_iterations:
        iteration += 1
        current_features = [col for col in feature_cols if col not in removed_features]
        
        if len(current_features) <= 1:
            if verbose:
                print(f"\n⚠️ Only {len(current_features)} feature(s) remaining. Stopping.")
            break
        
        X_current = df[current_features]
        vif_data = calculate_vif(X_current)
        
        max_vif = vif_data["VIF"].max()
        max_vif_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        
        if max_vif <= vif_threshold:
            if verbose:
                print(f"\n✅ Iteration {iteration}: All VIF values now ≤ {vif_threshold}")
            break
        
        if verbose:
            print(f"\n🔄 Iteration {iteration}: Highest VIF = {max_vif:,.2f} ({max_vif_feature})")
        
        removed_features.append(max_vif_feature)
        
        if verbose:
            print(f"   ❌ Removed: {max_vif_feature}")
    
    else:
        if verbose:
            print(f"\n⚠️ Reached max iterations ({max_iterations}). Stopping.")
    
    # Final results
    final_features = [col for col in feature_cols if col not in removed_features]
    
    # Recalculate final VIF table
    vif_data = calculate_vif(df[final_features])
    vif_data["Status"] = vif_data["VIF"].apply(lambda x: vif_status(x, vif_threshold))
    
    if verbose:
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"Features removed ({len(removed_features)}):")
        for i, feat in enumerate(removed_features, 1):
            print(f"   {i}. {feat}")
        print(f"\nRemaining features: {len(final_features)}")
    
    return final_features, removed_features, vif_data


def get_clean_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
    vif_threshold: float = 10.0
) -> pd.DataFrame:
    """
    Convenience function that returns cleaned DataFrame directly.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, optional
        Target variable column name
    id_cols : list of str, optional
        ID columns to preserve but exclude from VIF
    vif_threshold : float, default=10.0
        VIF threshold
    
    Returns
    -------
    pd.DataFrame
        DataFrame with only the features that passed VIF check,
        plus target_col and id_cols if specified
    """
    final_features, _, _ = iterative_vif_removal(
        df=df,
        target_col=target_col,
        id_cols=id_cols,
        vif_threshold=vif_threshold,
        verbose=False
    )
    
    # Include ID and target columns in output
    output_cols = []
    if id_cols:
        output_cols.extend(id_cols)
    output_cols.extend(final_features)
    if target_col:
        output_cols.append(target_col)
    
    return df[output_cols]


if __name__ == "__main__":
    # Example usage / quick test
    import numpy as np
    
    # Create sample data with multicollinearity
    np.random.seed(42)
    n = 1000
    
    x1 = np.random.normal(0, 1, n)
    x2 = x1 * 0.9 + np.random.normal(0, 0.1, n)  # Highly correlated with x1
    x3 = np.random.normal(0, 1, n)
    x4 = x3 * 0.8 + np.random.normal(0, 0.2, n)  # Correlated with x3
    x5 = np.random.normal(0, 1, n)  # Independent
    y = (x1 + x3 + x5 + np.random.normal(0, 0.5, n) > 0).astype(int)
    
    test_df = pd.DataFrame({
        'id': range(n),
        'feature_1': x1,
        'feature_2': x2,
        'feature_3': x3,
        'feature_4': x4,
        'feature_5': x5,
        'target': y
    })
    
    print("Testing iterative_vif_removal()...\n")
    final, removed, vif_table = iterative_vif_removal(
        df=test_df,
        target_col='target',
        id_cols=['id'],
        vif_threshold=5
    )
    
    print("\nFinal VIF Table:")
    print(vif_table.to_string(index=False))

