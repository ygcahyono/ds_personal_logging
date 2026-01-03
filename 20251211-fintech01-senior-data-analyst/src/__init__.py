"""
Source utilities for data science projects.
"""

from .vif_iterate import iterative_vif_removal, get_clean_features, calculate_vif
from .outlier_analysis import describe_features, plot_distributions, remove_outliers, remove_outliers_multi
from .linearity_check import check_linearity

__all__ = [
    # VIF functions
    'iterative_vif_removal', 
    'get_clean_features', 
    'calculate_vif',
    # Outlier functions
    'describe_features',
    'plot_distributions',
    'remove_outliers',
    'remove_outliers_multi',
    # Linearity check
    'check_linearity'
]

