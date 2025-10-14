"""
Common utilities for the Baseoffset-Fitting-Manager project.

This package contains shared components for quantitative finance analysis,
including option pricing models, regression analysis, and data management.

Available modules:
- black76_option_pricer: Black-76 option pricing model for forward contracts
- deribit_md_manager: Market data processing for Deribit options
- weight_least_square_regressor: Weighted least squares regression
- nonlinear_minimization: Constrained optimization for put-call parity
- plotly_manager: Interactive visualization and plotting utilities
"""

# Import key classes for easy access
from .black76_option_pricer import Black76OptionPricer
from .pricer_helper import (
    black76_call,
    black76_put,
    black76_vega,
    find_implied_volatility,
    find_vol  # Alias for backward compatibility
)

__all__ = [
    'Black76OptionPricer',
    'black76_call',
    'black76_put', 
    'black76_vega',
    'find_implied_volatility',
    'find_vol'
]

__version__ = "1.0.0"