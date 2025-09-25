"""
Bitcoin Options Analysis Library

This package provides comprehensive tools for analyzing Bitcoin options data from Deribit,
implementing put-call parity regression analysis and forward pricing extraction.

Modules:
- data_managers: Market data processing and option chain construction
- analytics: Regression analysis and optimization algorithms  
- visualization: Interactive plotting and table generation
"""

# Version info
__version__ = "1.0.0"
__author__ = "Bitcoin Options Analysis Team"

# Import key classes for easy access
from .data_managers.deribit_md_manager import DeribitMDManager
from .data_managers.orderbook_deribit_md_manager import OrderbookDeribitMDManager
from .analytics.weight_least_square_regressor import WLSRegressor
from .analytics.nonlinear_minimization import NonlinearMinimization
from .visualization.plotly_manager import PlotlyManager
from .visualization.html_table_generator import (
    generate_price_comparison_table, 
    calculate_tightening_stats, 
    print_tightening_effectiveness
)

__all__ = [
    'DeribitMDManager',
    'OrderbookDeribitMDManager', 
    'WLSRegressor',
    'NonlinearMinimization',
    'PlotlyManager',
    'generate_price_comparison_table',
    'calculate_tightening_stats',
    'print_tightening_effectiveness'
]