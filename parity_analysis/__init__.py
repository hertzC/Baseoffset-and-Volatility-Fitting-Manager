"""
Bitcoin Options Parity Analysis Library

This package provides comprehensive tools for analyzing Bitcoin options data from Deribit,
implementing put-call parity regression analysis and forward pricing extraction.

Modules:
- market_data: Market data processing and option chain construction
- fitting: Regression analysis and optimization algorithms  
- reporting: Interactive plotting and table generation
"""

# Version info
__version__ = "1.0.0"
__author__ = "Bitcoin Options Analysis Team"

# Import key classes for easy access
from .market_data.deribit_md_manager import DeribitMDManager
from .market_data.orderbook_deribit_md_manager import OrderbookDeribitMDManager
from .fitting.weight_least_square_regressor import WLSRegressor
from .fitting.nonlinear_minimization import NonlinearMinimization
from .reporting.plotly_manager import PlotlyManager
from .reporting.html_table_generator import (
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