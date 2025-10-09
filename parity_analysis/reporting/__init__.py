"""
Visualization Module

Interactive plotting and table generation for Bitcoin options analysis results.

Classes:
- PlotlyManager: Interactive regression plots and visualizations

Functions:
- generate_price_comparison_table: HTML table generation for price analysis
- calculate_tightening_stats: Statistical analysis of spread tightening
- print_tightening_effectiveness: Summary reporting of tightening results
"""

from .plotly_manager import PlotlyManager
from .html_table_generator import (
    generate_price_comparison_table,
    calculate_tightening_stats, 
    print_tightening_effectiveness
)

__all__ = [
    'PlotlyManager',
    'generate_price_comparison_table',
    'calculate_tightening_stats',
    'print_tightening_effectiveness'
]