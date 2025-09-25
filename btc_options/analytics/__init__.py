"""
Analytics Module

Regression analysis and optimization algorithms for put-call parity fitting.

Classes:
- WLSRegressor: Weighted Least Squares regression for rate extraction
- NonlinearMinimization: Constrained optimization with futures bounds
"""

from .weight_least_square_regressor import WLSRegressor
from .nonlinear_minimization import NonlinearMinimization

__all__ = ['WLSRegressor', 'NonlinearMinimization']