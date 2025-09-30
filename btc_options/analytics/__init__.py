"""
Analytics Module

Regression analysis and optimization algorithms for put-call parity fitting.

Classes:
- WLSRegressor: Weighted Least Squares regression for rate extraction
- NonlinearMinimization: Constrained optimization with futures bounds
"""

from typing import TypedDict

__all__ = ['WLSRegressor', 'NonlinearMinimization']


class Result(TypedDict):
    """Result dictionary for regression fitting."""
    S: float
    F: float           # fitted forward price
    r: float           # USD interest rate
    q: float           # BTC funding rate
    tau: float         # time to expiry
    discount_rate: float
    base_offset: float
    r2: float
    const: float
    coef: float        # regression coefficients
    sse: float         # sum of squared errors