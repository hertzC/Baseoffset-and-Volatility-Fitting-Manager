"""
Analytics Module

Regression analysis and optimization algorithms for put-call parity fitting.

Classes:
- WLSRegressor: Weighted Least Squares regression for rate extraction
- NonlinearMinimization: Constrained optimization with futures bounds
"""

import datetime
from enum import Enum
from typing import TypedDict

__all__ = ['WLSRegressor', 'NonlinearMinimization']


class FittingFailureReason(Enum):
    InsufficientStrike = 'insufficient_strikes_for_fitting_the_synthetic'
    OptimizationConstraint = 'Inequality constraints incompatible'
    OptimizationGradient = 'Positive directional derivative for linesearch'


class Result(TypedDict):
    """Result dictionary for regression fitting."""
    expiry: str
    timestamp: datetime.datetime
    S: float
    r: float           # USD interest rate
    q: float           # BTC funding rate
    tau: float         # time to expiry
    r2: float
    const: float
    coef: float        # regression coefficients
    sse: float         # sum of squared errors
    success_fitting: bool
    failure_reason: FittingFailureReason | None