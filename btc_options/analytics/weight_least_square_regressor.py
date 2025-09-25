"""
Weighted Least Squares Regressor

Implements weighted least squares regression for put-call parity analysis.
Extracts USD and BTC interest rates from Bitcoin options pricing data.
"""

from typing import TypedDict
import numpy as np
import polars as pl
import statsmodels.api as sm


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


class WLSRegressor:
    """Weighted Least Squares regressor for put-call parity analysis."""
    
    def __init__(self):
        self._can_print = False

    def set_printable(self, value: bool):
        """Enable/disable print output."""
        self._can_print = value

    def own_print(self, msg: str):
        """Print message if printing is enabled."""
        if self._can_print:
            print(msg)

    def construct_inputs(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct inputs for regression from option synthetic data.
        
        Args:
            df: Option synthetic DataFrame
            
        Returns:
            Tuple of (y, X_with_const, weights)
        """
        y = df["mid"].to_numpy()
        X = df["strike"].to_numpy()
        weight = 1 / df["spread"].to_numpy()
        weight *= weight  # Square weights for WLS
        X_with_const = sm.add_constant(X)
        return y, X_with_const, weight

    def fit(self, df: pl.DataFrame, prev_const: float = None, prev_coef: float = None) -> Result:
        """
        Fit put-call parity regression to extract interest rates.
        
        The regression solves: P - C = K*exp(-r*t) - S*exp(-q*t)
        Linearized as: y = a*K + b, where a = exp(-r*t), b = -S*exp(-q*t)
        
        Args:
            df: Option synthetic data
            prev_const: Previous constant (for warm starts)
            prev_coef: Previous coefficient (for warm starts)
            
        Returns:
            Result dictionary with fitted parameters and derived rates
        """
        if df.is_empty():
            raise ValueError("DataFrame is empty. Cannot fit model.")
            
        y, X_with_const, w = self.construct_inputs(df)
        model = sm.WLS(y, X_with_const, weights=w).fit()
        self.own_print(model.summary())
        
        return self.get_result_from_optimization(
            model.params[0], model.params[1], 
            df["S"][0], df["tau"][0], 
            float(model.rsquared_adj), float(model.ssr)
        )

    def get_result_from_optimization(self, const: float, coef: float, S: float, 
                                   tau: float, r2_adj: float, sse: float) -> Result:
        """
        Convert regression parameters to financial rates and metrics.
        
        Args:
            const: Regression constant (-S*exp(-q*t))
            coef: Regression coefficient (exp(-r*t))
            S: Spot price
            tau: Time to expiry
            r2_adj: Adjusted R-squared
            sse: Sum of squared errors
            
        Returns:
            Result dictionary with derived financial parameters
        """
        r = float(np.log(coef) / -tau)  # USD interest rate
        q = float(np.log(-const / S) / -tau)  # BTC funding rate
        discount_rate = float(np.exp((r - q) * tau))
        implied_F = round(discount_rate * S, 3)  # Forward price
        base_offset = implied_F - S
        
        return Result(
            S=S,
            F=implied_F,
            r=r,
            q=q,
            tau=tau,
            discount_rate=discount_rate,
            base_offset=base_offset,
            r2=r2_adj,
            const=float(const),
            coef=float(coef),
            sse=float(sse)
        )