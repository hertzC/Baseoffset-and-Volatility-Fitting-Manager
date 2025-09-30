"""
Weighted Least Squares Regressor for Bitcoin options put-call parity analysis.
Extracts USD and BTC interest rates from options pricing data.
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

    def fit(self, df: pl.DataFrame, **kwargs) -> Result:
        """
        Fit put-call parity regression to extract interest rates.
        
        Solves: P - C = K*exp(-r*t) - S*exp(-q*t)
        Linearized as: y = a*K + b, where a = exp(-r*t), b = -S*exp(-q*t)
        
        Args:
            df: Option synthetic data with columns: 'mid', 'strike', 'spread', 'S', 'tau'
            
        Returns:
            Result dictionary with fitted parameters and derived rates
        """
        if df.is_empty():
            raise ValueError("DataFrame is empty. Cannot fit model.")
            
        # Extract data and construct regression inputs
        y = df["mid"].to_numpy()
        X = df["strike"].to_numpy()
        weight = 1 / df["spread"].to_numpy()
        weight = weight * weight  # Square weights for WLS
        X_with_const = sm.add_constant(X)
        
        # Fit weighted least squares
        model = sm.WLS(y, X_with_const, weights=weight).fit()
        self.own_print(model.summary())
        
        # Convert parameters to financial rates
        const, coef = model.params[0], model.params[1]
        S, tau = df["S"][0], df["tau"][0]
        r2_adj, sse = float(model.rsquared_adj), float(model.ssr)
        
        return self._convert_to_result(const, coef, S, tau, r2_adj, sse)

    def _convert_to_result(self, const: float, coef: float, S: float, 
                          tau: float, r2_adj: float, sse: float) -> Result:
        """Convert regression parameters to financial rates and metrics."""
        # Calculate rates from regression parameters
        r = float(np.log(coef) / -tau)  # USD interest rate
        q = float(np.log(-const / S) / -tau)  # BTC funding rate
        
        # Calculate derived metrics
        discount_rate = float(np.exp((r - q) * tau))
        forward_price = round(discount_rate * S, 3)
        base_offset = forward_price - S
        
        return Result(
            S=S, F=forward_price, r=r, q=q, tau=tau,
            discount_rate=discount_rate, base_offset=base_offset,
            r2=r2_adj, const=float(const), coef=float(coef), sse=float(sse)
        )