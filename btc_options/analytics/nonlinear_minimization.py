"""
Nonlinear Minimization with Futures Constraints

Extends WLS regression with constrained optimization when futures data is available.
Uses scipy.optimize to enforce forward price bounds from futures market.
"""

from typing import Any
import numpy as np
import polars as pl
from scipy.optimize import minimize
import scipy.optimize as opt

from .weight_least_square_regressor import WLSRegressor, Result


class NonlinearMinimization(WLSRegressor):
    """Constrained optimization for put-call parity with futures bounds."""
    
    def __init__(self, future_spread_mult: float = 0.0005, future_spread_threshold: float = 0.0020):
        """
        Initialize with futures constraint parameters.
        
        Args:
            future_spread_mult: Additional spread buffer for constraints
            future_spread_threshold: Maximum allowed futures spread (as fraction of spot)
        """
        super().__init__()
        self.future_spread_mult = future_spread_mult
        self.future_spread_threshold = future_spread_threshold    
        self.r_min, self.r_max = -0.01, 0.5
        self.q_min, self.q_max = -0.005, 0.1
        self.minimum_rate, self.maximum_rate = -0.005, 0.30  # on r-q

    def objective(self, params, X, y, weights):
        """Calculate weighted sum of squared residuals."""
        const, x1 = params
        residuals = y - (const + x1 * X[:, 1])
        return np.sum(weights * residuals**2)

    def create_future_boundaries(self, best_bid_price: float, best_ask_price: float) -> tuple[float, float]:
        """
        Create constraint boundaries based on futures prices.
        
        Args:
            best_bid_price: Futures bid price
            best_ask_price: Futures ask price
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mid_price = (best_bid_price + best_ask_price) / 2
        lb, ub = mid_price * np.array([1 - self.future_spread_mult / 2, 1 + self.future_spread_mult / 2])
        lower_bound = min(best_bid_price, lb)
        upper_bound = max(best_ask_price, ub)
        
        self.own_print(f"Constraint future bounds: {lower_bound:.2f} to {upper_bound:.2f} "
                      f"based on future price {best_bid_price:.2f} - {best_ask_price:.2f}")
        return lower_bound, upper_bound

    def fit(self, df: pl.DataFrame, prev_const: float, prev_coef: float) -> Result:
        """
        Fit constrained optimization with futures bounds when available.
        
        Args:
            df: Option synthetic data
            prev_const: Previous constant for warm start
            prev_coef: Previous coefficient for warm start
            
        Returns:
            Result dictionary with fitted parameters
        """
        if df.is_empty():
            raise ValueError("DataFrame is empty. Cannot fit model.")        

        initial_guess = np.array([prev_const, prev_coef])
        tau = df['tau'][0]
        spot_price = df['S'][0]
        best_future_bid_price = df['bid_price_fut'][0]
        best_future_ask_price = df['ask_price_fut'][0]

        is_future_expiry = best_future_bid_price is not None and best_future_ask_price is not None
        return self.minimize_error(df, initial_guess, spot_price, tau, best_future_bid_price, best_future_ask_price, is_future_expiry)

    def minimize_error(self, df: pl.DataFrame, initial_guess: np.ndarray, spot: float, tau: float,
                      future_best_bid: np.ndarray, future_best_ask: np.ndarray, is_future_expiry: bool) -> Result:
        """
        Perform the actual optimization.
        
        Args:
            df: Option synthetic data
            initial_guess: Starting parameters
            lower_bound: Lower constraint bound
            upper_bound: Upper constraint bound
            use_constraints: Whether to apply constraints
            
        Returns:
            Result dictionary
        """
        y, X_with_const, weight = self.construct_inputs(df)
        # constraint functions for interest rate and funding rate bounds from the slope (= exp(-r * T) and constant (= S*exp(-q * T))).
        constraints = [
            {'type': 'ineq', 'fun': lambda params: -np.log(params[1]) - self.r_min * tau},  # Ensure r > r_min
            {'type': 'ineq', 'fun': lambda params: np.log(params[1]) + self.r_max * tau},  # Ensure r < r_max
            {'type': 'ineq', 'fun': lambda params: -np.log(-params[0] / spot) - self.q_min * tau},  # Ensure q > q_min
            {'type': 'ineq', 'fun': lambda params: np.log(-params[0] / spot) + self.q_max * tau},  # Ensure q < q_max
            {'type': 'ineq', 'fun': lambda params: -np.log(params[1]) + np.log(-params[0] / spot) - self.minimum_rate * tau},  # Ensure r - q >= minimum_rate
            {'type': 'ineq', 'fun': lambda params: np.log(params[1]) - np.log(-params[0] / spot) + self.maximum_rate * tau},   # Ensure r - q <= maximum_rate
        ]
        if is_future_expiry:
            lower_bound, upper_bound = self.create_future_boundaries(future_best_bid, future_best_ask)
            constraints += [
                {'type': 'ineq', 'fun': lambda params: -params[0] / params[1] - lower_bound},
                {'type': 'ineq', 'fun': lambda params: upper_bound - (-params[0] / params[1])},
            ]
        
        result = minimize(
                fun=self.objective,
                x0=initial_guess,
                args=(X_with_const, y, weight),
                method='SLSQP',
                constraints=constraints
            )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        const, coef = result.x
        self.own_print("Optimization successful.")
        self.own_print(f"Optimal parameters (const, coef): {const:.6f}, {coef:.6f}")
        self.own_print(f"Optimal objective value (SSE): {result.fun:.4f}")
        
        # Calculate R-squared approximation
        residuals = y - (const + coef * X_with_const[:, 1])
        sse = np.sum(weight * residuals**2)
        y_weighted_mean = np.sum(weight * y) / np.sum(weight)
        sst = np.sum(weight * (y - y_weighted_mean)**2)
        r_squared = 1 - (sse / sst)
        
        return self.get_result_from_optimization(
            const, coef, df["S"][0], df["tau"][0], float(r_squared), float(sse)
        )