"""
Nonlinear Minimization with Futures Constraints for Bitcoin options analysis.
Extends WLS regression with constrained optimization when futures data is available.
"""

from datetime import datetime, time
from typing import Any, Union
import numpy as np
import polars as pl
from btc_options.analytics import Result
from btc_options.analytics.fitter import Fitter
from btc_options.data_managers.deribit_md_manager import DeribitMDManager
from btc_options.data_managers.orderbook_deribit_md_manager import OrderbookDeribitMDManager
from btc_options.analytics.maths import convert_paramter_into_rate
from scipy.optimize import minimize


class NonlinearMinimization(Fitter):
    """Constrained optimization for put-call parity with futures bounds."""
    
    def __init__(self, 
                 symbol_manager: Union[DeribitMDManager, OrderbookDeribitMDManager],
                 future_spread_mult: float = 0.0005, 
                 future_spread_threshold: float = 0.0020,
                 r_min: float = -0.01, 
                 r_max: float = 0.5,
                 q_min: float = -0.005, 
                 q_max: float = 0.1,
                 minimum_rate: float = -0.005, 
                 maximum_rate: float = 0.30,
                 minimum_strikes: int = 5, 
                 lambda_reg: float = 0.00,  # control the regularization
                 cutoff_time_for_0DTE: time = time(hour=4)
                 ):
        """
        Initialize with futures constraint parameters and rate bounds.
        
        Args:
            future_spread_mult: Multiplier for futures spread constraints
            future_spread_threshold: Threshold for futures spread validation
            r_min: Minimum USD interest rate
            r_max: Maximum USD interest rate  
            q_min: Minimum BTC funding rate
            q_max: Maximum BTC funding rate
            minimum_rate: Minimum rate spread (r-q)
            maximum_rate: Maximum rate spread (r-q)
        """
        super().__init__(symbol_manager, minimum_strikes, cutoff_time_for_0DTE)
        self.future_spread_mult = future_spread_mult
        self.future_spread_threshold = future_spread_threshold    
        self.r_min, self.r_max = r_min, r_max
        self.q_min, self.q_max = q_min, q_max
        self.minimum_rate, self.maximum_rate = minimum_rate, maximum_rate
        self.lambda_reg = lambda_reg
        
        # Store original parameters for reset functionality
        self._original_params.update({
            'future_spread_mult': future_spread_mult,
            'future_spread_threshold': future_spread_threshold,
            'r_min': r_min, 'r_max': r_max,
            'q_min': q_min, 'q_max': q_max,
            'minimum_rate': minimum_rate, 'maximum_rate': maximum_rate,
            'lambda_reg': lambda_reg
        })

    def fit(self, df: pl.DataFrame, prev_const: float, prev_coef: float, **kwargs) -> Result:
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
        tau, spot = df['tau'][0], df['S'][0]
        expiry, timestamp = kwargs['expiry'], kwargs['timestamp']
        is_cutoff, result = self.check_if_cutoff_for_0DTE(expiry, timestamp, self.symbol_manager.is_expiry_today(expiry), spot, tau)
        
        if not is_cutoff:
            # Check if futures data is available
            has_futures = 'bid_price_fut' in df.columns and 'ask_price_fut' in df.columns
            if has_futures:
                future_bid, future_ask = df['bid_price_fut'][0], df['ask_price_fut'][0]
                has_futures = future_bid is not None and future_ask is not None
            else:
                future_bid = future_ask = None

            result = self._optimize(expiry, timestamp, df, initial_guess, spot, tau, future_bid, future_ask, has_futures)
        self.fit_results.append(result)
        return result

    def _optimize(self, expiry: str, timestamp: datetime, df: pl.DataFrame, initial_guess: np.ndarray, spot: float, tau: float,
                 future_bid: float, future_ask: float, has_futures: bool) -> Result:
        """Perform the actual optimization with constraints."""
        def objective(params, x, y, weight, past_params, lambda_reg, spot, tau):
            const, coef = params
            optimized_rate = convert_paramter_into_rate(params, spot, tau)
            residuals = y - (const + coef * x)
            sse =  np.sum(weight * residuals**2)
            penalty = lambda_reg * np.sum((optimized_rate - past_params)**2)
            # print(f"constant={const:10.2f} coefficient={coef:8.4f} r={optimized_rate[0]:10.4f} q={optimized_rate[1]:.4f} sse={sse:18.4f} penalty={penalty:12.4f}")
            return sse + penalty
        
        enough_strikes = (df.height >= self.minimum_strikes)
        if enough_strikes:
            # Set up constraints
            constraints = self._build_constraints(spot, tau, future_bid, future_ask, has_futures)
            
            # Run optimization
            initial_rate = convert_paramter_into_rate(initial_guess, spot, tau)
            result = minimize(fun=objective, 
                              x0=initial_guess, 
                              args=((X:=df['strike'].to_numpy()),
                                    (Y:=df['mid'].to_numpy()),
                                    (weight:= (1 / df['spread'].to_numpy())**2),
                                    initial_rate,
                                    self.lambda_reg ,
                                    spot,
                                    tau),
                              method='SLSQP', 
                              constraints=constraints)
        
        if not enough_strikes or not result.success:
            error_msg = result.message if enough_strikes else "insufficient strikes"
            print(f"   ⚠️ {timestamp}: optimization failed on {expiry}, Error = {error_msg}; initial_guess = ({initial_guess[0]:.2f}, {initial_guess[1]:.6f}) (r={initial_guess[0]:.4f}, q={initial_guess[1]:.4f})")
            return Result(expiry=expiry,
                          timestamp=timestamp,
                          S=spot,
                          r=self.fit_results[-1]['r'],
                          q=self.fit_results[-1]['q'],
                          tau=tau,
                          const=self.fit_results[-1]['const'],
                          coef=self.fit_results[-1]['coef'],
                          r2=self.fit_results[-1]['r2'],
                          sse=self.fit_results[-1]['sse'],
                          success_fitting=False,
                          failure_reason=error_msg
                          )
        
        const, coef = result.x
        self.own_print("Optimization successful.")
        self.own_print(f"Optimal parameters (const, coef): {const:.6f}, {coef:.6f}")
        
        # Calculate rates from parameters for logging
        r, q = convert_paramter_into_rate(result.x, spot, tau)
        self.own_print(f"Optimal parameters (r, q): {r:.6f}, {q:.6f}")
        self.own_print(f"Optimal objective value (SSE): {result.fun:.4f}")
        
        # Calculate R-squared
        residuals = Y - (const + coef * X)
        sse = np.sum(weight * residuals**2)
        y_weighted_mean = np.sum(weight * Y) / np.sum(weight)
        sst = np.sum(weight * (Y - y_weighted_mean)**2)
        r_squared = 1 - (sse / sst)
        
        return self._convert_to_result(expiry, timestamp, result.x, spot, tau, float(r_squared), float(sse))

    def _build_constraints(self, spot: float, tau: float, future_bid: float, 
                          future_ask: float, has_futures: bool) -> list:
        """Build optimization constraints for rates and futures bounds."""
        constraints = [
            # Rate bounds: r_min < r < r_max
            {'type': 'ineq', 'fun': lambda p: -np.log(p[1]) - self.r_min * tau},
            {'type': 'ineq', 'fun': lambda p: np.log(p[1]) + self.r_max * tau},
            # Funding rate bounds: q_min < q < q_max  
            {'type': 'ineq', 'fun': lambda p: -np.log(-p[0] / spot) - self.q_min * tau},
            {'type': 'ineq', 'fun': lambda p: np.log(-p[0] / spot) + self.q_max * tau},
            # Rate spread bounds: minimum_rate < r-q < maximum_rate
            {'type': 'ineq', 'fun': lambda p: -np.log(p[1]) + np.log(-p[0] / spot) - self.minimum_rate * tau},
            {'type': 'ineq', 'fun': lambda p: np.log(p[1]) - np.log(-p[0] / spot) + self.maximum_rate * tau},
        ]
        
        # Add futures constraints if available
        if has_futures:
            lb, ub = self._get_future_bounds(future_bid, future_ask)
            constraints.extend([
                {'type': 'ineq', 'fun': lambda p: -p[0] / p[1] - lb},
                {'type': 'ineq', 'fun': lambda p: ub - (-p[0] / p[1])},
            ])
        
        return constraints

    def _get_future_bounds(self, bid: float, ask: float) -> tuple[float, float]:
        """Create constraint boundaries based on futures prices."""
        mid = (bid + ask) / 2
        buffer = self.future_spread_mult / 2
        lb = min(bid, mid * (1 - buffer))
        ub = max(ask, mid * (1 + buffer))
        
        self.own_print(f"Future bounds: {lb:.2f} to {ub:.2f} from {bid:.2f}-{ask:.2f}")
        return lb, ub