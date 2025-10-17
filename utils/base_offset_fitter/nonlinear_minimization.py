"""
Nonlinear Minimization with Futures Constraints for Bitcoin options analysis.
Extends WLS regression with constrained optimization when futures data is available.
"""

from datetime import datetime, time
from typing import Any, Union
import numpy as np
import polars as pl
from utils.base_offset_fitter import Result
from utils.base_offset_fitter.fitter import Fitter
from utils.market_data.deribit_md_manager import DeribitMDManager
from utils.market_data.orderbook_deribit_md_manager import OrderbookDeribitMDManager
from utils.base_offset_fitter.maths import convert_paramter_into_rate
from config.config_loader import Config
from scipy.optimize import minimize


class NonlinearMinimization(Fitter):
    """Constrained optimization for put-call parity with futures bounds."""
    
    def __init__(self, 
                 symbol_manager: Union[DeribitMDManager, OrderbookDeribitMDManager],
                 config_loader: Config                 
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
        super().__init__(symbol_manager, config_loader)
        self.future_spread_mult = self.config_loader.future_spread_mult
        rate_constraints = self.config_loader.get_rate_constraints()
        self.r_min, self.r_max = rate_constraints['r_min'], rate_constraints['r_max']
        self.q_min, self.q_max = rate_constraints['q_min'], rate_constraints['q_max']
        self.minimum_rate, self.maximum_rate = rate_constraints['minimum_rate'], rate_constraints['maximum_rate']
        self.lambda_reg = self.config_loader.lambda_reg
        self.set_printable(self.config_loader.get('fitting.nonlinear.printable', False))
        
        # Store original parameters for reset functionality
        self._original_params.update({
            'future_spread_mult': self.future_spread_mult,
            'r_min': self.r_min, 'r_max': self.r_max,
            'q_min': self.q_min, 'q_max': self.q_max,
            'minimum_rate': self.minimum_rate, 'maximum_rate': self.maximum_rate,
            'lambda_reg': self.lambda_reg
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
        
        initial_rate = convert_paramter_into_rate(initial_guess, spot, tau)
        enough_strikes = (df.height >= self.config_loader.minimum_strikes)
        if enough_strikes:
            # Set up constraints
            constraints = self._build_constraints(spot, tau, future_bid, future_ask, has_futures)
            
            # Run optimization
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
            print(f"   ⚠️ {timestamp}: optimization failed on {expiry}, Error = {error_msg}; initial_guess = ({initial_guess[0]:.2f}, {initial_guess[1]:.6f}) (r={initial_rate[0]:.4f}, q={initial_rate[1]:.4f})")
            
            return Result(expiry=expiry,
                          timestamp=timestamp,
                          S=spot,
                          r=self.fit_results[-1]['r'] if self.fit_results else initial_rate[0],
                          q=self.fit_results[-1]['q'] if self.fit_results else initial_rate[1],
                          tau=tau,
                          const=self.fit_results[-1]['const'] if self.fit_results else initial_guess[0],
                          coef=self.fit_results[-1]['coef'] if self.fit_results else initial_guess[1],
                          r2=self.fit_results[-1]['r2'] if self.fit_results else 0.0,
                          sse=self.fit_results[-1]['sse'] if self.fit_results else 0.0,
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
            lb, ub = self._get_future_bounds(future_bid, future_ask, self.future_spread_mult)
            constraints.extend([
                {'type': 'ineq', 'fun': lambda p: -p[0] / p[1] - lb},
                {'type': 'ineq', 'fun': lambda p: ub - (-p[0] / p[1])},
            ])
            self.own_print(f"Future bounds: {lb:.2f} to {ub:.2f} from {future_bid:.2f}-{future_ask:.2f}")
        
        return constraints

    @staticmethod
    def _get_future_bounds(bid: float, ask: float, future_spread_mult: float) -> tuple[float, float]:
        """Create constraint boundaries based on futures prices."""
        mid = (bid + ask) / 2
        buffer = future_spread_mult / 2
        return min(bid, mid * (1 - buffer)), max(ask, mid * (1 + buffer))
        