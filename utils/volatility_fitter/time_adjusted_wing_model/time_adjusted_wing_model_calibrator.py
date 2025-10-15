'''
@Time: 2024/10/15
@Author: Adapted for Time-Adjusted Wing Model
@Contact: 
@File: time_adjusted_wing_model_calibrator.py
@Desc: Time-Adjusted Wing Model Calibrator for Parameter Optimization
'''

import random
from typing import List, Tuple
import numpy as np
from scipy import optimize
from dataclasses import dataclass

from .time_adjusted_wing_model import TimeAdjustedWingModel, TimeAdjustedWingModelParameters


@dataclass
class TimeAdjustedCalibrationResult:
    """Data class to hold time-adjusted calibration results"""
    success: bool
    parameters: TimeAdjustedWingModelParameters
    error: float = 0.0
    message: str = ""


class TimeAdjustedWingModelCalibrator:
    """Calibrator class for Time-Adjusted Wing Model parameter optimization"""
    
    def __init__(self, 
                 enable_bounds: bool = True,
                 tolerance: float = 1e-16,
                 method: str = "SLSQP",
                 arbitrage_penalty: float = 1e5):
        """
        Initialize calibrator
        
        Args:
            enable_bounds: whether to enable parameter bounds
            tolerance: optimization tolerance
            method: optimization method
            arbitrage_penalty: penalty for arbitrage violations
        """
        self.enable_bounds = enable_bounds
        self.tolerance = tolerance
        self.method = method
        self.arbitrage_penalty = arbitrage_penalty
        
    def calibrate(self,
                  strike_list: List[float],
                  market_vol_list: List[float],
                  market_vega_list: List[float],
                  forward_price: float,
                  time_to_expiry: float,
                  initial_atm_vol: float = 0.8,
                  enforce_arbitrage_free: bool = True) -> TimeAdjustedCalibrationResult:
        """
        Calibrate time-adjusted wing model parameters
        
        Args:
            strike_list: list of strike prices
            market_vol_list: list of market implied volatilities (decimal form)
            market_vega_list: list of market vegas for weighting
            forward_price: forward price of the underlying
            time_to_expiry: time to expiry in years
            initial_atm_vol: initial guess for ATM volatility
            enforce_arbitrage_free: whether to enforce arbitrage-free conditions
            
        Returns:
            TimeAdjustedCalibrationResult object
        """
        # Initial guess for optimization parameters 
        # [atm_vol, slope, curve_up, curve_down, cut_up, cut_dn, mSmUp, mSmDn]
        initial_guess = [
            initial_atm_vol,    # atm_vol
            0.0,                # slope
            0.5,                # curve_up
            0.5,                # curve_down
            1.0,                # cut_up
            -1.0,               # cut_dn
            0.5,                # mSmUp
            0.5                 # mSmDn
        ]
        
        # Set bounds if enabled
        bounds = self._get_parameter_bounds() if self.enable_bounds else None
        
        # Prepare arguments for loss function
        args = (strike_list, market_vol_list, market_vega_list, 
               forward_price, time_to_expiry, enforce_arbitrage_free)
        
        try:
            # Run optimization
            result = optimize.minimize(
                fun=self._loss_function,
                x0=initial_guess,
                args=args,
                method=self.method,
                bounds=bounds,
                tol=self.tolerance
            )
            
            # Extract results
            optimal_params = TimeAdjustedWingModelParameters(
                atm_vol=result.x[0],
                slope=result.x[1],
                curve_up=result.x[2],
                curve_down=result.x[3],
                cut_up=result.x[4],
                cut_dn=result.x[5],
                mSmUp=result.x[6],
                mSmDn=result.x[7],
                forward_price=forward_price,
                time_to_expiry=time_to_expiry
            )
            
            calibration_result = TimeAdjustedCalibrationResult(
                success=result.success,
                parameters=optimal_params,
                error=result.fun,
                message=result.message if hasattr(result, 'message') else ""
            )
            
            return calibration_result
            
        except Exception as e:
            # Return failed result with default parameters
            default_params = TimeAdjustedWingModelParameters(
                atm_vol=initial_atm_vol, slope=0.0, curve_up=0.5, curve_down=0.5,
                cut_up=1.0, cut_dn=-1.0, mSmUp=0.5, mSmDn=0.5,
                forward_price=forward_price, time_to_expiry=time_to_expiry
            )
            
            return TimeAdjustedCalibrationResult(
                success=False,
                parameters=default_params,
                error=float('inf'),
                message=str(e)
            )
    
    def _loss_function(self,
                      solve_params: List[float],
                      strike_list: List[float],
                      market_vol_list: List[float],
                      market_vega_list: List[float],
                      forward_price: float,
                      time_to_expiry: float,
                      enforce_arbitrage_free: bool = True) -> float:
        """
        Loss function for optimization
        
        Args:
            solve_params: parameters to optimize
            strike_list: list of strike prices
            market_vol_list: list of market implied volatilities
            market_vega_list: list of market vegas
            forward_price: forward price
            time_to_expiry: time to expiry
            enforce_arbitrage_free: whether to enforce arbitrage-free conditions
            
        Returns:
            loss value
        """
        try:
            # Create wing model with current parameters
            current_params = TimeAdjustedWingModelParameters(
                atm_vol=solve_params[0],
                slope=solve_params[1],
                curve_up=solve_params[2],
                curve_down=solve_params[3],
                cut_up=solve_params[4],
                cut_dn=solve_params[5],
                mSmUp=solve_params[6],
                mSmDn=solve_params[7],
                forward_price=forward_price,
                time_to_expiry=time_to_expiry
            )
            
            wing_model = TimeAdjustedWingModel(current_params)
            
            # Calculate vega-weighted Mean Squared Error (MSE)
            max_vega = max(market_vega_list) if market_vega_list else 1.0
            squared_errors = []
            
            for i, strike in enumerate(strike_list):
                model_vol = wing_model.calculate_volatility_from_strike(strike)
                
                # Check for invalid volatilities
                if model_vol <= 0 or np.isnan(model_vol) or np.isinf(model_vol):
                    return 1e10
                
                # Vega-weighted error
                weight = market_vega_list[i] / max_vega if market_vega_list else 1.0
                weighted_error = ((model_vol - market_vol_list[i]) * weight) ** 2
                squared_errors.append(weighted_error)
            
            # Root mean squared error
            rmse = np.sqrt(np.mean(squared_errors))
            
            # Add arbitrage penalty if enabled
            arbitrage_penalty = 0.0
            if enforce_arbitrage_free:
                _, g_values = wing_model.calculate_durrleman_condition()
                if np.min(g_values) < 0:
                    arbitrage_penalty = self.arbitrage_penalty
            
            return rmse #+ arbitrage_penalty
            
        except Exception:
            # Return high penalty for invalid parameters
            return 1e10
    
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        return [
            (0.01, 5.0),    # atm_vol: reasonable volatility range
            (-2.0, 2.0),    # slope: reasonable skew range
            (-2.0, 2.0),    # curve_up: curvature bounds
            (-2.0, 2.0),    # curve_down: curvature bounds
            (0.1, 5.0),     # cut_up: positive cutoff
            (-5.0, -0.1),   # cut_dn: negative cutoff
            (0.1, 2.0),     # mSmUp: smoothing parameter
            (0.1, 2.0)      # mSmDn: smoothing parameter
        ]


def create_time_adjusted_wing_model_from_result(
    result: optimize.OptimizeResult,
    forward_price: float,
    time_to_expiry: float
) -> TimeAdjustedWingModelParameters:
    """
    Create TimeAdjustedWingModelParameters from optimization result
    
    Args:
        result: scipy optimization result
        forward_price: forward price
        time_to_expiry: time to expiry
        
    Returns:
        TimeAdjustedWingModelParameters object
    """
    return TimeAdjustedWingModelParameters(
        atm_vol=result.x[0],
        slope=result.x[1],
        curve_up=result.x[2],
        curve_down=result.x[3],
        cut_up=result.x[4],
        cut_dn=result.x[5],
        mSmUp=result.x[6],
        mSmDn=result.x[7],
        forward_price=forward_price,
        time_to_expiry=time_to_expiry
    )