"""
Local calibrator module for gradient-based optimization methods.

This module provides the LocalVolatilityCalibrator class which specializes in
local optimization methods like SLSQP and L-BFGS-B. These methods are efficient
for refining parameters when you have a good starting point.
"""

import time
from typing import Any, List, Optional, Tuple
from scipy import optimize

from config.volatility_config import VolatilityConfig

from .base_calibrator import BaseVolatilityCalibrator
from ..calibration_result import CalibrationResult


class LocalVolatilityCalibrator(BaseVolatilityCalibrator):
    """
    Calibrator specialized for local optimization methods.
    
    This calibrator uses gradient-based optimization methods that are efficient
    for local search around a starting point. Best used when you have good
    initial parameter estimates.
    
    Supported methods:
    - SLSQP: Sequential Least Squares Programming
    - L-BFGS-B: Limited-memory Broyden-Fletcher-Goldfarb-Shanno with bounds
    """
    
    def __init__(self, model_class: type, method: str = "SLSQP", enable_bounds: bool = True, tolerance: float = 1e-6,
                 arbitrage_penalty: float = 1e5, max_iterations: int = 1000, config_loader: VolatilityConfig|None = None):
        """
        Initialize local calibrator with specific optimization method.
        
        Args:
            model_class: The volatility model class to calibrate
            method: Local optimization method ("SLSQP" or "L-BFGS-B")
            enable_bounds: Whether to enable parameter bounds during optimization
            tolerance: Convergence tolerance for optimization
            arbitrage_penalty: Penalty weight for arbitrage violations
            max_iterations: Maximum number of optimization iterations
        """
        super().__init__(model_class, enable_bounds, tolerance, arbitrage_penalty, max_iterations, config_loader=config_loader)
        
        if method not in ["SLSQP", "L-BFGS-B"]:
            raise ValueError(f"LocalVolatilityCalibrator only supports SLSQP and L-BFGS-B methods, got: {method}")
        
        self.method = method

    def calibrate(self, initial_params: Any, strikes: List[float], market_volatilities: List[float],
                  market_vegas: List[float], parameter_bounds: Optional[List[Tuple[float, float]]] = None,
                  enforce_arbitrage_free: bool = True, additional_constraints: Optional[List] = None,
                  weights: Optional[List[float]] = None) -> CalibrationResult:
        """
        Calibrate volatility model parameters using local optimization.
        
        This method uses gradient-based optimization to find optimal parameters
        starting from the provided initial values. It's efficient but may get
        trapped in local minima.
        
        Args:
            initial_params: Starting parameter values
            strikes: Option strike prices
            market_volatilities: Market implied volatilities
            market_vegas: Market vegas for weighting
            parameter_bounds: Optional bounds for parameters
            enforce_arbitrage_free: Whether to enforce arbitrage-free conditions
            additional_constraints: Optional additional optimization constraints
            weights: Optional custom weights for strikes
            
        Returns:
            CalibrationResult with optimization outcome
        """
        param_names, initial_values = initial_params.get_parameter_names(), initial_params.get_fitted_vol_parameter()
        bounds = parameter_bounds if (self.enable_bounds and parameter_bounds) else None
        constraints = additional_constraints or []

        objective_wrapper = lambda x: self._objective_function(
            x, initial_params, param_names, strikes, market_volatilities, 
            market_vegas, enforce_arbitrage_free, weights
        )
        
        try:
            start_t = time.time_ns()
            result = optimize.minimize(
                objective_wrapper, 
                initial_values, 
                method=self.method, 
                bounds=bounds, 
                constraints=constraints, 
                options={'ftol': self.tolerance, 'maxiter': self.max_iterations}
            )
            end_t = time.time_ns()
            elapsed_time = (end_t - start_t) / 1_000_000_000  # Convert to seconds

            if result.success:
                optimized_params = self._create_parameter_object(initial_params, param_names, result.x)
                return CalibrationResult(
                    success=True, 
                    optimization_method=self.method, 
                    parameters=optimized_params, 
                    error=result.fun, 
                    message="Optimization successful", 
                    optimisation_result=result,
                    time_elapsed=elapsed_time
                )
            else:
                return CalibrationResult(
                    success=False, 
                    optimization_method=self.method, 
                    parameters=initial_params, 
                    error=getattr(result, 'fun', float('inf')), 
                    message=f"Optimization failed: {result.message}"
                )
                
        except Exception as e:
            return CalibrationResult(
                success=False, 
                optimization_method=self.method, 
                parameters=initial_params, 
                error=float('inf'), 
                message=f"Calibration error: {str(e)}"
            )