"""
Global calibrator module for global optimization methods.

This module provides the GlobalVolatilityCalibrator class which specializes in
global optimization methods like Differential Evolution and multi-start optimization.
These methods are robust to local minima but computationally more expensive.
"""

import time
from typing import Any, List, Optional, Tuple
from scipy import optimize

from .base_calibrator import BaseVolatilityCalibrator, DEObjectiveFunction
from ..calibration_result import CalibrationResult


class GlobalVolatilityCalibrator(BaseVolatilityCalibrator):
    """
    Calibrator specialized for global optimization methods.
    
    This calibrator uses Differential Evolution as the default global optimization
    method that can escape local minima and find better global solutions. More 
    computationally expensive but more robust when the parameter space is complex 
    or poorly understood.
    
    Default method:
    - Differential Evolution: Population-based global optimizer (default)
    """
    
    def __init__(self, model_class: type, enable_bounds: bool = True, 
                 tolerance: float = 1e-6, arbitrage_penalty: float = 1e5, 
                 max_iterations: int = 1000, workers: int = 1):
        """
        Initialize global calibrator.
        
        Args:
            model_class: The volatility model class to calibrate
            enable_bounds: Whether to enable parameter bounds during optimization
            tolerance: Convergence tolerance for optimization
            arbitrage_penalty: Penalty weight for arbitrage violations
            max_iterations: Maximum number of optimization iterations
            workers: Number of parallel workers for optimization
        """
        super().__init__(model_class, enable_bounds, tolerance, arbitrage_penalty, max_iterations)
        self.workers = workers

    def calibrate(self, initial_params: Any, strikes: List[float], market_volatilities: List[float],
                  market_vegas: List[float], parameter_bounds: Optional[List[Tuple[float, float]]] = None,
                  enforce_arbitrage_free: bool = True, **kwargs) -> CalibrationResult:
        """
        Default global calibration using Differential Evolution.
        
        This is the default method which uses Differential Evolution for robust global
        optimization. For other global methods, use the specific method directly.
        
        Args:
            initial_params: Starting parameter values
            strikes: Option strike prices
            market_volatilities: Market implied volatilities
            market_vegas: Market vegas for weighting
            parameter_bounds: Required bounds for parameters (DE needs bounds)
            enforce_arbitrage_free: Whether to enforce arbitrage-free conditions
            **kwargs: Additional arguments passed to calibrate_with_differential_evolution
            
        Returns:
            CalibrationResult with optimization outcome
        """
        if parameter_bounds is None:
            return CalibrationResult(
                success=False, 
                optimization_method="Differential Evolution", 
                parameters=initial_params, 
                error=float('inf'), 
                message="Global calibration requires parameter bounds for Differential Evolution"
            )
        
        return self.calibrate_with_differential_evolution(
            initial_params, strikes, market_volatilities, market_vegas,
            parameter_bounds, enforce_arbitrage_free=enforce_arbitrage_free, **kwargs
        )
    
    def calibrate_with_differential_evolution(self, initial_params: Any, strikes: List[float], 
                                            market_volatilities: List[float], market_vegas: List[float], 
                                            parameter_bounds: List[Tuple[float, float]], 
                                            enforce_arbitrage_free: bool = True, popsize: int = 15, 
                                            maxiter: int = 1000, seed: Optional[int] = None, 
                                            weights: Optional[List[float]] = None) -> CalibrationResult:
        """
        Calibrate using Differential Evolution (global optimization) to avoid local minima.
        
        Differential Evolution is a population-based optimization algorithm that is
        very robust for global optimization but requires parameter bounds.
        
        Args:
            initial_params: Starting parameter values (used for bounds if not provided)
            strikes: Option strike prices
            market_volatilities: Market implied volatilities
            market_vegas: Market vegas for weighting
            parameter_bounds: Required bounds for all parameters
            enforce_arbitrage_free: Whether to enforce arbitrage-free conditions
            popsize: Population size multiplier (actual size is popsize * dimensionality)
            maxiter: Maximum number of generations
            seed: Random seed for reproducibility
            weights: Optional custom weights for strikes
            
        Returns:
            CalibrationResult with optimization outcome
        """
        if parameter_bounds is None or len(parameter_bounds) == 0:
            return CalibrationResult(
                success=False, 
                optimization_method="Differential Evolution", 
                parameters=initial_params, 
                error=float('inf'), 
                message="Differential Evolution requires parameter bounds"
            )
            
        param_names = initial_params.get_parameter_names()
        
        # Create pickleable objective function wrapper for parallel processing
        objective_wrapper = DEObjectiveFunction(
            self, initial_params, param_names, strikes,
            market_volatilities, market_vegas, enforce_arbitrage_free, weights
        )
        
        try:
            start_t = time.time_ns()
            result = optimize.differential_evolution(
                objective_wrapper, 
                bounds=parameter_bounds, 
                popsize=popsize, 
                maxiter=maxiter, 
                seed=seed, 
                disp=False, 
                polish=True, 
                workers=self.workers 
            )
            end_t = time.time_ns()
            elapsed_time = (end_t - start_t) / 1_000_000_000  # Convert to seconds
            
            if result.success:
                optimized_params = self._create_parameter_object(initial_params, param_names, result.x)
                success_msg = f"Differential Evolution successful (nfev: {result.nfev})"
                return CalibrationResult(
                    success=True, 
                    optimization_method="Differential Evolution", 
                    parameters=optimized_params, 
                    error=float(result.fun), 
                    message=success_msg,
                    optimisation_result=result,
                    time_elapsed=elapsed_time
                )
            else:
                error_msg = f"Differential Evolution failed: {result.message}"
                return CalibrationResult(
                    success=False, 
                    optimization_method="Differential Evolution", 
                    parameters=initial_params, 
                    error=float(getattr(result, 'fun', float('inf'))), 
                    message=error_msg
                )
        except Exception as e:
            exception_msg = f"Differential Evolution error: {str(e)}"
            return CalibrationResult(
                success=False, 
                optimization_method="Differential Evolution", 
                parameters=initial_params, 
                error=float('inf'), 
                message=exception_msg
            )