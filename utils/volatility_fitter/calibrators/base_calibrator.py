"""
Base calibrator module containing the abstract base class and common utilities.

This module provides the foundation for all volatility model calibrators with
shared functionality and defines the common interface.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import numpy as np

from config.volatility_config import VolatilityConfig
from ..calibration_result import CalibrationResult
from ..base_volatility_model_abstract import BaseVolatilityModel


class BaseVolatilityCalibrator(ABC):
    """
    Abstract base class for all volatility model calibrators.
    
    This class defines the common interface and shared functionality for both
    local and global optimization calibrators. Subclasses must implement the
    calibrate method with their specific optimization strategy.
    """
    
    def __init__(self, model_class: type, enable_bounds: bool = True, tolerance: float = 1e-6, arbitrage_penalty: float = 1e5,
                 max_iterations: int = 1000, config_loader: VolatilityConfig|None = None):
        """
        Initialize base calibrator with common parameters.
        
        Args:
            model_class: The volatility model class to calibrate
            enable_bounds: Whether to enable parameter bounds during optimization
            tolerance: Convergence tolerance for optimization
            arbitrage_penalty: Penalty weight for arbitrage violations
            max_iterations: Maximum number of optimization iterations
        """
        self.model_class = model_class
        self.enable_bounds = enable_bounds
        self.tolerance = tolerance
        self.arbitrage_penalty = arbitrage_penalty
        self.max_iterations = max_iterations
        self.config_loader = config_loader

    def _objective_function(self, x: np.ndarray, initial_params: Any, param_names: List[str],
                           strikes: List[float], market_volatilities: List[float], 
                           market_vegas: List[float], enforce_arbitrage_free: bool = True,
                           weights: Optional[List[float]] = None) -> float:
        """
        Shared objective function that calculates weighted RMSE with optional arbitrage penalty.
        
        This function is used by all calibrator subclasses to evaluate parameter fitness.
        
        Args:
            x: Parameter values to evaluate
            initial_params: Template parameter object for creating updated parameters
            param_names: Names of parameters being optimized
            strikes: Option strike prices
            market_volatilities: Market implied volatilities
            market_vegas: Market vegas for weighting
            enforce_arbitrage_free: Whether to add arbitrage penalty
            weights: Optional custom weights for strikes
            
        Returns:
            Objective value (lower is better)
        """
        try:
            # Create model and calculate volatilities
            updated_params = self._create_parameter_object(initial_params, param_names, x)
            model = self.model_class(updated_params)
            model_vols = np.array([model.calculate_volatility_from_strike(s) for s in strikes])
            
            # Calculate weights for RMSE
            market_vols, vegas = np.array(market_volatilities), np.array(market_vegas)
            
            if weights is None:
                weights = np.ones(len(market_vols))
            else:
                if len(weights) != len(market_vols):
                    raise ValueError(f"Length of weights ({len(weights)}) must match number of strikes ({len(market_vols)})")
            
            weighted_errors = ((model_vols - market_vols) * vegas * weights) ** 2           
            rmse = np.sqrt(np.mean(weighted_errors))
            
            # Add arbitrage penalty if enabled
            return rmse + (self._calculate_arbitrage_penalty(model) if enforce_arbitrage_free else 0)
            
        except Exception as e:
            print(f"Calibration error: {str(e)}")
            return 1e10

    def _create_parameter_object(self, initial_params: Any, param_names: List[str], optimized_values: np.ndarray) -> Any:
        """
        Create updated parameter object with optimized values.
        
        Args:
            initial_params: Template parameter object
            param_names: Names of parameters being optimized
            optimized_values: New parameter values from optimization
            
        Returns:
            New parameter object with updated values
        """
        param_dict = {name: value for name, value in zip(param_names, optimized_values)}
        
        # Copy non-fitted parameters if template has them
        for attr_name in dir(initial_params):
            if not attr_name.startswith('_') and attr_name not in param_names:
                attr_value = getattr(initial_params, attr_name)
                # Skip methods, functions, properties, and class attributes
                if not callable(attr_value):
                    param_dict[attr_name] = attr_value
        if self.config_loader is not None:
            param_dict['config'] = self.config_loader
        
        return type(initial_params)(**param_dict)

    def _calculate_arbitrage_penalty(self, model: BaseVolatilityModel) -> float:
        """
        Calculate penalty for arbitrage violations using Durrleman condition.
        
        Args:
            model: Volatility model to check for arbitrage
            
        Returns:
            Penalty value (0 if no arbitrage, positive penalty otherwise)
        """
        try:
            # Calculate Durrleman condition
            _, g_values = model.calculate_durrleman_condition()
            
            # Penalty for negative g values (butterfly arbitrage)
            negative_g = g_values[g_values < 0]
            if len(negative_g) > 0:
                return self.arbitrage_penalty * np.sum(negative_g ** 2)
            
            return 0.0
            
        except Exception:
            # If arbitrage calculation fails, apply penalty
            return self.arbitrage_penalty

    @abstractmethod
    def calibrate(self, initial_params: Any, strikes: List[float], market_volatilities: List[float],
                  market_vegas: List[float], parameter_bounds: Optional[List[Tuple[float, float]]] = None,
                  enforce_arbitrage_free: bool = True, additional_constraints: Optional[List] = None,
                  weights: Optional[List[float]] = None) -> CalibrationResult:
        """
        Abstract calibration method that must be implemented by subclasses.
        
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
        pass


class DEObjectiveFunction:
    """
    Pickleable wrapper for objective function used in Differential Evolution.
    
    This class wraps the objective function to make it pickleable for parallel
    processing in scipy.optimize.differential_evolution.
    """
    
    def __init__(self, calibrator: BaseVolatilityCalibrator, initial_params: Any, 
                 param_names: List[str], strikes: List[float], 
                 market_volatilities: List[float], market_vegas: List[float], 
                 enforce_arbitrage_free: bool = True, weights: Optional[List[float]] = None):
        """
        Initialize pickleable objective function wrapper.
        
        Args:
            calibrator: The calibrator instance with the objective function
            initial_params: Template parameter object
            param_names: Names of parameters being optimized
            strikes: Option strike prices
            market_volatilities: Market implied volatilities
            market_vegas: Market vegas for weighting
            enforce_arbitrage_free: Whether to enforce arbitrage-free conditions
            weights: Optional custom weights for strikes
        """
        self.calibrator = calibrator
        self.initial_params = initial_params
        self.param_names = param_names
        self.strikes = strikes
        self.market_volatilities = market_volatilities
        self.market_vegas = market_vegas
        self.enforce_arbitrage_free = enforce_arbitrage_free
        self.weights = weights
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Call the wrapped objective function.
        
        Args:
            x: Parameter values to evaluate
            
        Returns:
            Objective function value
        """
        return self.calibrator._objective_function(
            x, self.initial_params, self.param_names, self.strikes,
            self.market_volatilities, self.market_vegas, 
            self.enforce_arbitrage_free, self.weights
        )