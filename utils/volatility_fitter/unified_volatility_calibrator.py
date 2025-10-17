'''
@Time: 2025/10/17
@Author: Base Model Architecture
@Contact: 
@File: unified_volatility_calibrator.py
@Desc: Unified Volatility Calibrator for all models
'''

from typing import List, Tuple, Any, Optional
import numpy as np
from scipy import optimize
import random

from .calibration_result import CalibrationResult
from .base_volatility_model_abstract import BaseVolatilityModel


class UnifiedVolatilityCalibrator:
    """Unified calibrator that can work with any volatility model implementing BaseVolatilityModel"""
    
    def __init__(self, 
                 model_class: type,
                 enable_bounds: bool = True,
                 tolerance: float = 1e-16,
                 method: str = "SLSQP",
                 arbitrage_penalty: float = 1e5,
                 max_iterations: int = 1000):
        """
        Initialize unified calibrator
        
        Args:
            model_class: The volatility model class to calibrate
            enable_bounds: whether to enable parameter bounds
            tolerance: optimization tolerance
            method: optimization method
            arbitrage_penalty: penalty for arbitrage violations
            max_iterations: maximum optimization iterations
        """
        self.model_class = model_class
        self.enable_bounds = enable_bounds
        self.tolerance = tolerance
        self.method = method
        self.arbitrage_penalty = arbitrage_penalty
        self.max_iterations = max_iterations
        
    def calibrate(self,
                  initial_params: Any,
                  strikes: List[float],
                  market_volatilities: List[float],
                  market_vegas: List[float],
                  parameter_bounds: Optional[List[Tuple[float, float]]] = None,
                  enforce_arbitrage_free: bool = True,
                  additional_constraints: Optional[List] = None) -> CalibrationResult:
        """
        Calibrate volatility model parameters using unified approach
        
        Args:
            initial_params: Initial parameter object for the specific model
            strikes: List of strike prices
            market_volatilities: List of market implied volatilities
            market_vegas: List of market vegas for weighting
            parameter_bounds: Optional bounds for parameters
            enforce_arbitrage_free: Whether to enforce arbitrage-free constraints
            additional_constraints: Additional model-specific constraints
            
        Returns:
            CalibrationResult with optimized parameters
        """
        # Get parameter names and initial values from the parameter object
        param_names = initial_params.get_parameter_names()
        initial_values = initial_params.get_fitted_vol_parameter()
        
        # Set up bounds if enabled
        bounds = None
        if self.enable_bounds and parameter_bounds is not None:
            bounds = parameter_bounds
            
        # Set up constraints
        constraints = []
        if additional_constraints:
            constraints.extend(additional_constraints)
            
        # Define objective function
        def objective_function(x: np.ndarray) -> float:
            try:
                # Create parameter object with current values
                updated_params = self._create_parameter_object(initial_params, param_names, x)
                
                # Create model instance
                model = self.model_class(updated_params)
                
                # Calculate model volatilities
                model_vols = []
                for strike in strikes:
                    vol = model.calculate_volatility_from_strike(strike)
                    model_vols.append(vol)
                
                model_vols = np.array(model_vols)
                market_vols = np.array(market_volatilities)
                vegas = np.array(market_vegas)
                
                # Use vega-weighted RMSE (matching traditional calibrator exactly)
                squared_errors = []
                
                # Create normalized weights from vegas (matching traditional approach)
                weights = vegas / np.max(vegas)
                
                for i in range(len(strikes)):
                    # Double-weighted error: vegas AND normalized weights (matching traditional)
                    weighted_error = ((model_vols[i] - market_vols[i]) * vegas[i] * weights[i]) ** 2
                    squared_errors.append(weighted_error)
                
                # Root mean squared error
                rmse = np.sqrt(np.mean(squared_errors))
                
                # Add arbitrage penalty if enabled
                if enforce_arbitrage_free:
                    arbitrage_penalty = self._calculate_arbitrage_penalty(model)
                    rmse += arbitrage_penalty
                
                return rmse
                
            except Exception as e:
                # Return large error if calculation fails
                return 1e10
        
        # Perform optimization
        try:
            result = optimize.minimize(
                objective_function,
                initial_values,
                method=self.method,
                bounds=bounds,
                constraints=constraints,
                options={
                    'ftol': self.tolerance,
                    'maxiter': self.max_iterations
                }
            )
            
            if result.success:
                # Create final parameter object
                optimized_params = self._create_parameter_object(initial_params, param_names, result.x)
                
                return CalibrationResult(
                    success=True,
                    parameters=optimized_params,
                    error=result.fun,
                    message="Optimization successful",
                    optimisation_result=result
                )
            else:
                return CalibrationResult(
                    success=False,
                    parameters=initial_params,
                    error=result.fun if hasattr(result, 'fun') else float('inf'),
                    message=f"Optimization failed: {result.message}"                    
                )
                
        except Exception as e:
            return CalibrationResult(
                success=False,
                parameters=initial_params,
                error=float('inf'),
                message=f"Calibration error: {str(e)}"
            )
    
    def _create_parameter_object(self, template_params: Any, param_names: List[str], values: np.ndarray) -> Any:
        """
        Create a new parameter object with updated values
        
        Args:
            template_params: Template parameter object
            param_names: List of parameter names to update
            values: New parameter values
            
        Returns:
            Updated parameter object
        """
        # Create a copy of the template parameters
        param_dict = {}
        
        # Get all attributes from template
        for attr_name in dir(template_params):
            if not attr_name.startswith('_') and not callable(getattr(template_params, attr_name)):
                param_dict[attr_name] = getattr(template_params, attr_name)
        
        # Update with new values
        for name, value in zip(param_names, values):
            param_dict[name] = value
            
        # Create new parameter object
        return type(template_params)(**param_dict)
    
    def _calculate_arbitrage_penalty(self, model: BaseVolatilityModel) -> float:
        """
        Calculate penalty for arbitrage violations
        
        Args:
            model: The volatility model instance
            
        Returns:
            Arbitrage penalty value
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
    
    def calibrate_with_multiple_starts(self,
                                     initial_params: Any,
                                     strikes: List[float],
                                     market_volatilities: List[float],
                                     market_vegas: List[float],
                                     parameter_bounds: Optional[List[Tuple[float, float]]] = None,
                                     num_starts: int = 5,
                                     enforce_arbitrage_free: bool = True) -> CalibrationResult:
        """
        Calibrate with multiple random starting points to avoid local minima
        
        Args:
            initial_params: Initial parameter object
            strikes: List of strike prices
            market_volatilities: List of market implied volatilities
            market_vegas: List of market vegas for weighting
            parameter_bounds: Optional bounds for parameters
            num_starts: Number of random starting points
            enforce_arbitrage_free: Whether to enforce arbitrage-free constraints
            
        Returns:
            Best CalibrationResult from all attempts
        """
        best_result = None
        best_error = float('inf')
        
        for i in range(num_starts):
            # Create random starting point
            if i == 0:
                # First attempt uses provided initial parameters
                start_params = initial_params
            else:
                # Generate random starting parameters within bounds
                start_params = self._generate_random_start(initial_params, parameter_bounds)
            
            # Perform calibration
            result = self.calibrate(
                start_params,
                strikes,
                market_volatilities,
                market_vegas,
                parameter_bounds,
                enforce_arbitrage_free
            )
            
            # Keep best result
            if result.success and result.error < best_error:
                best_result = result
                best_error = result.error
        
        return best_result if best_result is not None else CalibrationResult(
            success=False,
            parameters=initial_params,
            error=float('inf'),
            message="All calibration attempts failed"
        )
    
    def _generate_random_start(self, template_params: Any, bounds: Optional[List[Tuple[float, float]]]) -> Any:
        """
        Generate random starting parameters within bounds
        
        Args:
            template_params: Template parameter object
            bounds: Parameter bounds
            
        Returns:
            Random parameter object
        """
        param_names = template_params.get_parameter_names()
        
        if bounds is None or len(bounds) != len(param_names):
            # If no bounds, use small random variations around initial values
            initial_values = template_params.get_fitted_vol_parameter()
            random_values = []
            for val in initial_values:
                # Add 10% random variation
                variation = val * 0.1 * (random.random() - 0.5) * 2
                random_values.append(val + variation)
        else:
            # Generate random values within bounds
            random_values = []
            for lower, upper in bounds:
                random_val = lower + random.random() * (upper - lower)
                random_values.append(random_val)
        
        return self._create_parameter_object(template_params, param_names, random_values)