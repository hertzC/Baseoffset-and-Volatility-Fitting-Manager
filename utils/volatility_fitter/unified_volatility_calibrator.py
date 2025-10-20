'''
@Time: 2025/10/17
@Author: Base Model Architecture  
@File: unified_volatility_calibrator.py
@Desc: Unified Volatility Calibrator for all models
'''

from typing import List, Tuple, Any, Optional
import numpy as np
from scipy import optimize
import random
from dataclasses import fields, is_dataclass

from .calibration_result import CalibrationResult
from .base_volatility_model_abstract import BaseVolatilityModel


class UnifiedVolatilityCalibrator:
    """Unified calibrator that can work with any volatility model implementing BaseVolatilityModel"""
    
    def __init__(self, model_class: type, enable_bounds: bool = True, tolerance: float = 1e-16,
                 method: str = "SLSQP", arbitrage_penalty: float = 1e5, max_iterations: int = 1000):
        """Initialize unified calibrator"""
        self.model_class, self.enable_bounds, self.tolerance = model_class, enable_bounds, tolerance
        self.method, self.arbitrage_penalty, self.max_iterations = method, arbitrage_penalty, max_iterations
    
    def _objective_function(self, x: np.ndarray, initial_params: Any, param_names: List[str],
                           strikes: List[float], market_volatilities: List[float], 
                           market_vegas: List[float], enforce_arbitrage_free: bool = True) -> float:
        """Objective function for optimization that calculates vega-weighted RMSE"""
        try:
            # Create model and calculate volatilities
            updated_params = self._create_parameter_object(initial_params, param_names, x)
            model = self.model_class(updated_params)
            model_vols = np.array([model.calculate_volatility_from_strike(s) for s in strikes])
            
            # Vega-weighted RMSE calculation
            market_vols, vegas = np.array(market_volatilities), np.array(market_vegas)
            weights = vegas / np.max(vegas)
            weighted_errors = ((model_vols - market_vols) * vegas * weights) ** 2
            rmse = np.sqrt(np.mean(weighted_errors))
            
            # Add arbitrage penalty if enabled
            return rmse + (self._calculate_arbitrage_penalty(model) if enforce_arbitrage_free else 0)
            
        except Exception as e:
            print(f"Calibration error: {str(e)}")
            return 1e10
        
    def calibrate(self, initial_params: Any, strikes: List[float], market_volatilities: List[float],
                  market_vegas: List[float], parameter_bounds: Optional[List[Tuple[float, float]]] = None,
                  enforce_arbitrage_free: bool = True, additional_constraints: Optional[List] = None) -> CalibrationResult:
        """Calibrate volatility model parameters using unified approach"""
        param_names, initial_values = initial_params.get_parameter_names(), initial_params.get_fitted_vol_parameter()
        bounds = parameter_bounds if (self.enable_bounds and parameter_bounds) else None
        constraints = additional_constraints or []
        
        objective_wrapper = lambda x: self._objective_function(
            x, initial_params, param_names, strikes, market_volatilities, market_vegas, enforce_arbitrage_free)
        
        try:
            result = optimize.minimize(objective_wrapper, initial_values, method=self.method, bounds=bounds, 
                                     constraints=constraints, options={'ftol': self.tolerance, 'maxiter': self.max_iterations})
            
            if result.success:
                optimized_params = self._create_parameter_object(initial_params, param_names, result.x)
                return CalibrationResult(True, optimized_params, result.fun, "Optimization successful", result)
            else:
                return CalibrationResult(False, initial_params, getattr(result, 'fun', float('inf')), 
                                       f"Optimization failed: {result.message}")
        except Exception as e:
            return CalibrationResult(False, initial_params, float('inf'), f"Calibration error: {str(e)}")
    
    def _create_parameter_object(self, initial_params: Any, param_names: List[str], optimized_values: np.ndarray) -> Any:
        """Create updated parameter object with optimized values"""
        param_dict = {name: value for name, value in zip(param_names, optimized_values)}
        
        # Copy non-fitted parameters if template has them
        for attr_name in dir(initial_params):
            if not attr_name.startswith('_') and attr_name not in param_names:
                attr_value = getattr(initial_params, attr_name)
                # Skip methods, functions, properties, and class attributes
                if not callable(attr_value):
                    param_dict[attr_name] = attr_value
        
        return type(initial_params)(**param_dict)

    def _calculate_arbitrage_penalty(self, model: BaseVolatilityModel) -> float:
        """Calculate penalty for arbitrage violations"""
        try:
            _, g_values = model.calculate_durrleman_condition()
            negative_g = g_values[g_values < 0]
            return self.arbitrage_penalty * np.sum(negative_g ** 2) if len(negative_g) > 0 else 0.0
        except Exception:
            return self.arbitrage_penalty
    
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
    
    def calibrate_with_multiple_starts(self, initial_params: Any, strikes: List[float], market_volatilities: List[float],
                                     market_vegas: List[float], parameter_bounds: Optional[List[Tuple[float, float]]] = None,
                                     num_starts: int = 5, enforce_arbitrage_free: bool = True) -> CalibrationResult:
        """Calibrate with multiple random starting points to avoid local minima"""
        best_result, best_error = None, float('inf')
        
        for i in range(num_starts):
            start_params = initial_params if i == 0 else self._generate_random_start(initial_params, parameter_bounds)
            result = self.calibrate(start_params, strikes, market_volatilities, market_vegas, parameter_bounds, enforce_arbitrage_free)
            
            if result.success and result.error < best_error:
                best_result, best_error = result, result.error
        
        return best_result if best_result else CalibrationResult(False, initial_params, float('inf'), "All calibration attempts failed")
    
    def _generate_random_start(self, template_params: Any, bounds: Optional[List[Tuple[float, float]]]) -> Any:
        """Generate random starting parameters within bounds with strategic randomization"""
        param_names, initial_values = template_params.get_parameter_names(), template_params.get_fitted_vol_parameter()
        
        if bounds is None or len(bounds) != len(param_names):
            random_values = [val + val * (0.05 if name in ['vr', 'sr'] else 0.3 if name in ['dc', 'uc'] else 0.15) * 
                           (random.random() - 0.5) * 2 for name, val in zip(param_names, initial_values)]
        else:
            random_values = []
            for i, (name, (lower, upper)) in enumerate(zip(param_names, bounds)):
                if name in ['vr', 'sr']:
                    random_values.append(initial_values[i])
                elif name in ['dc', 'uc']:
                    random_values.append(lower + random.random() * (upper - lower))
                else:
                    range_size, center = (upper - lower) * 0.6, (upper + lower) / 2
                    min_val, max_val = max(lower, center - range_size/2), min(upper, center + range_size/2)
                    random_values.append(min_val + random.random() * (max_val - min_val))
        
        return self._create_parameter_object(template_params, param_names, random_values)
    
    def calibrate_with_differential_evolution(self, initial_params: Any, strikes: List[float], market_volatilities: List[float],
                                            market_vegas: List[float], parameter_bounds: List[Tuple[float, float]], 
                                            enforce_arbitrage_free: bool = True, popsize: int = 15, maxiter: int = 1000, 
                                            seed: Optional[int] = None) -> CalibrationResult:
        """Calibrate using Differential Evolution (global optimization) to avoid local minima"""
        if parameter_bounds is None:
            raise ValueError("Differential Evolution requires parameter bounds")
            
        param_names = initial_params.get_parameter_names()
        objective_wrapper = lambda x: self._objective_function(x, initial_params, param_names, strikes, 
                                                              market_volatilities, market_vegas, enforce_arbitrage_free)
        
        try:
            result = optimize.differential_evolution(objective_wrapper, bounds=parameter_bounds, popsize=popsize, 
                                                   maxiter=maxiter, seed=seed, disp=False, polish=True)
            
            if result.success:
                optimized_params = self._create_parameter_object(initial_params, param_names, result.x)
                return CalibrationResult(True, optimized_params, result.fun, 
                                       f"Differential Evolution successful (nfev: {result.nfev})", result)
            else:
                return CalibrationResult(False, initial_params, getattr(result, 'fun', float('inf')), 
                                       f"Differential Evolution failed: {result.message}")
        except Exception as e:
            return CalibrationResult(False, initial_params, float('inf'), f"Differential Evolution error: {str(e)}")

    def evaluate_parameters(self, params: Any, strikes: List[float], market_volatilities: List[float],
                           market_vegas: List[float], enforce_arbitrage_free: bool = True) -> Tuple[float, List[float]]:
        """Evaluate parameters and return both total error and fitted volatilities"""
        param_names, param_values = params.get_parameter_names(), params.get_fitted_vol_parameter()
        fitted_vols = [params.implied_volatility(strike) for strike in strikes]
        error = self._objective_function(param_values, params, param_names, strikes, 
                                       market_volatilities, market_vegas, enforce_arbitrage_free)
        return error, fitted_vols