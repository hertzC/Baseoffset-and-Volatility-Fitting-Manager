'''
@Time: 2024/4/9 2:32 PM
@Author: Jincheng Gong
@Contact: Jincheng.Gong@hotmail.com
@File: wing_model_calibrator.py
@Desc: Wing Model Calibrator for Parameter Optimization
'''

import random
from typing import List, Tuple
import numpy as np
from scipy import optimize

from .wing_model_parameters import WingModelParameters, CalibrationResult
from .wing_model import WingModel


class WingModelCalibrator:
    """Calibrator class for Wing Model parameter optimization"""
    
    def __init__(self, 
                 enable_bounds: bool = False,
                 tolerance: float = 1e-16,
                 method: str = "SLSQP",
                 butterfly_arbitrage_penalty: float = 1e5):
        """
        Initialize calibrator
        
        Args:
            enable_bounds: whether to enable parameter bounds
            tolerance: optimization tolerance
            method: optimization method
            butterfly_arbitrage_penalty: penalty for butterfly arbitrage violations
        """
        self.enable_bounds = enable_bounds
        self.tolerance = tolerance
        self.method = method
        self.butterfly_arbitrage_penalty = butterfly_arbitrage_penalty
        
    def calibrate(self,
                  fixed_params: WingModelParameters,
                  moneyness_list: List[float],
                  market_vol_list: List[float],
                  market_vega_list: List[float],
                  enforce_arbitrage_free: bool = True) -> CalibrationResult:
        """
        Calibrate wing model parameters
        
        Args:
            fixed_params: fixed wing model parameters (dc, uc, dsm, usm)
            moneyness_list: list of moneyness values
            market_vol_list: list of market implied volatilities
            market_vega_list: list of market vegas
            enforce_arbitrage_free: whether to enforce arbitrage-free conditions
            
        Returns:
            CalibrationResult object
        """
        # Initial guess for optimization parameters [vr, sr, pc, cc]
        initial_guess = [random.random() for _ in range(4)]
        
        # Set bounds if enabled
        bounds = self._get_parameter_bounds() if self.enable_bounds else None
        
        # Prepare arguments for loss function
        args = (fixed_params, moneyness_list, market_vol_list, 
               market_vega_list, enforce_arbitrage_free)
        
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
            optimal_params = WingModelParameters(
                vr=result.x[0],
                sr=result.x[1],
                pc=result.x[2],
                cc=result.x[3],
                dc=fixed_params.dc,
                uc=fixed_params.uc,
                dsm=fixed_params.dsm,
                usm=fixed_params.usm,
                vcr=fixed_params.vcr,
                scr=fixed_params.scr,
                ssr=fixed_params.ssr,
                atm=fixed_params.atm,
                ref=fixed_params.ref
            )
            
            calibration_result = CalibrationResult(
                success=result.success,
                parameters=optimal_params,
                error=result.fun,
                message=result.message if hasattr(result, 'message') else ""
            )
            
            return calibration_result
            
        except Exception as e:
            return CalibrationResult(
                success=False,
                parameters=fixed_params,
                error=float('inf'),
                message=str(e)
            )
    
    def _loss_function(self,
                      solve_params: List[float],
                      fixed_params: WingModelParameters,
                      moneyness_list: List[float],
                      market_vol_list: List[float],
                      market_vega_list: List[float],
                      enforce_arbitrage_free: bool = True) -> float:
        """
        Loss function for optimization
        
        Args:
            solve_params: parameters to optimize [vr, sr, pc, cc]
            fixed_params: fixed parameters
            moneyness_list: list of moneyness values
            market_vol_list: list of market implied volatilities
            market_vega_list: list of market vegas
            enforce_arbitrage_free: whether to enforce arbitrage-free conditions
            
        Returns:
            loss value
        """
        # Create wing model with current parameters
        current_params = WingModelParameters(
            vr=solve_params[0],
            sr=solve_params[1],
            pc=solve_params[2],
            cc=solve_params[3],
            dc=fixed_params.dc,
            uc=fixed_params.uc,
            dsm=fixed_params.dsm,
            usm=fixed_params.usm,
            vcr=fixed_params.vcr,
            scr=fixed_params.scr,
            ssr=fixed_params.ssr,
            atm=fixed_params.atm,
            ref=fixed_params.ref
        )
        
        wing_model = WingModel(current_params)
        
        # Calculate Mean Squared Error (MSE)
        max_vega = max(market_vega_list)
        squared_errors = []
        
        for i, moneyness in enumerate(moneyness_list):
            try:
                model_vol = wing_model.calculate_volatility(moneyness)
                weighted_error = ((model_vol - market_vol_list[i]) * 
                                market_vega_list[i] / max_vega) ** 2
                squared_errors.append(weighted_error)
            except Exception:
                # Return high penalty for invalid parameters
                return 1e10
        
        mse = (sum(squared_errors) ** 0.5) / len(moneyness_list)
        
        # Add butterfly arbitrage penalty if enabled
        arbitrage_penalty = 0.0
        if enforce_arbitrage_free:
            _, g_values = wing_model.calculate_durrleman_condition()
            if np.min(g_values) < 0:
                arbitrage_penalty = self.butterfly_arbitrage_penalty
        
        return mse + arbitrage_penalty
    
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        return [(-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3)]