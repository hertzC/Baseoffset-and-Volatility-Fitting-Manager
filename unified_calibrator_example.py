'''
@Time: 2025/10/17
@Author: Unified Calibrator Example
@Contact: 
@File: unified_calibrator_example.py
@Desc: Example demonstrating how to use the unified calibrator with different models
'''

import numpy as np
from utils.volatility_fitter import (
    UnifiedVolatilityCalibrator,
    TimeAdjustedWingModel,
    TimeAdjustedWingModelParameters,
    WingModel,
    WingModelParameters
)


def example_time_adjusted_wing_calibration():
    """Example of calibrating Time-Adjusted Wing Model using unified calibrator"""
    
    print("=== Time-Adjusted Wing Model Calibration Example ===")
    
    # Create sample market data
    forward_price = 100000.0
    time_to_expiry = 30/365.25
    
    strikes = [90000, 95000, 100000, 105000, 110000]
    market_vols = [0.8, 0.7, 0.6, 0.65, 0.75]
    market_vegas = [1.0, 1.2, 1.5, 1.2, 1.0]
    
    # Initial parameters
    initial_params = TimeAdjustedWingModelParameters(
        atm_vol=0.6,
        slope=-0.1,
        call_curve=0.05,
        put_curve=0.08,
        up_cutoff=0.5,
        down_cutoff=-0.5,
        up_smoothing=0.1,
        down_smoothing=0.1,
        forward_price=forward_price,
        time_to_expiry=time_to_expiry
    )
    
    # Parameter bounds (optional)
    bounds = [
        (0.1, 2.0),    # atm_vol
        (-1.0, 1.0),   # slope  
        (0.001, 0.5),  # call_curve
        (0.001, 0.5),  # put_curve
        (0.1, 2.0),    # up_cutoff
        (-2.0, -0.1),  # down_cutoff
        (0.01, 1.0),   # up_smoothing
        (0.01, 1.0),   # down_smoothing
    ]
    
    # Create unified calibrator for Time-Adjusted Wing Model
    calibrator = UnifiedVolatilityCalibrator(
        model_class=TimeAdjustedWingModel,
        enable_bounds=True,
        tolerance=1e-8,
        arbitrage_penalty=1e6
    )
    
    # Perform calibration
    result = calibrator.calibrate(
        initial_params=initial_params,
        strikes=strikes,
        market_volatilities=market_vols,
        market_vegas=market_vegas,
        parameter_bounds=bounds,
        enforce_arbitrage_free=True
    )
    
    print(f"Calibration Success: {result.success}")
    print(f"Final Error: {result.error:.6f}")
    print(f"Message: {result.message}")
    print(f"Optimized Parameters: {result.parameters}")
    
    return result


def example_wing_model_calibration():
    """Example of calibrating Traditional Wing Model using unified calibrator"""
    
    print("\n=== Traditional Wing Model Calibration Example ===")
    
    # Create sample market data  
    atm_price = 100000.0
    
    strikes = [90000, 95000, 100000, 105000, 110000]
    market_vols = [0.8, 0.7, 0.6, 0.65, 0.75]
    market_vegas = [1.0, 1.2, 1.5, 1.2, 1.0]
    
    # Initial parameters
    initial_params = WingModelParameters(
        vr=0.6,
        sr=-0.1, 
        pc=0.08,
        cc=0.05,
        dc=-0.5,
        uc=0.5,
        dsm=0.1,
        usm=0.1,
        atm=atm_price,
        ref_price=atm_price
    )
    
    # Parameter bounds
    bounds = [
        (0.1, 2.0),    # vr
        (-1.0, 1.0),   # sr
        (0.001, 0.5),  # pc  
        (0.001, 0.5),  # cc
        (-2.0, -0.1),  # dc
        (0.1, 2.0),    # uc
        (0.01, 1.0),   # dsm
        (0.01, 1.0),   # usm
    ]
    
    # Create unified calibrator for Wing Model
    calibrator = UnifiedVolatilityCalibrator(
        model_class=WingModel,
        enable_bounds=True,
        tolerance=1e-8,
        arbitrage_penalty=1e6
    )
    
    # Perform calibration
    result = calibrator.calibrate(
        initial_params=initial_params,
        strikes=strikes,
        market_volatilities=market_vols,
        market_vegas=market_vegas,
        parameter_bounds=bounds,
        enforce_arbitrage_free=True
    )
    
    print(f"Calibration Success: {result.success}")
    print(f"Final Error: {result.error:.6f}")
    print(f"Message: {result.message}")
    print(f"Optimized Parameters: {result.parameters}")
    
    return result


def compare_models():
    """Compare both models on the same market data"""
    
    print("\n=== Model Comparison ===")
    
    # Run both calibrations
    ta_result = example_time_adjusted_wing_calibration()
    wing_result = example_wing_model_calibration()
    
    print(f"\nComparison Summary:")
    print(f"Time-Adjusted Wing Model Error: {ta_result.error:.6f}")
    print(f"Traditional Wing Model Error: {wing_result.error:.6f}")
    
    if ta_result.error < wing_result.error:
        print("Time-Adjusted Wing Model performed better!")
    elif wing_result.error < ta_result.error:
        print("Traditional Wing Model performed better!")
    else:
        print("Both models performed equally well!")


if __name__ == "__main__":
    try:
        compare_models()
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()