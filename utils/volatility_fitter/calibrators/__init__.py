"""
Calibrators package for volatility model optimization.

This package provides a clean, organized approach to volatility model calibration
with specialized classes for different optimization strategies:

- BaseVolatilityCalibrator: Abstract base class with shared functionality
- LocalVolatilityCalibrator: Gradient-based local optimization (SLSQP, L-BFGS-B)  
- GlobalVolatilityCalibrator: Global optimization (Differential Evolution, Multi-start)

Usage Examples:
    # Choose the appropriate calibrator for your use case
    from utils.volatility_fitter.calibrators import LocalVolatilityCalibrator, GlobalVolatilityCalibrator
    
    # For local optimization (when you have good starting parameters)
    local_cal = LocalVolatilityCalibrator(MyModel, method="SLSQP")
    
    # For global optimization (when parameter space is complex)
    global_cal = GlobalVolatilityCalibrator(MyModel)
    
    # Import everything at once if needed
    from utils.volatility_fitter.calibrators import *
"""

from .base_calibrator import BaseVolatilityCalibrator, DEObjectiveFunction
from .local_calibrator import LocalVolatilityCalibrator
from .global_calibrator import GlobalVolatilityCalibrator

# Public API - these are the main classes users should import
__all__ = [
    'BaseVolatilityCalibrator',
    'LocalVolatilityCalibrator', 
    'GlobalVolatilityCalibrator',
    'DEObjectiveFunction',  # Exposed for advanced users who need it
]

# Convenience aliases for common usage patterns
LocalCalibrator = LocalVolatilityCalibrator
GlobalCalibrator = GlobalVolatilityCalibrator