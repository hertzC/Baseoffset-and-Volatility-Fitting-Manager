'''
@Time: 2025/10/17
@Author: Base Model Architecture
@Contact: 
@File: calibration_result.py
@Desc: Calibration Result Data Class
'''

from dataclasses import dataclass
from typing import Any, Optional
from scipy import optimize


@dataclass
class CalibrationResult:
    """Generic data class to hold calibration results for any volatility model"""
    success: bool
    optimization_method: str
    parameters: Any  # Will be the specific parameter class for each model
    time_elapsed: float = 0.0
    error: float = 0.0
    message: str = ""
    optimisation_result: Optional[optimize.OptimizeResult] = None
    
    def __repr__(self):
        return (
            f"CalibrationResult(\n"
            f"  success={self.success},\n"
            f"  method='{self.optimization_method}',\n"
            f"  parameters={self.parameters},\n" 
            f"  error={self.error:.6f},\n"
            f"  message='{self.message}'\n"
            f"  optimisation_result={self.optimisation_result}\n"
            f"  time_elapsed={self.time_elapsed:.6f} seconds\n"
            f")"
        )