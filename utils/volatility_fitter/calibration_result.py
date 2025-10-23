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
    
    def to_dict(self) -> dict:
        return {'method': self.optimization_method,
                'success': self.success,
                'error': self.error,
                'message': self.message,
                'time_elapsed': self.time_elapsed,
                'num_evaluations': self.optimisation_result.nfev if self.optimisation_result else 0} | self.parameters.to_dict()