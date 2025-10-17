'''
@Time: 2024/10/15
@Author: Adapted from ORC Wing Model Implementation
@Contact: 
@File: time_adjusted_wing_model_parameters.py
@Desc: Time-Adjusted Wing Model Parameters Data Class
'''

from dataclasses import dataclass, fields
import numpy as np


@dataclass
class TimeAdjustedWingModelParameters:
    """Data class to hold time-adjusted wing model parameters"""
    # Core volatility surface parameters
    atm_vol: float  # At-the-money volatility
    slope: float    # Controls the skew of the smile
    call_curve: float     # Controls the curvature for the upside
    put_curve: float   # Controls the curvature for the downside
    up_cutoff: float       # Moneyness threshold for the upside parabola
    down_cutoff: float       # Moneyness threshold for the downside parabola
    up_smoothing: float        # Smoothing factor for the upside wing
    down_smoothing: float        # Smoothing factor for the downside wing
    
    # Market context parameters
    forward_price: float  # Forward price of the underlying
    ref_price: float  # Reference forward price
    time_to_expiry: float  # Time to expiry in years

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names that are fitted during calibration"""
        return [field.name for field in fields(self) if field.name not in ['forward_price', 'ref_price', 'time_to_expiry']]

    def get_fitted_vol_parameter(self) -> list[float]:
        """Get list of parameter values that are fitted during calibration"""
        return [float(getattr(self, name)) for name in self.get_parameter_names()]
    
    def __repr__(self):
        """String representation of the parameters"""
        return (f"atm_vol={self.atm_vol:.4f}| slope={self.slope:.4f}| "
                f"CC={self.call_curve:.4f}| PC={self.put_curve:.4f}| "
                f"UCO={self.up_cutoff:.4f}| DCO={self.down_cutoff:.4f}| "
                f"USM={self.up_smoothing:.4f}| DSM={self.down_smoothing:.4f}\n"
                f"forward_price={self.forward_price:.2f}| ref_price={self.ref_price:.2f}| time_to_expiry={self.time_to_expiry:.4f}")


def create_time_adjusted_wing_model_from_result(result: np.ndarray|list, forward_price: float, ref_price: float, time_to_expiry: float):
    """
    Create TimeAdjustedWingModelParameters from optimization result
    
    Args:
        result: Optimization result containing parameter values
        forward_price: Forward price of the underlying
        ref_price: Reference forward price
        time_to_expiry: Time to expiry in years
        
    Returns:
        TimeAdjustedWingModelParameters instance
    """
    return TimeAdjustedWingModelParameters(
        atm_vol=result[0],
        slope=result[1],
        call_curve=result[2],
        put_curve=result[3],
        up_cutoff=result[4],
        down_cutoff=result[5],
        up_smoothing=result[6],
        down_smoothing=result[7],
        forward_price=forward_price,
        ref_price=ref_price,
        time_to_expiry=time_to_expiry
    )