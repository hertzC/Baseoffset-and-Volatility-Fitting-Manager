"""
Time-Adjusted Wing Model Module

This module contains the time-adjusted wing model implementation with time-dependent moneyness calculation.
Now uses WingModelParameters for consistency.
"""

from .time_adjusted_wing_model import TimeAdjustedWingModel
from ..wing_model.wing_model_parameters import WingModelParameters, create_wing_model_from_result

# Create alias for backward compatibility
def create_time_adjusted_wing_model_from_result(result, forward_price: float, ref_price: float, time_to_expiry: float):
    """
    Create WingModelParameters for TimeAdjustedWingModel from optimization result
    
    Args:
        result: Optimization result containing parameter values [atm_vol, slope, call_curve, put_curve, up_cutoff, down_cutoff, up_smoothing, down_smoothing]
        forward_price: Forward price of the underlying
        ref_price: Reference forward price
        time_to_expiry: Time to expiry in years
        
    Returns:
        WingModelParameters instance with parameter mapping:
        atm_vol -> vr, slope -> sr, call_curve -> cc, put_curve -> pc,
        up_cutoff -> uc, down_cutoff -> dc, up_smoothing -> usm, down_smoothing -> dsm
    """
    return WingModelParameters(
        vr=result[0],      # atm_vol -> vr
        sr=result[1],      # slope -> sr
        pc=result[3],      # put_curve -> pc  
        cc=result[2],      # call_curve -> cc
        dc=result[5],      # down_cutoff -> dc
        uc=result[4],      # up_cutoff -> uc
        dsm=result[7],     # down_smoothing -> dsm
        usm=result[6],     # up_smoothing -> usm
        forward_price=forward_price,
        ref_price=ref_price,
        time_to_expiry=time_to_expiry
    )

__all__ = [
    'TimeAdjustedWingModel',
    'WingModelParameters',
    'create_time_adjusted_wing_model_from_result'
]