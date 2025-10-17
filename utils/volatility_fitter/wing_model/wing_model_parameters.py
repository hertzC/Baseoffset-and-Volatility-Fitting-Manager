from dataclasses import dataclass, fields
import numpy as np


@dataclass
class WingModelParameters:
    """Data class to hold wing model parameters"""
    # Solve parameters
    vr: float  # volatility reference
    sr: float  # slope reference
    pc: float  # put curvature
    cc: float  # call curvature
    dc: float  # down cutoff
    uc: float  # up cutoff
    dsm: float  # down smoothing range
    usm: float  # up smoothing range

    forward_price: float  # forward price of the underlying
    ref_price: float  # reference forward price
    time_to_expiry: float  # time to expiry in years

    # Optional parameters
    vcr: float = 0.0  # volatility change rate
    scr: float = 0.0  # slope change rate
    ssr: float = 100.0  # skew swimmingness rate
    
    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names that are fitted during calibration"""
        return [field.name for field in fields(self) if field.name not in ['forward_price', 'ref_price', 'time_to_expiry', 'vcr', 'scr', 'ssr']]

    def get_fitted_vol_parameter(self) -> list[float]:
        """Get list of parameter values that are fitted during calibration"""
        return [float(getattr(self, name)) for name in self.get_parameter_names()]


def create_wing_model_from_result(result: np.ndarray|list, forward_price: float, ref_price: float, time_to_expiry: float, vcr: float=1.0, scr: float=1.0, ssr: float=0.0):
    return WingModelParameters(
        vr=result[0],
        sr=result[1],
        pc=result[2],
        cc=result[3],
        dc=result[4],
        uc=result[5],
        dsm=result[6],
        usm=result[7],
        vcr=vcr,
        scr=scr,
        ssr=ssr,
        forward_price=forward_price,
        ref_price=ref_price,
        time_to_expiry=time_to_expiry
    )