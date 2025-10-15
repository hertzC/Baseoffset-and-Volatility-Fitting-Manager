from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_VOLA = 0.05  # Minimum volatility to avoid non-physical values
MAX_VOLA = 500.0 # Maximum volatility (also the upper limit in Deribit)

@dataclass
class WingModelParameters:
    """Data class to hold wing model parameters"""
    # Solve parameters
    vr: float = 0.0  # volatility reference
    sr: float = 0.0  # slope reference
    pc: float = 0.0  # put curvature
    cc: float = 0.0  # call curvature
    
    # Input parameters
    dc: float = -0.5  # down cutoff
    uc: float = 0.5   # up cutoff
    dsm: float = 0.1  # down smoothing range
    usm: float = 0.1  # up smoothing range
    
    # Optional parameters
    vcr: float = 0.0  # volatility change rate
    scr: float = 0.0  # slope change rate
    ssr: float = 100.0  # skew swimmingness rate
    atm: float = 1.0  # atm forward
    ref: float = 1.0  # reference forward price

    def __post_init__(self):
        if self.dc >= 0:
            raise ValueError("dc must be negative")
        if self.uc <= 0:
            raise ValueError("uc must be positive")
        if self.vr < MIN_VOLA or self.vr > MAX_VOLA:
            raise ValueError(f"vr must be between {MIN_VOLA} and {MAX_VOLA}")
        if self.dsm < 0:
            raise ValueError("dsm must be positive")
        if self.usm < 0:
            raise ValueError("usm must be positive")


@dataclass
class CalibrationResult:
    """Data class to hold calibration results"""
    success: bool
    parameters: WingModelParameters
    error: float = 0.0
    message: str = ""


def create_wing_model_from_result(result: np.ndarray, atm_price: float, ref_price: float, vcr: float=1.0, scr: float=1.0, ssr: float=0.0):
    return WingModelParameters(
        vr=result.x[0],
        sr=result.x[1], 
        pc=result.x[2],
        cc=result.x[3],
        dc=result.x[4],
        uc=result.x[5],
        dsm=result.x[6],
        usm=result.x[7],
        vcr=vcr,
        scr=scr,
        ssr=ssr,
        atm=atm_price,
        ref=ref_price
    )