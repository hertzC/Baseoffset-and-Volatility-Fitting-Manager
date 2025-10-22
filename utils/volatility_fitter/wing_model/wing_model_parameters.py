# Parameter constraints definition (outside the dataclass to avoid initialization issues)
WING_MODEL_PARAMETER_CONSTRAINTS = {
    'vr': {'min': 0.01, 'max': 5.0, 'description': 'volatility reference (1% to 500%)'},
    'sr': {'min': -3.0, 'max': 3.0, 'description': 'slope reference'},
    'pc': {'min': 0.01, 'max': 10.0, 'description': 'put curvature (non-negative)'},
    'cc': {'min': 0.01, 'max': 10.0, 'description': 'call curvature (non-negative)'},
    'dc': {'min': -5.0, 'max': -0.01, 'description': 'down cutoff (negative)'},
    'uc': {'min': 0.01, 'max': 5.0, 'description': 'up cutoff (positive)'},
    'dsm': {'min': 0.01, 'max': 10.0, 'description': 'down smoothing range'},
    'usm': {'min': 0.01, 'max': 10.0, 'description': 'up smoothing range'},
}

from dataclasses import dataclass, fields
import numpy as np


@dataclass
class WingModelParameters:
    """Data class to hold wing model parameters with built-in constraints"""
    
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
    
    def __post_init__(self):
        """Apply constraints and validate parameter values"""
        # Validate fitted parameters using constraint definitions
        for param_name in self.get_parameter_names():
            if param_name in WING_MODEL_PARAMETER_CONSTRAINTS:
                value = getattr(self, param_name)
                constraints = WING_MODEL_PARAMETER_CONSTRAINTS[param_name]
                
                # Check bounds and clip if necessary
                min_val = constraints['min']
                max_val = constraints['max']
                
                if value < min_val:
                    setattr(self, param_name, min_val)
                    print(f"Warning: {param_name} ({value:.4f}) clipped to minimum {min_val:.4f}")
                elif value > max_val:
                    setattr(self, param_name, max_val)
                    print(f"Warning: {param_name} ({value:.4f}) clipped to maximum {max_val:.4f}")
        
        # Additional strict validations (raise errors for critical violations)
        if self.time_to_expiry <= 0:
            raise ValueError(f"time_to_expiry must be positive, got {self.time_to_expiry}")
        if self.forward_price <= 0:
            raise ValueError(f"forward_price must be positive, got {self.forward_price}")
        if self.ref_price <= 0:
            raise ValueError(f"ref_price must be positive, got {self.ref_price}")
        
    
    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names that are fitted during calibration"""
        return [field.name for field in fields(self) if field.name not in ['forward_price', 'ref_price', 'time_to_expiry', 'vcr', 'scr', 'ssr']]

    def get_fitted_vol_parameter(self) -> list[float]:
        """Get list of parameter values that are fitted during calibration"""
        return [float(getattr(self, name)) for name in self.get_parameter_names()]
    
    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization algorithms"""
        bounds = []
        for name in self.get_parameter_names():
            if name in WING_MODEL_PARAMETER_CONSTRAINTS:
                constraints = WING_MODEL_PARAMETER_CONSTRAINTS[name]
                bounds.append((constraints['min'], constraints['max']))
            else:
                bounds.append((None, None))  # No bounds for unknown parameters
        return bounds
        
    @classmethod
    def get_constraint_info(cls) -> dict:
        """Get detailed information about parameter constraints"""
        return WING_MODEL_PARAMETER_CONSTRAINTS.copy()
    
    def __repr__(self) -> str:
        pairs = (f"{n}={v:.4f}" for n, v in zip(self.get_parameter_names(), self.get_fitted_vol_parameter()))
        return f"{self.__class__.__name__}(" + ", ".join(pairs) + ")"

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