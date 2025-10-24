from dataclasses import dataclass, fields
import numpy as np
from typing import Optional


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
    
    # Configuration object for getting parameter bounds
    config: Optional[object] = None
    model_name: str = "time_adjusted_wing_model"  # Which model to get bounds for
    
    def __post_init__(self):
        """Apply constraints and validate parameter values"""
        # Validate fitted parameters using parameter bounds
        bounds = self.get_parameter_bounds()
        param_names = self.get_parameter_names()
        
        for i, param_name in enumerate(param_names):
            if bounds[i] != (None, None):  # Only check if bounds exist
                value = getattr(self, param_name)
                min_val, max_val = bounds[i]
            
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
        return [field.name for field in fields(self) if field.name not in ['forward_price', 'ref_price', 'time_to_expiry', 'vcr', 'scr', 'ssr', 'config', 'model_name']]

    def get_fitted_vol_parameter(self) -> list[float]:
        """Get list of parameter values that are fitted during calibration"""
        return [float(getattr(self, name)) for name in self.get_parameter_names()]
    
    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization algorithms"""
        bounds = []
        param_names = self.get_parameter_names()
        
        if self.config is not None:
            # Use bounds from configuration
            try:
                config_bounds = self.config.get_parameter_bounds(self.model_name)
                # config_bounds is a list of tuples, so we can use it directly
                return config_bounds
            except Exception as e:
                print(f"Warning: Could not get bounds from config: {e}")
                # Fall back to default bounds
        
        # Default bounds if no configuration provided
        default_bounds = {
            'vr': (0.05, 5.0),    # volatility reference bounds
            'sr': (-2.0, 2.0),    # slope reference bounds  
            'pc': (0.001, 5.0),   # put curvature bounds
            'cc': (0.001, 5.0),   # call curvature bounds
            'dc': (-5.0, 0.0),    # down cutoff bounds (must be negative)
            'uc': (0.0, 5.0),     # up cutoff bounds (must be positive)
            'dsm': (0.01, 10.0),  # down smoothing bounds (must be positive)
            'usm': (0.01, 10.0),  # up smoothing bounds (must be positive)
        }
        
        for name in param_names:
            if name in default_bounds:
                bounds.append(default_bounds[name])
            else:
                bounds.append((None, None))  # No bounds for unknown parameters
        
        return bounds
            
    def __repr__(self) -> str:
        pairs = (f"{n}={v:.4f}" for n, v in zip(self.get_parameter_names(), self.get_fitted_vol_parameter()))
        return f"{self.__class__.__name__}(" + ", ".join(pairs) + ")"
    
    def to_dict(self) -> dict:
        return {name: value for name, value in zip(self.get_parameter_names(), self.get_fitted_vol_parameter())}

def create_wing_model_from_result(result: np.ndarray|list, forward_price: float, ref_price: float, 
                                  time_to_expiry: float, vcr: float=1.0, scr: float=1.0, ssr: float=0.0,
                                  config: Optional[object]=None, model_name: str="wing_model"):
    """
    Create wing model parameters from optimization result.
    
    Args:
        result: Array of parameter values from optimization
        forward_price: Forward price of the underlying
        ref_price: Reference forward price
        time_to_expiry: Time to expiry in years
        vcr: Volatility change rate
        scr: Slope change rate
        ssr: Skew swimmingness rate
        config: Configuration object for parameter bounds
        model_name: Which model to get bounds for ("wing_model" or "time_adjusted_wing_model")
    """
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
        time_to_expiry=time_to_expiry,
        config=config,
        model_name=model_name
    )