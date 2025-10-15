'''
@Time: 2024/10/15
@Author: Adapted from ORC Wing Model Implementation
@Contact: 
@File: time_adjusted_wing_model.py
@Desc: Time-Adjusted Wing Model Implementation with Class Structure
'''

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class TimeAdjustedWingModelParameters:
    """Data class to hold time-adjusted wing model parameters"""
    # Core volatility surface parameters
    atm_vol: float  # At-the-money volatility
    slope: float    # Controls the skew of the smile
    curve_up: float     # Controls the curvature for the upside
    curve_down: float   # Controls the curvature for the downside
    cut_up: float       # Moneyness threshold for the upside parabola
    cut_dn: float       # Moneyness threshold for the downside parabola
    mSmUp: float        # Smoothing factor for the upside wing
    mSmDn: float        # Smoothing factor for the downside wing
    
    # Market context parameters
    forward_price: float  # Forward price of the underlying
    time_to_expiry: float  # Time to expiry in years
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        if self.atm_vol <= 0:
            raise ValueError("atm_vol must be positive")
        if self.cut_dn >= 0:
            raise ValueError("cut_dn must be negative")
        if self.cut_up <= 0:
            raise ValueError("cut_up must be positive")
        if self.forward_price <= 0:
            raise ValueError("forward_price must be positive")
        if self.time_to_expiry <= 0:
            raise ValueError("time_to_expiry must be positive")


class TimeAdjustedWingModel:
    """Time-Adjusted Wing Model implementation for volatility surface modeling"""
    
    DAYS_IN_YEAR = 365.25
    
    def __init__(self, parameters: TimeAdjustedWingModelParameters):
        """Initialize time-adjusted wing model with parameters"""
        self.parameters = parameters
    
    def calculate_moneyness(self, forward_price: float, strike_price: float, 
                          time_to_expiry: float, atm_vol: float) -> float:
        """
        Calculate the specific moneyness used by the ORC Wing Model.
        
        This moneyness is based on a normalized Black-76 d1 term.
        
        Args:
            forward_price: The forward price of the underlying asset
            strike_price: The strike price of the option
            time_to_expiry: Time to expiry in years
            atm_vol: The at-the-money volatility
            
        Returns:
            The calculated moneyness for the model
        """
        # Ensure time to expiry is not zero to avoid division errors
        if time_to_expiry <= 1e-9:
            return 0.0

        # Calculate Black-76 d1
        sigma_sqrt_t = atm_vol * np.sqrt(time_to_expiry)
        # Avoid division by zero if sigma_sqrt_t is zero
        if abs(sigma_sqrt_t) < 1e-9:
            return 0.0
            
        d1 = (np.log(forward_price / strike_price) + (atm_vol**2 / 2) * time_to_expiry) / sigma_sqrt_t

        # Calculate Normalization Term
        days_to_expiry = time_to_expiry * self.DAYS_IN_YEAR
        n_term = 0.2 * np.sqrt(30 / days_to_expiry) if days_to_expiry > 0 else 0

        # Final moneyness as defined in the document
        moneyness = -n_term * d1
        return moneyness
    
    def calculate_volatility_from_strike(self, strike_price: float) -> float:
        """
        Calculate implied volatility for a given strike price.
        
        Args:
            strike_price: The strike price of the option
            
        Returns:
            The implied volatility
        """
        params = self.parameters
        
        # Step 1: Calculate model-specific moneyness
        moneyness = self.calculate_moneyness(
            params.forward_price, strike_price, 
            params.time_to_expiry, params.atm_vol
        )
        
        # Step 2: Calculate volatility from moneyness
        return self.calculate_volatility_from_moneyness(moneyness)
    
    def calculate_volatility_from_moneyness(self, moneyness: float) -> float:
        """
        Calculate implied volatility from a pre-calculated moneyness value.
        
        This is the core logic of the Wing Model, defining the smile based on
        a central parabola, smoothing wings, and flat extrapolation.
        
        Args:
            moneyness: The pre-calculated model-specific moneyness
            
        Returns:
            The calculated implied volatility
        """
        params = self.parameters
        
        # --- 1. Calculate Smoothing Parameters ---
        
        # Volatility at the cutoff points using the central parabola formulas
        vol_at_cut_up = params.atm_vol + params.slope * params.cut_up + params.curve_up * params.cut_up**2
        vol_at_cut_dn = params.atm_vol + params.slope * params.cut_dn + params.curve_down * params.cut_dn**2

        # Curvature for the smoothing wings
        curve_smooth_up = (-(params.slope + 2 * params.curve_up * params.cut_up) / 
                          (2 * params.cut_up * params.mSmUp) if (params.cut_up * params.mSmUp) != 0 else 0)
        curve_smooth_down = (-(params.slope + 2 * params.curve_down * params.cut_dn) / 
                            (2 * params.cut_dn * params.mSmDn) if (params.cut_dn * params.mSmDn) != 0 else 0)

        # Slope for the smoothing wings
        slope_smooth_up = -2 * curve_smooth_up * params.cut_up * (1 + params.mSmUp)
        slope_smooth_down = -2 * curve_smooth_down * params.cut_dn * (1 + params.mSmDn)

        # ATM volatility for the smoothing wings
        atm_vol_smooth_up = vol_at_cut_up - curve_smooth_up * params.cut_up**2 - slope_smooth_up * params.cut_up
        atm_vol_smooth_down = vol_at_cut_dn - curve_smooth_down * params.cut_dn**2 - slope_smooth_down * params.cut_dn

        # --- 2. Determine Volatility based on Moneyness Region ---
        
        # Region 1: Central Upside Parabola (OTM Calls)
        if 0 <= moneyness <= params.cut_up:
            return params.atm_vol + params.slope * moneyness + params.curve_up * moneyness**2
        
        # Region 2: Central Downside Parabola (OTM Puts)
        elif params.cut_dn <= moneyness < 0:
            return params.atm_vol + params.slope * moneyness + params.curve_down * moneyness**2
        
        # Region 3: Upside Smoothing Wing
        elif params.cut_up < moneyness <= params.cut_up * (1 + params.mSmUp):
            return atm_vol_smooth_up + slope_smooth_up * moneyness + curve_smooth_up * moneyness**2
            
        # Region 4: Downside Smoothing Wing
        elif params.cut_dn * (1 + params.mSmDn) <= moneyness < params.cut_dn:
            return atm_vol_smooth_down + slope_smooth_down * moneyness + curve_smooth_down * moneyness**2
            
        # Region 5: Flat Extrapolation (Far OTM Calls)
        elif moneyness > params.cut_up * (1 + params.mSmUp):
            edge_moneyness = params.cut_up * (1 + params.mSmUp)
            return atm_vol_smooth_up + slope_smooth_up * edge_moneyness + curve_smooth_up * edge_moneyness**2
            
        # Region 6: Flat Extrapolation (Far OTM Puts)
        elif moneyness < params.cut_dn * (1 + params.mSmDn):
            edge_moneyness = params.cut_dn * (1 + params.mSmDn)
            return atm_vol_smooth_down + slope_smooth_down * edge_moneyness + curve_smooth_down * edge_moneyness**2
            
        else:
            raise ValueError(f"Invalid moneyness value: {moneyness}")
    
    def calculate_durrleman_condition(self, num_points: int = 201) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Durrleman no-arbitrage condition g(k).
        
        Args:
            num_points: Number of points to evaluate
            
        Returns:
            Tuple of (log_moneyness_array, g_values_array)
        """
        params = self.parameters
        
        # Create strike range
        strikes = np.linspace(params.forward_price * 0.5, params.forward_price * 1.5, num_points)
        log_moneyness_k = np.log(strikes / params.forward_price)
        
        # Calculate total variance for each strike
        total_variance_w = []
        for k_strike in strikes:
            # Step 1: Calculate model moneyness from strike
            moneyness = self.calculate_moneyness(
                params.forward_price, k_strike, 
                params.time_to_expiry, params.atm_vol
            )
            # Step 2: Calculate volatility from model moneyness
            vol = self.calculate_volatility_from_moneyness(moneyness)
            total_variance_w.append(vol**2 * params.time_to_expiry)
            
        total_variance_w = np.array(total_variance_w)
        
        # Calculate derivatives
        dw_dk = np.gradient(total_variance_w, log_moneyness_k)
        d2w_dk2 = np.gradient(dw_dk, log_moneyness_k)
        
        # Ensure positive total variance to avoid division by zero
        w = np.maximum(total_variance_w, 1e-9)
        
        # Calculate Durrleman condition terms
        term1 = (1 - log_moneyness_k * dw_dk / (2 * w))**2
        term2 = (dw_dk**2) / 4 * (1/w + 1/4)
        term3 = d2w_dk2 / 2
        
        g_k = term1 - term2 + term3
        
        return log_moneyness_k, g_k
    
    def generate_volatility_surface(self, strike_range: Tuple[float, float], 
                                  num_strikes: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate volatility surface for a range of strikes.
        
        Args:
            strike_range: Tuple of (min_strike, max_strike)
            num_strikes: Number of strike points to generate
            
        Returns:
            Tuple of (strikes_array, volatilities_array)
        """
        strikes = np.linspace(strike_range[0], strike_range[1], num_strikes)
        volatilities = [self.calculate_volatility_from_strike(strike) for strike in strikes]
        
        return strikes, np.array(volatilities)
