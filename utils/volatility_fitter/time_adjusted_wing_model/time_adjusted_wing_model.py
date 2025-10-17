'''
@Time: 2024/10/15
@Author: Adapted from ORC Wing Model Implementation
@Contact: 
@File: time_adjusted_wing_model.py
@Desc: Time-Adjusted Wing Model Implementation with Class Structure
'''

from dataclasses import dataclass, fields
from typing import Optional, Tuple
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
    time_to_expiry: float  # Time to expiry in years

    def get_parameter_names(self) -> list[str]:
        return [field.name for field in fields(self) if field.name not in ['forward_price','time_to_expiry']]

    def get_fitted_vol_parameter(self) -> list[float]:
        return [float(getattr(self, name)) for name in self.get_parameter_names()]
    

class TimeAdjustedWingModel:
    """Time-Adjusted Wing Model implementation for volatility surface modeling"""
    
    DAYS_IN_YEAR = 365.25
    ARBITRAGE_LOWER_BOUND = 0.5
    ARBITRAGE_UPPER_BOUND = 2.0
    
    def __init__(self, parameters: TimeAdjustedWingModelParameters, use_norm_term: bool=True):
        """Initialize time-adjusted wing model with parameters"""
        self.parameters = parameters
        self.use_norm_term = use_norm_term

    def get_normalization_term(self, time_to_expiry: float) -> float:
        # Calculate Normalization Term
        if not self.use_norm_term:
            return 1.0
        days_to_expiry = time_to_expiry * self.DAYS_IN_YEAR
        n_term = 0.2 * np.sqrt(30 / days_to_expiry) if days_to_expiry > 0 else 0
        if n_term <= 0:
            raise Exception(f"the normalization term returned is not positive, value={n_term}")
        return n_term

    
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
        n_term = self.get_normalization_term(time_to_expiry)

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
        vol_at_cut_up = params.atm_vol + params.slope * params.up_cutoff + params.call_curve * params.up_cutoff**2
        vol_at_cut_dn = params.atm_vol + params.slope * params.down_cutoff + params.put_curve * params.down_cutoff**2

        # Curvature for the smoothing wings
        curve_smooth_up = (-(params.slope + 2 * params.call_curve * params.up_cutoff) / 
                          (2 * params.up_cutoff * params.up_smoothing) if (params.up_cutoff * params.up_smoothing) != 0 else 0)
        curve_smooth_down = (-(params.slope + 2 * params.put_curve * params.down_cutoff) / 
                            (2 * params.down_cutoff * params.down_smoothing) if (params.down_cutoff * params.down_smoothing) != 0 else 0)

        # Slope for the smoothing wings
        slope_smooth_up = -2 * curve_smooth_up * params.up_cutoff * (1 + params.up_smoothing)
        slope_smooth_down = -2 * curve_smooth_down * params.down_cutoff * (1 + params.down_smoothing)

        # ATM volatility for the smoothing wings
        atm_vol_smooth_up = vol_at_cut_up - curve_smooth_up * params.up_cutoff**2 - slope_smooth_up * params.up_cutoff
        atm_vol_smooth_down = vol_at_cut_dn - curve_smooth_down * params.down_cutoff**2 - slope_smooth_down * params.down_cutoff

        # --- 2. Determine Volatility based on Moneyness Region ---
        
        # Region 1: Central Upside Parabola (OTM Calls)
        if 0 <= moneyness <= params.up_cutoff:
            return params.atm_vol + params.slope * moneyness + params.call_curve * moneyness**2
        
        # Region 2: Central Downside Parabola (OTM Puts)
        elif params.down_cutoff <= moneyness < 0:
            return params.atm_vol + params.slope * moneyness + params.put_curve * moneyness**2
        
        # Region 3: Upside Smoothing Wing
        elif params.up_cutoff < moneyness <= params.up_cutoff * (1 + params.up_smoothing):
            return atm_vol_smooth_up + slope_smooth_up * moneyness + curve_smooth_up * moneyness**2
            
        # Region 4: Downside Smoothing Wing
        elif params.down_cutoff * (1 + params.down_smoothing) <= moneyness < params.down_cutoff:
            return atm_vol_smooth_down + slope_smooth_down * moneyness + curve_smooth_down * moneyness**2
            
        # Region 5: Flat Extrapolation (Far OTM Calls)
        elif moneyness > params.up_cutoff * (1 + params.up_smoothing):
            edge_moneyness = params.up_cutoff * (1 + params.up_smoothing)
            return atm_vol_smooth_up + slope_smooth_up * edge_moneyness + curve_smooth_up * edge_moneyness**2
            
        # Region 6: Flat Extrapolation (Far OTM Puts)
        elif moneyness < params.down_cutoff * (1 + params.down_smoothing):
            edge_moneyness = params.down_cutoff * (1 + params.down_smoothing)
            return atm_vol_smooth_down + slope_smooth_down * edge_moneyness + curve_smooth_down * edge_moneyness**2
            
        else:
            raise ValueError(f"Invalid moneyness value: {moneyness}")
    
    def calculate_durrleman_condition(self, num_points: int = 501) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Durrleman no-arbitrage condition g(k).
        
        Args:
            num_points: Number of points to evaluate
            
        Returns:
            Tuple of (log_moneyness_array, g_values_array)
        """
        params = self.parameters
        
        # Create strike range
        strikes = np.linspace(params.forward_price * self.ARBITRAGE_LOWER_BOUND, params.forward_price * self.ARBITRAGE_UPPER_BOUND, num_points)
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
    
    def get_strike_ranges(self) -> dict:
        """
        Calculate equivalent strike ranges from moneyness ranges
        Note: For time-adjusted wing model, we need to convert from model-specific moneyness to strikes
        
        Returns:
            dict: Dictionary containing strike boundaries for each region
        """
        moneyness_ranges = {'downSmoothing': self.parameters.down_cutoff * (1+self.parameters.down_smoothing),
                            'downCutOff': self.parameters.down_cutoff,
                            'upCutOff': self.parameters.up_cutoff,
                            'upSmoothing': self.parameters.up_cutoff * (1+self.parameters.up_smoothing)}
        strike_ranges = {}
        for region, moenyness in moneyness_ranges.items():
            # For time-adjusted wing model, we need to solve for strike from moneyness
            # Since moneyness = -n_term * d1, and d1 involves strike, this is an approximation
            # We'll use the inverse relationship: strike ≈ forward * exp(-moneyness / n_term)   
            strike_ranges[region] = self.get_strike_from_moneyness(moenyness)                     
        
        return strike_ranges
        
    def get_strike_from_moneyness(self, moneyness: float) -> float:
        """
        given the moneyness, return the equivalent strike price
        """  
        n_term = self.get_normalization_term(self.parameters.time_to_expiry)  # normalization_term is positive
        d1_approx = -moneyness / n_term
        sigma_sqrt_t = self.parameters.atm_vol * np.sqrt(self.parameters.time_to_expiry)
        return self.parameters.forward_price * np.exp(
            -d1_approx * sigma_sqrt_t + self.parameters.atm_vol**2 * self.parameters.time_to_expiry / 2)

    
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
