'''
@Time: 2024/10/15
@Author: Adapted from ORC Wing Model Implementation
@Contact: 
@File: time_adjusted_wing_model.py
@Desc: Time-Adjusted Wing Model Implementation with Class Structure
'''

from typing import Tuple
import numpy as np
from ..base_volatility_model_abstract import BaseVolatilityModel
from ..wing_model.wing_model_parameters import WingModelParameters


class TimeAdjustedWingModel(BaseVolatilityModel):
    """Time-Adjusted Wing Model implementation for volatility surface modeling"""
    
    DAYS_IN_YEAR = 365.25
    
    def __init__(self, parameters: WingModelParameters, use_norm_term: bool=True):
        """Initialize time-adjusted wing model with parameters"""
        super().__init__(parameters)
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

    def calculate_moneyness_from_delta(self, delta: float, option_type: str = 'call', 
                                     time_to_expiry: float = None, atm_vol: float = None) -> float:
        """
        Calculate moneyness based on option delta.
        
        This function converts option delta to the model-specific moneyness by:
        1. Converting delta to d1 using the inverse normal CDF
        2. Applying the model's normalization term
        
        Args:
            delta: Option delta value (0-1 for calls, -1-0 for puts)
            option_type: 'call' or 'put' to handle delta sign convention
            time_to_expiry: Time to expiry in years (uses model params if None)
            atm_vol: At-the-money volatility (uses model params if None)
            
        Returns:
            The calculated moneyness for the model
            
        Raises:
            ValueError: If delta is outside valid range or option_type is invalid
        """
        from scipy.stats import norm
        
        # Use model parameters if not provided
        if time_to_expiry is None:
            time_to_expiry = self.parameters.time_to_expiry
        if atm_vol is None:
            atm_vol = self.parameters.vr
        
        # Ensure time to expiry is not zero
        if time_to_expiry <= 1e-9:
            return 0.0
            
        # Validate option type
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
            
        # Validate delta ranges
        if option_type.lower() == 'call':
            if not (0 <= delta <= 1):
                raise ValueError(f"Call delta must be between 0 and 1, got {delta}")
            # For calls: delta = N(d1), so d1 = N^(-1)(delta)
            d1 = norm.ppf(delta)
        else:  # put
            if not (-1 <= delta <= 0):
                raise ValueError(f"Put delta must be between -1 and 0, got {delta}")
            # For puts: delta = -N(-d1) = N(d1) - 1, so N(d1) = delta + 1
            delta_adjusted = delta + 1
            if delta_adjusted <= 0 or delta_adjusted >= 1:
                raise ValueError(f"Adjusted put delta out of range: {delta_adjusted}")
            d1 = norm.ppf(delta_adjusted)
        
        # Apply normalization term
        n_term = self.get_normalization_term(time_to_expiry)
        
        # Calculate moneyness: moneyness = -n_term * d1
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
            params.time_to_expiry, params.vr  # atm_vol -> vr
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
        vol_at_cut_up = params.vr + params.sr * params.uc + params.cc * params.uc**2  # atm_vol->vr, slope->sr, up_cutoff->uc, call_curve->cc
        vol_at_cut_dn = params.vr + params.sr * params.dc + params.pc * params.dc**2  # atm_vol->vr, slope->sr, down_cutoff->dc, put_curve->pc

        # Curvature for the smoothing wings
        curve_smooth_up = (-(params.sr + 2 * params.cc * params.uc) / 
                          (2 * params.uc * params.usm) if (params.uc * params.usm) != 0 else 0)  # slope->sr, call_curve->cc, up_cutoff->uc, up_smoothing->usm
        curve_smooth_down = (-(params.sr + 2 * params.pc * params.dc) / 
                            (2 * params.dc * params.dsm) if (params.dc * params.dsm) != 0 else 0)  # slope->sr, put_curve->pc, down_cutoff->dc, down_smoothing->dsm

        # Slope for the smoothing wings
        slope_smooth_up = -2 * curve_smooth_up * params.uc * (1 + params.usm)  # up_cutoff->uc, up_smoothing->usm
        slope_smooth_down = -2 * curve_smooth_down * params.dc * (1 + params.dsm)  # down_cutoff->dc, down_smoothing->dsm

        # ATM volatility for the smoothing wings
        atm_vol_smooth_up = vol_at_cut_up - curve_smooth_up * params.uc**2 - slope_smooth_up * params.uc  # up_cutoff->uc
        atm_vol_smooth_down = vol_at_cut_dn - curve_smooth_down * params.dc**2 - slope_smooth_down * params.dc  # down_cutoff->dc

        # --- 2. Determine Volatility based on Moneyness Region ---
        
        # Region 1: Central Upside Parabola (OTM Calls)
        if 0 <= moneyness <= params.uc:  # up_cutoff->uc
            return params.vr + params.sr * moneyness + params.cc * moneyness**2  # atm_vol->vr, slope->sr, call_curve->cc
        
        # Region 2: Central Downside Parabola (OTM Puts)
        elif params.dc <= moneyness < 0:  # down_cutoff->dc
            return params.vr + params.sr * moneyness + params.pc * moneyness**2  # atm_vol->vr, slope->sr, put_curve->pc
        
        # Region 3: Upside Smoothing Wing
        elif params.uc < moneyness <= params.uc * (1 + params.usm):  # up_cutoff->uc, up_smoothing->usm
            return atm_vol_smooth_up + slope_smooth_up * moneyness + curve_smooth_up * moneyness**2
            
        # Region 4: Downside Smoothing Wing
        elif params.dc * (1 + params.dsm) <= moneyness < params.dc:  # down_cutoff->dc, down_smoothing->dsm
            return atm_vol_smooth_down + slope_smooth_down * moneyness + curve_smooth_down * moneyness**2
            
        # Region 5: Flat Extrapolation (Far OTM Calls)
        elif moneyness > params.uc * (1 + params.usm):  # up_cutoff->uc, up_smoothing->usm
            edge_moneyness = params.uc * (1 + params.usm)  # up_cutoff->uc, up_smoothing->usm
            return atm_vol_smooth_up + slope_smooth_up * edge_moneyness + curve_smooth_up * edge_moneyness**2
            
        # Region 6: Flat Extrapolation (Far OTM Puts)
        elif moneyness < params.dc * (1 + params.dsm):  # down_cutoff->dc, down_smoothing->dsm
            edge_moneyness = params.dc * (1 + params.dsm)  # down_cutoff->dc, down_smoothing->dsm
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
                params.time_to_expiry, params.vr  # atm_vol -> vr
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
        moneyness_ranges = {'dSm': self.parameters.dc * (1+self.parameters.dsm),  # down_cutoff->dc, down_smoothing->dsm
                            'dCo': self.parameters.dc,  # down_cutoff->dc
                            'uCo': self.parameters.uc,  # up_cutoff->uc
                            'uSm': self.parameters.uc * (1+self.parameters.usm)}  # up_cutoff->uc, up_smoothing->usm
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
        sigma_sqrt_t = self.parameters.vr * np.sqrt(self.parameters.time_to_expiry)  # atm_vol -> vr
        return self.parameters.forward_price * np.exp(
            -d1_approx * sigma_sqrt_t + self.parameters.vr**2 * self.parameters.time_to_expiry / 2)  # atm_vol -> vr

    
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
