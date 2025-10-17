'''
@Time: 2024/4/9 2:32 PM
@Author: Jincheng Gong
@Contact: Jincheng.Gong@hotmail.com
@File: wing_model.py
@Desc: Wing Model Implementation for Volatility Surface Modeling
'''

from typing import Tuple
import numpy as np
from .wing_model_parameters import WingModelParameters
from ..base_volatility_model_abstract import BaseVolatilityModel


class WingModel(BaseVolatilityModel):
    """Wing Model implementation for volatility surface modeling"""
    
    def __init__(self, parameters: WingModelParameters):
        """Initialize wing model with parameters"""
        super().__init__(parameters)
    
    def calculate_volatility_from_strike(self, strike: float) -> float:
        """
        Calculate wing model volatility for given strike price
        
        Args:
            strike: strike price
            
        Returns:
            wing model volatility
        """
        # Convert strike to log forward moneyness
        k = np.log(strike / self.parameters.forward_price)
        return self.calculate_volatility_from_moneyness(k)
    
    def calculate_volatility_from_moneyness(self, k: float) -> float:
        """
        Calculate wing model volatility for given log forward moneyness
        
        Args:
            k: log forward moneyness
            
        Returns:
            wing model volatility
        """
        return self.calculate_volatility(k)
    
    def get_moneyness_ranges(self) -> dict:
        """
        Get the different moneyness ranges defined by the wing model
        
        Returns:
            dict: Dictionary containing the moneyness boundaries for each region
        """
        params = self.parameters
        
        return {
            'far_left_tail': {'min': float('-inf'), 'max': params.dc * (1 + params.dsm)},
            'left_smoothing': {'min': params.dc * (1 + params.dsm), 'max': params.dc},
            'left_wing': {'min': params.dc, 'max': 0.0},
            'right_wing': {'min': 0.0, 'max': params.uc},
            'right_smoothing': {'min': params.uc, 'max': params.uc * (1 + params.usm)},
            'far_right_tail': {'min': params.uc * (1 + params.usm), 'max': float('inf')},
            'full_range': {'min': params.dc * (1 + params.dsm), 'max': params.uc * (1 + params.usm)}
        }
    
    def get_strike_ranges(self) -> dict:
        """
        Calculate equivalent strike ranges from moneyness ranges
        
        Returns:
            dict: Dictionary containing strike boundaries for each region
        """
        moneyness_ranges = self.get_moneyness_ranges()
        atm = self.parameters.forward_price
        
        strike_ranges = {}
        for region, bounds in moneyness_ranges.items():
            if bounds['min'] == float('-inf'):
                min_strike = 0.0
            else:
                min_strike = atm * np.exp(bounds['min'])
                
            if bounds['max'] == float('inf'):
                max_strike = float('inf')
            else:
                max_strike = atm * np.exp(bounds['max'])
                
            strike_ranges[region] = {
                'min': min_strike,
                'max': max_strike
            }
        
        return strike_ranges
    
    def calculate_volatility(self, k: float) -> float:
        """
        Calculate wing model volatility for given log forward moneyness
        
        Args:
            k: log forward moneyness
            
        Returns:
            wing model volatility
        """
        params = self.parameters
        
        # Calculate adjusted parameters
        vc = params.vr - params.vcr * params.ssr * ((params.forward_price - params.ref_price) / params.ref_price)
        sc = params.sr - params.scr * params.ssr * ((params.forward_price - params.ref_price) / params.ref_price)
        
        # Different regions of the wing model
        if k < params.dc * (1 + params.dsm):
            # Far left tail
            return (vc + params.dc * (2 + params.dsm) * (sc / 2) + 
                   (1 + params.dsm) * params.pc * params.dc ** 2)
        
        elif params.dc * (1 + params.dsm) < k <= params.dc:
            # Left smoothing region
            return self._calculate_left_smoothing_region(k, vc, sc)
        
        elif params.dc < k <= 0:
            # Left wing (put side)
            return vc + sc * k + params.pc * k ** 2
        
        elif 0 < k <= params.uc:
            # Right wing (call side)
            return vc + sc * k + params.cc * k ** 2
        
        elif params.uc < k <= params.uc * (1 + params.usm):
            # Right smoothing region
            return self._calculate_right_smoothing_region(k, vc, sc)
        
        elif params.uc * (1 + params.usm) < k:
            # Far right tail
            return (vc + params.uc * (2 + params.usm) * (sc / 2) + 
                   (1 + params.usm) * params.cc * params.uc ** 2)
        
        else:
            raise ValueError(f"Invalid log forward moneyness value: {k}")
    
    def _calculate_left_smoothing_region(self, k: float, vc: float, sc: float) -> float:
        """Calculate volatility in the left smoothing region"""
        params = self.parameters
        return (vc - (1 + 1 / params.dsm) * params.pc * params.dc ** 2 - 
               sc * params.dc / (2 * params.dsm) + 
               (1 + 1 / params.dsm) * (2 * params.pc * params.dc + sc) * k - 
               (params.pc / params.dsm + sc / (2 * params.dc * params.dsm)) * k ** 2)
    
    def _calculate_right_smoothing_region(self, k: float, vc: float, sc: float) -> float:
        """Calculate volatility in the right smoothing region"""
        params = self.parameters
        return (vc - (1 + 1 / params.usm) * params.cc * params.uc ** 2 - 
               sc * params.uc / (2 * params.usm) + 
               (1 + 1 / params.usm) * (2 * params.cc * params.uc + sc) * k - 
               (params.cc / params.usm + sc / (2 * params.uc * params.usm)) * k ** 2)
    
    def calculate_durrleman_condition(self, num_points: int = 501) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Durrleman condition for butterfly arbitrage checking
        
        Args:
            num_points: number of moneyness points to evaluate
            
        Returns:
            tuple of (moneyness_array, g_values_array)
        """
        params = self.parameters
        moneyness_list = np.linspace(params.dc, params.uc, num_points)
        
        # Calculate adjusted parameters
        vc = params.vr - params.vcr * params.ssr * ((params.forward_price - params.ref_price) / params.ref_price)
        sc = params.sr - params.scr * params.ssr * ((params.forward_price - params.ref_price) / params.ref_price)
        
        g_list = []
        
        for moneyness in moneyness_list:
            if moneyness <= params.dc * (1 + params.dsm):
                g_list.append(1.0)
            elif params.dc * (1 + params.dsm) < moneyness < params.dc:
                g_val = self._calculate_durrleman_left_smoothing(moneyness, vc, sc)
                g_list.append(g_val)
            elif params.dc <= moneyness <= 0:
                g_val = self._calculate_durrleman_left_wing(moneyness, vc, sc)
                g_list.append(g_val)
            elif 0 < moneyness <= params.uc:
                g_val = self._calculate_durrleman_right_wing(moneyness, vc, sc)
                g_list.append(g_val)
            elif params.uc < moneyness <= params.uc * (1 + params.usm):
                g_val = self._calculate_durrleman_right_smoothing(moneyness, vc, sc)
                g_list.append(g_val)
            elif params.uc * (1 + params.usm) < moneyness:
                g_list.append(1.0)
        
        return np.array(moneyness_list), np.array(g_list)
    
    def _calculate_durrleman_left_smoothing(self, moneyness: float, vc: float, sc: float) -> float:
        """Calculate Durrleman condition in left smoothing region"""
        params = self.parameters
        a = -params.pc / params.dsm - 0.5 * sc / (params.dc * params.dsm)
        
        term1 = (1 + 1 / params.dsm) * (2 * params.dc * params.pc + sc)
        term2 = 2 * (params.pc / params.dsm + 0.5 * sc / (params.dc * params.dsm)) * moneyness
        b1 = -0.25 * (term1 - term2) ** 2
        
        denominator = (-params.dc ** 2 * (1 + 1 / params.dsm) * params.pc - 
                      0.5 * params.dc * sc / params.dsm + vc + 
                      (1 + 1 / params.dsm) * (2 * params.dc * params.pc + sc) * moneyness - 
                      (params.pc / params.dsm + 0.5 * sc / (params.dc * params.dsm)) * moneyness ** 2)
        b2 = 0.25 + 1 / denominator
        b = b1 * b2
        
        c1 = moneyness * (term1 - term2)
        c2 = 2 * denominator
        c = (1 - c1 / c2) ** 2
        
        return a + b + c
    
    def _calculate_durrleman_left_wing(self, moneyness: float, vc: float, sc: float) -> float:
        """Calculate Durrleman condition in left wing region"""
        params = self.parameters
        denominator = vc + sc * moneyness + params.pc * moneyness ** 2
        term1 = params.pc - 0.25 * (sc + 2 * params.pc * moneyness) ** 2 * (0.25 + 1 / denominator)
        term2 = (1 - 0.5 * moneyness * (sc + 2 * params.pc * moneyness) / denominator) ** 2
        return term1 + term2
    
    def _calculate_durrleman_right_wing(self, moneyness: float, vc: float, sc: float) -> float:
        """Calculate Durrleman condition in right wing region"""
        params = self.parameters
        denominator = vc + sc * moneyness + params.cc * moneyness ** 2
        term1 = params.cc - 0.25 * (sc + 2 * params.cc * moneyness) ** 2 * (0.25 + 1 / denominator)
        term2 = (1 - 0.5 * moneyness * (sc + 2 * params.cc * moneyness) / denominator) ** 2
        return term1 + term2
    
    def _calculate_durrleman_right_smoothing(self, moneyness: float, vc: float, sc: float) -> float:
        """Calculate Durrleman condition in right smoothing region"""
        params = self.parameters
        a = -params.cc / params.usm - 0.5 * sc / (params.uc * params.usm)
        
        term1 = (1 + 1 / params.usm) * (2 * params.uc * params.cc + sc)
        term2 = 2 * (params.cc / params.usm + 0.5 * sc / (params.uc * params.usm)) * moneyness
        b1 = -0.25 * (term1 - term2) ** 2
        
        denominator = (-params.uc ** 2 * (1 + 1 / params.usm) * params.cc - 
                      0.5 * params.uc * sc / params.usm + vc + 
                      (1 + 1 / params.usm) * (2 * params.uc * params.cc + sc) * moneyness - 
                      (params.cc / params.usm + 0.5 * sc / (params.uc * params.usm)) * moneyness ** 2)
        b2 = 0.25 + 1 / denominator
        b = b1 * b2
        
        c1 = moneyness * (term1 - term2)
        c2 = 2 * denominator
        c = (1 - c1 / c2) ** 2
        
        return a + b + c