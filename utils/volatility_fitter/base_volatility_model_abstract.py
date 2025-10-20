'''
@Time: 2025/10/17
@Author: Base Model Architecture
@Contact: 
@File: base_volatility_model.py
@Desc: Abstract Base Class for Volatility Models
'''

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class BaseVolatilityModel(ABC):
    """Abstract base class for all volatility models"""
    
    # Class constants that can be overridden by subclasses
    ARBITRAGE_LOWER_BOUND = 1.0
    ARBITRAGE_UPPER_BOUND = 1.0
    
    def __init__(self, parameters: Any):
        """
        Initialize volatility model with parameters
        
        Args:
            parameters: Model-specific parameter object
        """
        self.parameters = parameters
    
    @abstractmethod
    def calculate_volatility_from_strike(self, strike_price: float) -> float:
        """
        Calculate implied volatility for a given strike price
        
        Args:
            strike_price: The strike price of the option
            
        Returns:
            The implied volatility
        """
        pass
    
    @abstractmethod
    def calculate_volatility_from_moneyness(self, moneyness: float) -> float:
        """
        Calculate implied volatility from a pre-calculated moneyness value
        
        Args:
            moneyness: The pre-calculated model-specific moneyness
            
        Returns:
            The calculated implied volatility
        """
        pass
    
    @abstractmethod
    def calculate_durrleman_condition(self, num_points: int = 501) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Durrleman no-arbitrage condition
        
        Args:
            num_points: Number of points to evaluate
            
        Returns:
            Tuple of (moneyness_or_log_strike_array, g_values_array)
        """
        pass
    
    @abstractmethod
    def get_strike_ranges(self) -> Dict[str, Any]:
        """
        Get the different strike ranges defined by the model
        
        Returns:
            dict: Dictionary containing the strike boundaries for each region
        """
        pass
    
    def generate_volatility_surface(self, strike_range: Tuple[float, float], 
                                  num_strikes: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate volatility surface for a range of strikes
        
        Args:
            strike_range: Tuple of (min_strike, max_strike)
            num_strikes: Number of strike points to generate
            
        Returns:
            Tuple of (strikes_array, volatilities_array)
        """
        strikes = np.linspace(strike_range[0], strike_range[1], num_strikes)
        volatilities = [self.calculate_volatility_from_strike(strike) for strike in strikes]
        
        return strikes, np.array(volatilities)