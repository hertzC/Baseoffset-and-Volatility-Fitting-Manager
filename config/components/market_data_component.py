#!/usr/bin/env python3
"""
Market Data Configuration Component

Handles the 'market_data' section of configuration, including data processing,
filtering, and validation settings.
"""

from typing import Dict, Any, List, Tuple
from .base_component import BaseConfigComponent


class MarketDataComponent(BaseConfigComponent):
    """
    Configuration component for market data processing settings.
    
    Provides specialized access to market data configuration including:
    - Strike and volatility filtering
    - Time constraints
    - Price processing settings
    - Volume thresholds
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize market data component."""
        super().__init__(config_data, 'market_data')
    
    @property
    def min_strikes(self) -> int:
        """Get minimum number of strikes required for calibration."""
        return self.get('min_strikes', 5)
    
    @property
    def min_strike_ratio(self) -> float:
        """Get minimum strike ratio (as percentage of forward price)."""
        return self.get('min_strike_ratio', 0.7)
    
    @property
    def max_strike_ratio(self) -> float:
        """Get maximum strike ratio (as percentage of forward price)."""
        return self.get('max_strike_ratio', 1.3)
    
    @property
    def min_time_to_expiry(self) -> float:
        """Get minimum time to expiry (in years) for analysis."""
        return self.get('min_time_to_expiry', 0.01)
    
    @property
    def min_volatility(self) -> float:
        """Get minimum volatility threshold."""
        return self.get('min_volatility', 0.05)
    
    @property
    def max_volatility(self) -> float:
        """Get maximum volatility threshold."""
        return self.get('max_volatility', 5.0)
    
    @property
    def conflation_every(self) -> str:
        """Get conflation interval."""
        return self.get('conflation.every', '1s')
    
    @property
    def conflation_period(self) -> str:
        """Get conflation period."""
        return self.get('conflation.period', '1s')
    
    @property
    def tightening_volume_threshold(self) -> float:
        """Get volume threshold for option spread tightening."""
        return self.get('option_constraints.tightening_volume_threshold', 5.0)
    
    @property
    def price_widening_factor(self) -> float:
        """Get price widening factor."""
        return self.get('price_widening_factor', 0.00025)
    
    @property
    def target_coin_volume(self) -> float:
        """Get target coin volume."""
        return self.get('target_coin_volume', 1.0)
    
    @property
    def future_min_tick_size(self) -> float:
        """Get future minimum tick size."""
        return self.get('future_min_tick_size', 0.0001)
    
    def get_strike_range(self) -> Tuple[float, float]:
        """Get the strike ratio range as a tuple."""
        return (self.min_strike_ratio, self.max_strike_ratio)
    
    def get_volatility_range(self) -> Tuple[float, float]:
        """Get the volatility range as a tuple."""
        return (self.min_volatility, self.max_volatility)
    
    def get_conflation_settings(self) -> Dict[str, str]:
        """Get conflation settings."""
        return {
            'every': self.conflation_every,
            'period': self.conflation_period
        }
    
    def get_constraint_settings(self) -> Dict[str, Any]:
        """Get option constraint settings."""
        return {
            'tightening_volume_threshold': self.tightening_volume_threshold,
            'price_widening_factor': self.price_widening_factor,
            'target_coin_volume': self.target_coin_volume,
            'future_min_tick_size': self.future_min_tick_size
        }
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all market data configuration settings."""
        return {
            'min_strikes': self.min_strikes,
            'strike_range': self.get_strike_range(),
            'min_time_to_expiry': self.min_time_to_expiry,
            'volatility_range': self.get_volatility_range(),
            'conflation': self.get_conflation_settings(),
            'constraints': self.get_constraint_settings()
        }