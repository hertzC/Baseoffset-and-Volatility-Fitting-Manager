#!/usr/bin/env python3
"""
Validation Configuration Component

Handles the 'validation' section of configuration, including quality checks,
arbitrage validation, and result validation settings.
"""

from typing import Dict, Any, List
from .base_component import BaseConfigComponent


class ValidationComponent(BaseConfigComponent):
    """
    Configuration component for validation-related settings.
    
    Provides specialized access to validation configuration including:
    - Arbitrage checking
    - Quality metrics and thresholds
    - Result validation
    - Error handling
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize validation component."""
        super().__init__(config_data, 'validation')
    
    # Arbitrage validation settings
    
    @property
    def durrleman_num_points(self) -> int:
        """Get number of points for Durrleman condition check."""
        return self.get('arbitrage.durrleman.num_points', 501)
    
    @property
    def min_g_value(self) -> float:
        """Get minimum g-value threshold for Durrleman condition."""
        return self.get('arbitrage.durrleman.min_g_value', 0.0)
    
    @property
    def arbitrage_lower_bound(self) -> float:
        """Get lower bound for arbitrage checking (as ratio of forward price)."""
        return self.get('arbitrage.arbitrage_bounds.lower_bound', 0.5)
    
    @property
    def arbitrage_upper_bound(self) -> float:
        """Get upper bound for arbitrage checking (as ratio of forward price)."""
        return self.get('arbitrage.arbitrage_bounds.upper_bound', 2.0)
    
    # Quality metrics settings
    
    @property
    def max_rmse_threshold(self) -> float:
        """Get maximum acceptable RMSE for calibration."""
        return self.get('quality.max_rmse', 0.1)
    
    @property
    def min_r_squared(self) -> float:
        """Get minimum R-squared for goodness of fit."""
        return self.get('quality.min_r_squared', 0.8)
    
    @property
    def max_param_change(self) -> float:
        """Get maximum parameter change tolerance between iterations."""
        return self.get('quality.max_param_change', 1e-6)
    
    @property
    def max_volatility_error(self) -> float:
        """Get maximum acceptable volatility fitting error."""
        return self.get('quality.max_volatility_error', 0.05)
    
    @property
    def min_market_coverage(self) -> float:
        """Get minimum market coverage requirement (as percentage)."""
        return self.get('quality.min_market_coverage', 0.8)
    
    def get_arbitrage_settings(self) -> Dict[str, Any]:
        """Get arbitrage checking settings."""
        return {
            'durrleman': {
                'num_points': self.durrleman_num_points,
                'min_g_value': self.min_g_value
            },
            'bounds': {
                'lower_bound': self.arbitrage_lower_bound,
                'upper_bound': self.arbitrage_upper_bound
            }
        }
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """Get quality validation settings."""
        return {
            'max_rmse': self.max_rmse_threshold,
            'min_r_squared': self.min_r_squared,
            'max_param_change': self.max_param_change,
            'max_volatility_error': self.max_volatility_error,
            'min_market_coverage': self.min_market_coverage
        }
    
    def get_error_handling_settings(self) -> Dict[str, Any]:
        """Get error handling settings."""
        return self.get('error_handling', {
            'strict_mode': False,
            'fail_on_arbitrage': False,
            'fail_on_quality_threshold': False,
            'log_warnings': True
        })
    
    def get_reporting_settings(self) -> Dict[str, Any]:
        """Get validation reporting settings."""
        return self.get('reporting', {
            'generate_report': True,
            'include_details': True,
            'save_failed_calibrations': False
        })
    
    def is_strict_mode(self) -> bool:
        """Check if strict validation mode is enabled."""
        return self.get('error_handling.strict_mode', False)
    
    def should_fail_on_arbitrage(self) -> bool:
        """Check if calibration should fail on arbitrage violations."""
        return self.get('error_handling.fail_on_arbitrage', False)
    
    def should_fail_on_quality(self) -> bool:
        """Check if calibration should fail on quality threshold violations."""
        return self.get('error_handling.fail_on_quality_threshold', False)
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all validation configuration settings."""
        return {
            'arbitrage': self.get_arbitrage_settings(),
            'quality': self.get_quality_settings(),
            'error_handling': self.get_error_handling_settings(),
            'reporting': self.get_reporting_settings()
        }