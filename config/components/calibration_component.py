#!/usr/bin/env python3
"""
Calibration Configuration Component

Handles the 'calibration' section of configuration, including optimization
methods, tolerance settings, and calibration-specific parameters.
"""

from typing import Dict, Any, List, Tuple, Optional
from .base_component import BaseConfigComponent


class CalibrationComponent(BaseConfigComponent):
    """
    Configuration component for calibration-related settings.
    
    Provides specialized access to calibration configuration including:
    - Optimization methods and parameters
    - Tolerance and convergence settings
    - Bounds and constraints
    - Weighting schemes
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize calibration component."""
        super().__init__(config_data, 'calibration')
    
    @property
    def method(self) -> str:
        """Get calibration optimization method."""
        return self.get('unified_calibrator.method', 'SLSQP')
    
    @property
    def tolerance(self) -> float:
        """Get calibration tolerance."""
        return self.get('unified_calibrator.tolerance', 1e-10)
    
    @property
    def max_iterations(self) -> int:
        """Get maximum calibration iterations."""
        return self.get('unified_calibrator.max_iterations', 1000)
    
    @property
    def enable_bounds(self) -> bool:
        """Get whether to enable parameter bounds in calibration."""
        return self.get('unified_calibrator.enable_bounds', True)
    
    @property
    def arbitrage_penalty(self) -> float:
        """Get arbitrage penalty factor."""
        return self.get('unified_calibrator.arbitrage_penalty', 1e5)
    
    @property
    def enforce_arbitrage_free(self) -> bool:
        """Get whether to enforce arbitrage-free constraints."""
        return self.get('model_specific.enforce_arbitrage_free', True)
    
    @property
    def weighting_scheme(self) -> str:
        """Get weighting scheme for calibration."""
        return self.get('weighting.scheme', 'vega')
    
    @property
    def min_vega_threshold(self) -> float:
        """Get minimum vega threshold for vega weighting."""
        return self.get('weighting.vega.min_vega', 0.01)
    
    @property
    def normalize_vega_weights(self) -> bool:
        """Get whether to normalize vega weights."""
        return self.get('weighting.vega.normalize', True)
    
    # Multi-start optimization settings
    
    @property
    def multi_start_enabled(self) -> bool:
        """Get whether multi-start optimization is enabled."""
        return self.get('unified_calibrator.multi_start.enabled', False)
    
    @property
    def multi_start_num_starts(self) -> int:
        """Get number of multi-start optimization attempts."""
        return self.get('unified_calibrator.multi_start.num_starts', 5)
    
    @property
    def multi_start_random_seed(self) -> int:
        """Get random seed for multi-start optimization."""
        return self.get('unified_calibrator.multi_start.random_seed', 42)
    
    def get_optimizer_settings(self) -> Dict[str, Any]:
        """Get optimizer configuration settings."""
        return {
            'method': self.method,
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations,
            'enable_bounds': self.enable_bounds,
            'arbitrage_penalty': self.arbitrage_penalty
        }
    
    def get_multi_start_settings(self) -> Dict[str, Any]:
        """Get multi-start optimization settings."""
        return {
            'enabled': self.multi_start_enabled,
            'num_starts': self.multi_start_num_starts,
            'random_seed': self.multi_start_random_seed
        }
    
    def get_weighting_settings(self) -> Dict[str, Any]:
        """Get weighting scheme settings."""
        settings = {
            'scheme': self.weighting_scheme
        }
        
        if self.weighting_scheme == 'vega':
            settings.update({
                'min_vega': self.min_vega_threshold,
                'normalize': self.normalize_vega_weights
            })
        
        return settings
    
    def get_constraint_settings(self) -> Dict[str, Any]:
        """Get constraint and model-specific settings."""
        return {
            'enforce_arbitrage_free': self.enforce_arbitrage_free,
            'additional_constraints': self.get('model_specific.additional_constraints', [])
        }
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all calibration configuration settings."""
        return {
            'optimizer': self.get_optimizer_settings(),
            'multi_start': self.get_multi_start_settings(),
            'weighting': self.get_weighting_settings(),
            'constraints': self.get_constraint_settings()
        }