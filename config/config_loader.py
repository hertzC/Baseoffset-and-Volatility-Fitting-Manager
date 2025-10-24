#!/usr/bin/env python3
"""
Legacy Configuration Loader (Backward Compatibility Shim)

This module provides backward compatibility for existing imports of config_loader.
All functionality has been moved to config_factory.py.

DEPRECATED: Import directly from config package instead:
    from config import BaseOffsetConfig, VolatilityConfig, load_config, etc.
"""

import warnings

# Import everything from the new consolidated factory
from .config_factory import (
    BaseConfig,
    BaseOffsetConfig, 
    VolatilityConfig,
    ConfigFactory,
    ConfigurationError,
    Config,
    load_config,
    get_config,
    get_legacy_config,
    reload_config,
    load_base_offset_config,
    load_volatility_config,
    get_base_offset_config,
    get_volatility_config,
    # Component access functions
    get_data_component,
    get_market_data_component,
    get_analysis_component,
    get_calibration_component,
    get_models_component,
    get_validation_component,
    get_output_component,
    get_performance_component,
    get_component_summary,
)

# Issue deprecation warning when this module is imported
warnings.warn(
    "config.config_loader is deprecated. Import directly from config package instead: "
    "from config import BaseOffsetConfig, VolatilityConfig, load_config",
    DeprecationWarning,
    stacklevel=2
)

# For backward compatibility, provide a get_config() that works like the old API
def get_config() -> BaseConfig:
    """
    Get global configuration instance (legacy compatibility function).
    
    Returns:
        Global configuration instance
        
    Raises:
        ConfigurationError: If configuration not loaded yet
    """
    return get_legacy_config()

__all__ = [
    'BaseConfig',
    'BaseOffsetConfig', 
    'VolatilityConfig',
    'ConfigFactory',
    'ConfigurationError',
    'Config',
    'load_config',
    'get_config',
    'reload_config',
    'load_base_offset_config',
    'load_volatility_config',
    # Component access functions
    'get_data_component',
    'get_market_data_component',
    'get_analysis_component',
    'get_calibration_component',
    'get_models_component',
    'get_validation_component',
    'get_output_component',
    'get_performance_component',
    'get_component_summary',
]