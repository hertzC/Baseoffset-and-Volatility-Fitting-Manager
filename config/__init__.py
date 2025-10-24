#!/usr/bin/env python3
"""
Configuration Package

Provides a modular configuration system for Bitcoin Options Analysis with
specialized configuration classes for different analysis types.

Main Components:
- BaseOffsetConfig: Put-call parity and interest rate extraction
- VolatilityConfig: Volatility surface modeling and calibration
- ConfigFactory: Factory for creating configuration objects

Usage Examples:

    # Use specific configuration classes (recommended)
    from config import BaseOffsetConfig, VolatilityConfig
    
    base_config = BaseOffsetConfig()  # Uses default base_offset_config.yaml
    vol_config = VolatilityConfig()   # Uses default volatility_config.yaml
    
    # Legacy compatibility (deprecated)
    from config import load_config
    
    config = load_config(config_type="base_offset")
    
    # Factory pattern
    from config import ConfigFactory
    
    config = ConfigFactory.create_config("volatility")
"""

# Import everything from the consolidated config_factory module
from .config_factory import (
    # Core configuration classes
    BaseConfig,
    BaseOffsetConfig, 
    VolatilityConfig,
    ConfigFactory,
    ConfigurationError,
    
    # Legacy compatibility
    Config,
    load_config,
    get_config,
    get_legacy_config,
    reload_config,
    load_base_offset_config,
    load_volatility_config,
    get_base_offset_config,
    get_volatility_config,
)

__version__ = "2.0.0"

__all__ = [
    # Core configuration classes
    'BaseConfig',
    'BaseOffsetConfig', 
    'VolatilityConfig',
    'ConfigFactory',
    'ConfigurationError',
    
    # Legacy compatibility
    'Config',
    'load_config',
    'get_config',
    'get_legacy_config',
    'reload_config',
    'load_base_offset_config',
    'load_volatility_config',
    'get_base_offset_config',
    'get_volatility_config',
]