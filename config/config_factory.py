#!/usr/bin/env python3
"""
Configuration Factory and Management

This module provides a factory pattern for creating configuration objects
and maintains compatibility with the legacy config_loader interface.

The factory automatically detects configuration type and returns the
appropriate specialized configuration object.
"""

import warnings
from typing import Union, Optional, Any, Dict
from pathlib import Path

from .base_config import BaseConfig, ConfigurationError
from .base_offset_config import BaseOffsetConfig
from .volatility_config import VolatilityConfig

# Global configuration instances
_base_offset_config: Optional[BaseOffsetConfig] = None
_volatility_config: Optional[VolatilityConfig] = None
_legacy_config_instance: Optional[BaseConfig] = None

class ConfigFactory:
    """Factory class for creating configuration objects."""
    
    @staticmethod
    def create_config(config_type: str, config_path: Union[str, Path] = None) -> BaseConfig:
        """
        Create a configuration object of the specified type.
        
        Args:
            config_type: Type of configuration ("base_offset" or "volatility")
            config_path: Path to configuration file (optional)
            
        Returns:
            Configuration object of the appropriate type
            
        Raises:
            ConfigurationError: If config_type is not supported
        """
        if config_type == "base_offset":
            return BaseOffsetConfig(config_path)
        elif config_type == "volatility":
            return VolatilityConfig(config_path)
        else:
            raise ConfigurationError(f"Unsupported configuration type: {config_type}")
    
    @staticmethod
    def auto_detect_config_type(config_path: Union[str, Path]) -> str:
        """
        Auto-detect configuration type based on file path or contents.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Detected configuration type
        """
        path_str = str(config_path).lower()
        
        if "volatility" in path_str:
            return "volatility"
        elif "base_offset" in path_str:
            return "base_offset"
        else:
            # Default to base_offset for backward compatibility
            return "base_offset"


class Config:
    """
    Legacy Configuration class for backward compatibility.
    
    This class delegates to the appropriate specialized configuration
    class based on the config_type parameter.
    
    DEPRECATED: Use BaseOffsetConfig or VolatilityConfig directly.
    """
    
    def __init__(self, config_path: Union[str, Path] = None, config_type: str = "base_offset"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file. If None, uses default based on config_type
            config_type: Type of configuration ("base_offset" or "volatility")
        """
        warnings.warn(
            "The Config class is deprecated. Use BaseOffsetConfig or VolatilityConfig directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create the appropriate specialized config
        self._delegate = ConfigFactory.create_config(config_type, config_path)
        self.config_type = config_type
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the specialized config object."""
        return getattr(self._delegate, name)
    
    def __repr__(self) -> str:
        """String representation of configuration.""" 
        return repr(self._delegate)


# New API functions (recommended)

def load_config(config_path: Union[str, Path] = None, config_type: str = "base_offset") -> BaseConfig:
    """
    Load global configuration instance.
    
    Args:
        config_path: Path to configuration file. If None, uses default based on config_type
        config_type: Type of configuration ("base_offset" or "volatility")
        
    Returns:
        Configuration instance
    """
    global _base_offset_config, _volatility_config, _legacy_config_instance
    
    config = ConfigFactory.create_config(config_type, config_path)
    
    # Store in global variables for compatibility
    if config_type == "base_offset":
        _base_offset_config = config
    elif config_type == "volatility":
        _volatility_config = config
    
    # Also store in legacy global for backward compatibility
    _legacy_config_instance = config
    
    return config

def get_config(config_type: str = "base_offset") -> BaseConfig:
    """
    Get global configuration instance.
    
    Args:
        config_type: Type of configuration to retrieve
        
    Returns:
        Global configuration instance
        
    Raises:
        ConfigurationError: If configuration not loaded yet
    """
    global _base_offset_config, _volatility_config, _legacy_config_instance
    
    if config_type == "base_offset":
        if _base_offset_config is None:
            raise ConfigurationError("Base offset configuration not loaded. Call load_config() first.")
        return _base_offset_config
    elif config_type == "volatility":
        if _volatility_config is None:
            raise ConfigurationError("Volatility configuration not loaded. Call load_config() first.")
        return _volatility_config
    else:
        raise ConfigurationError(f"Unsupported configuration type: {config_type}")

def get_legacy_config() -> BaseConfig:
    """
    Get legacy global configuration instance (for backward compatibility).
    
    Returns:
        Global configuration instance
        
    Raises:
        ConfigurationError: If configuration not loaded yet
    """
    global _legacy_config_instance
    if _legacy_config_instance is None:
        raise ConfigurationError("Configuration not loaded. Call load_config() first.")
    return _legacy_config_instance

def reload_config(config_path: Union[str, Path] = None, config_type: str = "base_offset") -> BaseConfig:
    """
    Reload global configuration instance.
    
    Args:
        config_path: Path to configuration file. If None, uses default based on config_type
        config_type: Type of configuration ("base_offset" or "volatility")
        
    Returns:
        Reloaded configuration instance
    """
    return load_config(config_path, config_type)

# Convenience functions for specific config types

def load_base_offset_config(config_path: Union[str, Path] = None) -> BaseOffsetConfig:
    """Load base offset configuration."""
    return load_config(config_path, "base_offset")

def load_volatility_config(config_path: Union[str, Path] = None) -> VolatilityConfig:
    """Load volatility configuration."""
    return load_config(config_path, "volatility")

def get_base_offset_config() -> BaseOffsetConfig:
    """Get global base offset configuration instance."""
    return get_config("base_offset")

def get_volatility_config() -> VolatilityConfig:
    """Get global volatility configuration instance."""
    return get_config("volatility")

# Component access functions

def get_data_component(config_type: str = "volatility"):
    """Get data configuration component."""
    config = get_config(config_type)
    return config.data

def get_market_data_component(config_type: str = "volatility"):
    """Get market data configuration component."""
    config = get_config(config_type)
    return config.market_data

def get_analysis_component(config_type: str = "volatility"):
    """Get analysis configuration component."""
    config = get_config(config_type)
    return config.analysis

def get_calibration_component(config_type: str = "volatility"):
    """Get calibration configuration component."""
    config = get_config(config_type)
    return config.calibration

def get_models_component(config_type: str = "volatility"):
    """Get models configuration component."""
    config = get_config(config_type)
    return config.models

def get_validation_component(config_type: str = "volatility"):
    """Get validation configuration component."""
    config = get_config(config_type)
    return config.validation

def get_output_component(config_type: str = "volatility"):
    """Get output configuration component."""
    config = get_config(config_type)
    return config.output

def get_performance_component(config_type: str = "volatility"):
    """Get performance configuration component."""
    config = get_config(config_type)
    return config.performance

def get_component_summary(config_type: str = "volatility") -> Dict[str, Dict[str, Any]]:
    """Get summary of all configuration components."""
    config = get_config(config_type)
    return config.get_component_summary()

# Direct access to configuration classes for type-specific usage
__all__ = [
    'BaseConfig',
    'BaseOffsetConfig', 
    'VolatilityConfig',
    'ConfigFactory',
    'ConfigurationError',
    'Config',  # Legacy class
    'load_config',
    'get_config',
    'get_legacy_config',
    'reload_config',
    'load_base_offset_config',
    'load_volatility_config',
    'get_base_offset_config',
    'get_volatility_config',
    # Component access functions
    'get_data_component',
    'get_market_data_component',
    'get_analysis_component',
    'get_calibration_component',
    'get_models_component',
    'get_validation_component',
    'get_output_component',
    'get_performance_component',
    'get_component_summary'
]