#!/usr/bin/env python3
"""
Configuration Loader for Bitcoin Options Analysis

This module provides utilities to load and manage configuration from YAML files.
All configuration variables used throughout the project are centralized here.
"""

import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigurationError(Exception):
    """Raised when there are configuration-related errors."""
    pass

class Config:
    """
    Configuration manager for Bitcoin Options Analysis.
    
    Loads configuration from YAML files and provides easy access to all settings.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            if config is None:
                raise ConfigurationError("Configuration file is empty or invalid")
            
            return config
        
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _validate_config(self):
        """Validate configuration structure and required fields."""
        required_sections = ['data', 'market_data', 'analysis', 'fitting', 'results']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate data section
        data_config = self._config['data']
        if 'date_str' not in data_config:
            raise ConfigurationError("Missing required field: data.date_str")
        
        # Validate date format
        date_str = data_config['date_str']
        try:
            datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            raise ConfigurationError(f"Invalid date format in data.date_str: {date_str}. Expected YYYYMMDD.")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'data.date_str')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise ConfigurationError(f"Configuration key not found: {key_path}")
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration (defaults to original path)
        """
        save_path = output_path or self.config_path
        
        try:
            with open(save_path, 'w') as file:
                yaml.dump(self._config, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    # Convenience properties for commonly used settings
    
    @property
    def date_str(self) -> str:
        """Get date string in YYYYMMDD format."""
        return self.get('data.date_str')
    
    @property
    def use_orderbook_data(self) -> bool:
        """Get whether to use orderbook depth data."""
        return self.get('data.use_orderbook_data', False)
    
    @property
    def conflation_every(self) -> str:
        """Get conflation interval."""
        return self.get('market_data.conflation.every')
    
    @property
    def conflation_period(self) -> str:
        """Get conflation period."""
        return self.get('market_data.conflation.period')
    
    ##############################  VWAP ##########################################
    @property
    def orderbook_level(self) -> int:
        """Get orderbook level to use."""
        return self.get('data.orderbook_level', 0)
    
    @property
    def price_widening_factor(self) -> float:
        """Get price widening factor."""
        return self.get('market_data.price_widening_factor', 0.00025)
    
    @property
    def target_coin_volume(self) -> float:
        """Get target coin volume."""
        return self.get('market_data.target_coin_volume', 1.0)
    
    @property
    def future_min_tick_size(self) -> float:
        """Get future minimum tick size."""
        return self.get('market_data.future_min_tick_size', 0.0001)
    ################################################################################


    ##############################  Optimization ###################################
    @property
    def use_constrained_optimization(self) -> bool:
        """Get whether to use constrained optimization."""
        return self.get('analysis.time_series.use_constrained_optimization', True)
    
    @property
    def time_interval_seconds(self) -> int:
        """Get time interval in seconds."""
        return self.get('analysis.time_series.time_interval_seconds', 60)
    
    @property
    def cutoff_hour_for_0DTE(self) -> int:
        """Get cutoff time (hours before expiry) for 0DTE analysis."""
        return self.get('analysis.time_series.cutoff_hour_for_0DTE', 4)

    @property
    def minimum_strikes(self) -> int:
        """Get minimum strikes required."""
        return self.get('analysis.time_series.minimum_strikes', 3)
    
    @property
    def future_spread_mult(self) -> float:
        """Get future spread multiplier."""
        return self.get('fitting.nonlinear.future_spread_mult', 0.0020)
    
    @property
    def lambda_reg(self) -> float:
        """Get lambda regularization parameter."""
        return self.get('fitting.nonlinear.lambda_reg', 500.0)
    
    @property
    def old_weight(self) -> float:
        """Get exponential smoothing old weight."""
        return self.get('results.smoothing.old_weight', 0.95)
    ################################################################################
    
    @property
    def export_dir(self) -> str:
        """Get export directory name."""
        return self.get('results.export.export_dir', 'exports')
    
    @property
    def analysis_timestamp(self) -> datetime:
        """Get analysis timestamp as datetime object."""
        ts_config = self.get('analysis.single_analysis.timestamp')
        return datetime(
            year=ts_config['year'],
            month=ts_config['month'],
            day=ts_config['day'],
            hour=ts_config['hour'],
            minute=ts_config['minute'],
            second=ts_config['second']
        )
    
    @property
    def analysis_expiry(self) -> str:
        """Get analysis expiry."""
        return self.get('analysis.single_analysis.expiry')
    
    def get_data_file_path(self) -> str:
        """
        Get full path to data file based on configuration.
        
        Returns:
            Full path to the data file
        """
        data_dir = self.get('paths.data_orderbook') if self.use_orderbook_data else self.get('paths.data_bbo')
        pattern = self.get('paths.data_file_pattern_orderbook') if self.use_orderbook_data else self.get('paths.data_file_pattern_bbo')
        
        return pattern.format(data_dir=data_dir, date_str=self.date_str)
    
    def get_rate_constraints(self) -> Dict[str, float]:
        """
        Get rate constraints for optimization.
        
        Returns:
            Dictionary with r_min, r_max, q_min, q_max
        """
        return {
            'r_min': self.get('fitting.nonlinear.constraints.r_min', -0.05),
            'r_max': self.get('fitting.nonlinear.constraints.r_max', 0.50),
            'q_min': self.get('fitting.nonlinear.constraints.q_min', -0.05),
            'q_max': self.get('fitting.nonlinear.constraints.q_max', 0.20),
            'minimum_rate': self.get('fitting.nonlinear.constraints.minimum_rate', -0.10),
            'maximum_rate': self.get('fitting.nonlinear.constraints.maximum_rate', 0.30)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(config_path='{self.config_path}', date='{self.date_str}')"


# Global configuration instance
_config_instance: Optional[Config] = None

def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Global configuration instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance

def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Global configuration instance
        
    Raises:
        ConfigurationError: If configuration not loaded yet
    """
    global _config_instance
    if _config_instance is None:
        raise ConfigurationError("Configuration not loaded. Call load_config() first.")
    return _config_instance

def reload_config(config_path: str = "config.yaml") -> Config:
    """
    Reload global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Reloaded configuration instance
    """
    return load_config(config_path)