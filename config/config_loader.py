#!/usr/bin/env python3
"""
Universal Configuration Loader

This module provides utilities to load and manage configuration from YAML files.
Supports multiple configuration files for different components:
- base_offset_config.yaml: Put-call parity and base offset analysis
- volatility_config.yaml: Volatility model fitting and calibration

All configuration variables used throughout the project are centralized here.
"""

import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

class ConfigurationError(Exception):
    """Raised when there are configuration-related errors."""
    pass

class Config:
    """
    Configuration manager for Bitcoin Options Analysis.
    
    Loads configuration from YAML files and provides easy access to all settings.
    """
    
    def __init__(self, config_path: Union[str, Path] = None, config_type: str = "base_offset"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file. If None, uses default based on config_type
            config_type: Type of configuration ("base_offset" or "volatility")
        """
        # Set default config path based on type if not provided
        if config_path is None:
            config_dir = Path(__file__).parent
            if config_type == "volatility":
                config_path = config_dir / "volatility_config.yaml"
            else:  # default to base_offset
                config_path = config_dir / "base_offset_config.yaml"
        
        self.config_path = str(config_path)
        self.config_type = config_type
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
        """Validate configuration structure and required fields based on config type."""
        if self.config_type == "volatility":
            self._validate_volatility_config()
        else:  # base_offset
            self._validate_base_offset_config()
        
        # Common validation for data section
        if 'data' in self._config:
            data_config = self._config['data']
            if 'date_str' in data_config:
                # Validate date format
                date_str = data_config['date_str']
                try:
                    datetime.strptime(date_str, '%Y%m%d')
                except ValueError:
                    raise ConfigurationError(f"Invalid date format in data.date_str: {date_str}. Expected YYYYMMDD.")
    
    def _validate_base_offset_config(self):
        """Validate base offset configuration structure."""
        required_sections = ['data', 'market_data', 'analysis', 'fitting', 'results']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate data section
        data_config = self._config['data']
        if 'date_str' not in data_config:
            raise ConfigurationError("Missing required field: data.date_str")
    
    def _validate_volatility_config(self):
        """Validate volatility configuration structure."""
        required_sections = ['data', 'models', 'calibration']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate models section
        models_config = self._config['models']
        if 'wing_model' not in models_config and 'time_adjusted_wing_model' not in models_config:
            raise ConfigurationError("At least one model (wing_model or time_adjusted_wing_model) must be configured")
    
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
    def default_interest_rate(self) -> float:
        """ Get the default interest rate """
        return self.get('data.default_interest_rate', 0.0)

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
    
    @property
    def tightening_volume_threshold(self) -> float:
        """Get volume threshold for option spread tightening."""
        return self.get('market_data.option_constraints.tightening_volume_threshold', 5.0)
    
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


    ############################## BO Optimization ###################################
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
    
    # Convenience properties for volatility configuration
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.get(f'models.{model_name}', {})
    
    def get_initial_params(self, model_name: str) -> Dict[str, float]:
        """Get initial parameters for a model."""
        return self.get(f'models.{model_name}.initial_params', {})
    
    def get_parameter_bounds(self, model_name: str) -> list:
        """Get parameter bounds for a model as list of tuples for optimization."""
        bounds_dict = self.get(f'models.{model_name}.bounds', {})
        
        if isinstance(bounds_dict, list):
            # Handle list format (legacy)
            return [tuple(float(x) for x in bounds) for bounds in bounds_dict]
        elif isinstance(bounds_dict, dict):
            # Handle dictionary format - convert to ordered list of tuples for the model
            param_order = ['vr', 'sr', 'pc', 'cc', 'dc', 'uc', 'dsm', 'usm']
            bounds_list = []
            for param in param_order:
                if param in bounds_dict:
                    bounds_list.append(tuple(float(x) for x in bounds_dict[param]))
            return bounds_list
        else:
            return []
    
    def get_calibration_config(self) -> Dict[str, Any]:
        """Get calibration configuration."""
        return self.get('calibration', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration.""" 
        return self.get('validation', {})
    
    @property
    def max_rmse_threshold(self) -> float:
        """Get maximum acceptable RMSE for calibration."""
        return self.get('validation.quality.max_rmse', 0.1)
    
    @property
    def calibration_method(self) -> str:
        """Get calibration optimization method."""
        return self.get('calibration.unified_calibrator.method', 'SLSQP')
    
    @property
    def calibration_tolerance(self) -> float:
        """Get calibration tolerance."""
        return self.get('calibration.unified_calibrator.tolerance', 1e-10)
    
    @property
    def max_calibration_iterations(self) -> int:
        """Get maximum calibration iterations."""
        return self.get('calibration.unified_calibrator.max_iterations', 1000)
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a specific model is enabled."""
        return self.get(f'models.enabled_models.{model_name}', True)
    
    @property
    def wing_model_enabled(self) -> bool:
        """Check if Traditional Wing Model is enabled."""
        return self.is_model_enabled('wing_model')
    
    @property
    def time_adjusted_wing_model_enabled(self) -> bool:
        """Check if Time-Adjusted Wing Model is enabled."""
        return self.is_model_enabled('time_adjusted_wing_model')
    
    def get_enabled_models(self) -> list:
        """Get list of enabled model names."""
        enabled_models = []
        models_config = self.get('models.enabled_models', {})
        for model_name, enabled in models_config.items():
            if enabled:
                enabled_models.append(model_name)
        return enabled_models
    
    def __repr__(self) -> str:
        """String representation of configuration.""" 
        date_info = f", date='{self.date_str}'" if 'data' in self._config and 'date_str' in self._config['data'] else ""
        return f"Config(type='{self.config_type}', config_path='{self.config_path}'{date_info})"


# Global configuration instance
_config_instance: Optional[Config] = None

def load_config(config_path: Union[str, Path] = None, config_type: str = "base_offset") -> Config:
    """
    Load global configuration instance.
    
    Args:
        config_path: Path to configuration file. If None, uses default based on config_type
        config_type: Type of configuration ("base_offset" or "volatility")
        
    Returns:
        Global configuration instance
    """
    global _config_instance
    _config_instance = Config(config_path, config_type)
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

def reload_config(config_path: Union[str, Path] = None, config_type: str = "base_offset") -> Config:
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
def load_base_offset_config(config_path: Union[str, Path] = None) -> Config:
    """Load base offset configuration."""
    return load_config(config_path, "base_offset")

def load_volatility_config(config_path: Union[str, Path] = None) -> Config:
    """Load volatility configuration."""
    return load_config(config_path, "volatility")