#!/usr/bin/env python3
"""
Base Configuration System

This module provides the base configuration class and common utilities for
all configuration types in the Bitcoin Options Analysis project.

All specific configuration types inherit from BaseConfig and implement
their own validation and accessor methods.
"""

import yaml
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

class ConfigurationError(Exception):
    """Raised when there are configuration-related errors."""
    pass

class BaseConfig(ABC):
    """
    Abstract base class for all configuration types.
    
    Provides common functionality for loading, validating, and accessing
    configuration data from YAML files.
    """
    
    def __init__(self, config_path: Union[str, Path] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file. If None, uses default
        """
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = str(config_path)
        self._config = self._load_config()
        self._validate_config()
        
        # Initialize components lazily
        self._components = {}
        self._initialize_components()
    
    @abstractmethod
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path for this config type."""
        pass
    
    @abstractmethod
    def _validate_config(self):
        """Validate configuration structure and required fields."""
        pass
    
    @property
    @abstractmethod
    def config_type(self) -> str:
        """Return the configuration type identifier."""
        pass
    
    def _initialize_components(self):
        """Initialize configuration components. Override in subclasses to add specific components."""
        # Import here to avoid circular imports
        from .components import (
            DataComponent, 
            MarketDataComponent, 
            AnalysisComponent,
            CalibrationComponent,
            ModelsComponent,
            ValidationComponent,
            OutputComponent,
            PerformanceComponent
        )
        
        # Initialize components that are common to all config types
        if 'data' in self._config:
            self._components['data'] = DataComponent(self._config)
        
        if 'market_data' in self._config:
            self._components['market_data'] = MarketDataComponent(self._config)
        
        if 'analysis' in self._config:
            self._components['analysis'] = AnalysisComponent(self._config)
        
        if 'calibration' in self._config:
            self._components['calibration'] = CalibrationComponent(self._config)
        
        if 'models' in self._config:
            self._components['models'] = ModelsComponent(self._config)
        
        if 'validation' in self._config:
            self._components['validation'] = ValidationComponent(self._config)
        
        if 'output' in self._config:
            self._components['output'] = OutputComponent(self._config)
        
        if 'performance' in self._config:
            self._components['performance'] = PerformanceComponent(self._config)
    
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
    
    def _validate_common_sections(self):
        """Validate common configuration sections present in all config types."""
        # Validate data section if present
        if 'data' in self._config:
            data_config = self._config['data']
            if 'date_str' in data_config:
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    # Common properties available in all configuration types
    
    @property
    def date_str(self) -> str:
        """Get date string in YYYYMMDD format."""
        return self.data.date_str
    
    @property
    def use_orderbook_data(self) -> bool:
        """Get whether to use orderbook depth data."""
        return self.data.use_orderbook_data
    
    @property
    def orderbook_level(self) -> int:
        """Get orderbook level to use."""
        return self.data.orderbook_level
    
    @property
    def default_interest_rate(self) -> float:
        """Get the default interest rate."""
        return self.data.default_interest_rate
    
    # Market Data Processing Properties
    
    @property
    def conflation_every(self) -> str:
        """Get conflation interval."""
        return self.market_data.conflation_every
    
    @property
    def conflation_period(self) -> str:
        """Get conflation period."""
        return self.market_data.conflation_period
    
    @property
    def tightening_volume_threshold(self) -> float:
        """Get volume threshold for option spread tightening."""
        return self.market_data.tightening_volume_threshold
    
    @property
    def price_widening_factor(self) -> float:
        """Get price widening factor."""
        return self.market_data.price_widening_factor
    
    @property
    def target_coin_volume(self) -> float:
        """Get target coin volume."""
        return self.market_data.target_coin_volume
    
    @property
    def future_min_tick_size(self) -> float:
        """Get future minimum tick size."""
        return self.market_data.future_min_tick_size

    def get_data_file_path(self) -> str:
        """
        Get full path to data file based on configuration.
        
        Returns:
            Full path to the data file
        """
        return self.data.get_data_file_path()
    
    # Component Access Methods
    
    @property
    def data(self):
        """Get data configuration component."""
        return self._components.get('data')
    
    @property
    def market_data(self):
        """Get market data configuration component."""
        return self._components.get('market_data')
    
    @property
    def analysis(self):
        """Get analysis configuration component."""
        return self._components.get('analysis')
    
    @property
    def calibration(self):
        """Get calibration configuration component."""
        return self._components.get('calibration')
    
    @property
    def models(self):
        """Get models configuration component."""
        return self._components.get('models')
    
    @property
    def validation(self):
        """Get validation configuration component."""
        return self._components.get('validation')
    
    @property
    def output(self):
        """Get output configuration component."""
        return self._components.get('output')
    
    @property
    def performance(self):
        """Get performance configuration component."""
        return self._components.get('performance')
    
    def get_component(self, component_name: str):
        """
        Get a specific configuration component by name.
        
        Args:
            component_name: Name of the component to retrieve
            
        Returns:
            Configuration component or None if not found
        """
        return self._components.get(component_name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all available configuration components."""
        return self._components.copy()
    
    def has_component(self, component_name: str) -> bool:
        """Check if a configuration component exists."""
        return component_name in self._components
    
    def get_component_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of all components and their key settings."""
        summary = {}
        for component_name, component in self._components.items():
            if hasattr(component, 'get_all_settings'):
                summary[component_name] = component.get_all_settings()
            else:
                summary[component_name] = {'available': True}
        return summary
    
    def __repr__(self) -> str:
        """String representation of configuration.""" 
        date_info = f", date='{self.date_str}'" if 'data' in self._config and 'date_str' in self._config['data'] else ""
        return f"{self.__class__.__name__}(type='{self.config_type}', config_path='{self.config_path}'{date_info})"