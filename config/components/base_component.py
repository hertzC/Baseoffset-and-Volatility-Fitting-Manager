#!/usr/bin/env python3
"""
Base Configuration Component

Provides the base class for all configuration components.
"""

from abc import ABC
from typing import Dict, Any, Optional


class BaseConfigComponent(ABC):
    """
    Base class for configuration components.
    
    Each component handles a specific section of the configuration
    and provides specialized access methods for that section.
    """
    
    def __init__(self, config_data: Dict[str, Any], section_name: str):
        """
        Initialize component with configuration data.
        
        Args:
            config_data: Full configuration dictionary
            section_name: Name of the section this component handles
        """
        self._config_data = config_data
        self._section_name = section_name
        self._section_data = config_data.get(section_name, {})
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation within this component's section.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._section_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def exists(self, key_path: str) -> bool:
        """Check if a configuration key exists in this component's section."""
        return self.get(key_path) is not None
    
    def get_section_data(self) -> Dict[str, Any]:
        """Get the entire section data as dictionary."""
        return self._section_data.copy()
    
    def get_section_name(self) -> str:
        """Get the name of the section this component handles."""
        return self._section_name
    
    def __repr__(self) -> str:
        """String representation of the component."""
        return f"{self.__class__.__name__}(section='{self._section_name}')"