#!/usr/bin/env python3
"""
Data Configuration Component

Handles the 'data' section of configuration, including data sources,
file paths, and data processing settings.
"""

from typing import Dict, Any, Optional
from .base_component import BaseConfigComponent


class DataComponent(BaseConfigComponent):
    """
    Configuration component for data-related settings.
    
    Provides specialized access to data configuration including:
    - Date settings
    - Data source configuration
    - File paths and patterns
    - Data loading options
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize data component."""
        super().__init__(config_data, 'data')
    
    @property
    def date_str(self) -> str:
        """Get date string in YYYYMMDD format."""
        return self.get('date_str', '20240229')
    
    @property
    def use_orderbook_data(self) -> bool:
        """Get whether to use orderbook depth data."""
        return self.get('use_orderbook_data', False)
    
    @property
    def orderbook_level(self) -> int:
        """Get orderbook level to use."""
        return self.get('orderbook_level', 0)
    
    @property
    def default_interest_rate(self) -> float:
        """Get the default interest rate."""
        return self.get('default_interest_rate', 0.0)
    
    def get_data_file_path(self, data_type: str = 'auto') -> str:
        """
        Get data file path based on configuration.
        
        Args:
            data_type: Type of data ('bbo', 'orderbook', or 'auto')
            
        Returns:
            Full path to the data file
        """
        paths_config = self._config_data.get('paths', {})
        
        if data_type == 'auto':
            data_type = 'orderbook' if self.use_orderbook_data else 'bbo'
        
        if data_type == 'orderbook':
            data_dir = paths_config.get('data_orderbook', 'data_orderbook')
            pattern = paths_config.get('data_file_pattern_orderbook', '{data_dir}/{date_str}.output.csv.gz')
        else:
            data_dir = paths_config.get('data_bbo', 'data_bbo')
            pattern = paths_config.get('data_file_pattern_bbo', '{data_dir}/{date_str}.market_updates.log')
        
        return pattern.format(data_dir=data_dir, date_str=self.date_str)
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all data configuration settings."""
        return {
            'date_str': self.date_str,
            'use_orderbook_data': self.use_orderbook_data,
            'orderbook_level': self.orderbook_level,
            'default_interest_rate': self.default_interest_rate,
            'data_file_path': self.get_data_file_path(),
        }