#!/usr/bin/env python3
"""
Analysis Configuration Component

Handles the 'analysis' section of configuration, including analysis parameters,
algorithms, and result management settings.
"""

from typing import Dict, Any, List, Optional
from .base_component import BaseConfigComponent


class AnalysisComponent(BaseConfigComponent):
    """
    Configuration component for analysis-related settings.
    
    Provides specialized access to analysis configuration including:
    - Analysis algorithms and parameters
    - Processing options
    - Result filtering and management
    - Performance settings
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize analysis component."""
        super().__init__(config_data, 'analysis')
    
    @property
    def algorithm(self) -> str:
        """Get the analysis algorithm to use."""
        return self.get('algorithm', 'default')
    
    @property
    def enabled(self) -> bool:
        """Get whether analysis is enabled."""
        return self.get('enabled', True)
    
    @property
    def parallel_processing(self) -> bool:
        """Get whether to use parallel processing."""
        return self.get('parallel_processing', False)
    
    @property
    def num_workers(self) -> int:
        """Get number of worker processes for parallel processing."""
        return self.get('num_workers', -1)
    
    @property
    def cache_results(self) -> bool:
        """Get whether to cache analysis results."""
        return self.get('cache_results', False)
    
    @property
    def cache_directory(self) -> str:
        """Get cache directory for analysis results."""
        return self.get('cache_directory', '.analysis_cache')
    
    @property
    def max_memory_usage(self) -> float:
        """Get maximum memory usage limit (in GB)."""
        return self.get('max_memory_usage', 4.0)
    
    @property
    def timeout_seconds(self) -> Optional[int]:
        """Get analysis timeout in seconds."""
        return self.get('timeout_seconds')
    
    def get_algorithm_parameters(self) -> Dict[str, Any]:
        """Get algorithm-specific parameters."""
        return self.get('algorithm_parameters', {})
    
    def get_filters(self) -> Dict[str, Any]:
        """Get result filtering settings."""
        return self.get('filters', {})
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output settings for analysis results."""
        return self.get('output', {})
    
    def is_algorithm_enabled(self, algorithm_name: str) -> bool:
        """Check if a specific algorithm is enabled."""
        algorithms = self.get('enabled_algorithms', [])
        return algorithm_name in algorithms if algorithms else True
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings."""
        return {
            'parallel_processing': self.parallel_processing,
            'num_workers': self.num_workers,
            'cache_results': self.cache_results,
            'cache_directory': self.cache_directory,
            'max_memory_usage': self.max_memory_usage,
            'timeout_seconds': self.timeout_seconds
        }
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all analysis configuration settings."""
        return {
            'algorithm': self.algorithm,
            'enabled': self.enabled,
            'algorithm_parameters': self.get_algorithm_parameters(),
            'filters': self.get_filters(),
            'output': self.get_output_settings(),
            'performance': self.get_performance_settings()
        }