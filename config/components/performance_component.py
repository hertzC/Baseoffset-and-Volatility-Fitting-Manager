#!/usr/bin/env python3
"""
Performance Configuration Component

Handles the 'performance' section of configuration, including parallel processing,
caching, debugging, and development settings.
"""

from typing import Dict, Any
from .base_component import BaseConfigComponent


class PerformanceComponent(BaseConfigComponent):
    """
    Configuration component for performance-related settings.
    
    Provides specialized access to performance configuration including:
    - Parallel processing settings
    - Caching configuration
    - Debug and development options
    - Memory and resource management
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize performance component."""
        super().__init__(config_data, 'performance')
    
    # Parallel processing settings
    
    @property
    def parallel_enabled(self) -> bool:
        """Get whether parallel processing is enabled."""
        return self.get('parallel.enabled', False)
    
    @property
    def num_processes(self) -> int:
        """Get number of processes for parallel processing."""
        return self.get('parallel.num_processes', -1)
    
    @property
    def chunk_size(self) -> int:
        """Get chunk size for parallel processing."""
        return self.get('parallel.chunk_size', 1)
    
    # Caching settings
    
    @property
    def cache_enabled(self) -> bool:
        """Get whether caching is enabled."""
        return self.get('cache.enabled', False)
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory."""
        return self.get('cache.cache_dir', '.cache')
    
    @property
    def cache_ttl(self) -> int:
        """Get cache time-to-live in seconds."""
        return self.get('cache.ttl', 3600)
    
    @property
    def cache_max_size(self) -> int:
        """Get maximum cache size in MB."""
        return self.get('cache.max_size_mb', 1024)
    
    # Debug settings
    
    @property
    def debug_enabled(self) -> bool:
        """Get whether debug mode is enabled."""
        return self.get('debug.enabled', False)
    
    @property
    def debug_verbose(self) -> bool:
        """Get debug verbosity setting."""
        return self.get('debug.verbose', False)
    
    @property
    def save_intermediate(self) -> bool:
        """Get whether to save intermediate results."""
        return self.get('debug.save_intermediate', False)
    
    @property
    def profiling_enabled(self) -> bool:
        """Get whether profiling is enabled."""
        return self.get('debug.profiling_enabled', False)
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.get('debug.log_level', 'INFO')
    
    # Memory and resource settings
    
    @property
    def max_memory_gb(self) -> float:
        """Get maximum memory usage in GB."""
        return self.get('memory.max_memory_gb', 8.0)
    
    @property
    def memory_monitoring(self) -> bool:
        """Get whether memory monitoring is enabled."""
        return self.get('memory.monitoring', False)
    
    @property
    def gc_threshold(self) -> float:
        """Get garbage collection threshold as memory percentage."""
        return self.get('memory.gc_threshold', 0.8)
    
    def get_parallel_settings(self) -> Dict[str, Any]:
        """Get parallel processing settings."""
        return {
            'enabled': self.parallel_enabled,
            'num_processes': self.num_processes,
            'chunk_size': self.chunk_size
        }
    
    def get_cache_settings(self) -> Dict[str, Any]:
        """Get caching settings."""
        return {
            'enabled': self.cache_enabled,
            'cache_dir': self.cache_dir,
            'ttl': self.cache_ttl,
            'max_size_mb': self.cache_max_size
        }
    
    def get_debug_settings(self) -> Dict[str, Any]:
        """Get debug and development settings."""
        return {
            'enabled': self.debug_enabled,
            'verbose': self.debug_verbose,
            'save_intermediate': self.save_intermediate,
            'profiling_enabled': self.profiling_enabled,
            'log_level': self.log_level
        }
    
    def get_memory_settings(self) -> Dict[str, Any]:
        """Get memory and resource management settings."""
        return {
            'max_memory_gb': self.max_memory_gb,
            'monitoring': self.memory_monitoring,
            'gc_threshold': self.gc_threshold
        }
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get performance optimization settings."""
        return self.get('optimization', {
            'use_numba': False,
            'vectorized_operations': True,
            'batch_processing': True,
            'lazy_loading': False
        })
    
    # Development and testing settings
    
    def get_synthetic_data_config(self) -> Dict[str, Any]:
        """Get synthetic data configuration for testing."""
        return self.get('development.test_mode.synthetic_data', {
            'num_strikes': 20,
            'forward_price': 60000.0,
            'time_to_expiry': 0.25,
            'atm_vol': 0.7
        })
    
    @property
    def baseline_dir(self) -> str:
        """Get baseline directory for regression tests."""
        return self.get('development.regression.baseline_dir', 'test_baselines')
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all performance configuration settings."""
        return {
            'parallel': self.get_parallel_settings(),
            'cache': self.get_cache_settings(),
            'debug': self.get_debug_settings(),
            'memory': self.get_memory_settings(),
            'optimization': self.get_optimization_settings()
        }