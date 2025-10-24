#!/usr/bin/env python3
"""
Volatility Configuration

Configuration module specifically for volatility surface modeling and calibration.
Handles all settings related to wing model and time-adjusted wing model fitting.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
from .base_config import BaseConfig, ConfigurationError

class VolatilityConfig(BaseConfig):
    """
    Configuration manager for Bitcoin Options Volatility Analysis.
    
    Handles configuration for:
    - Wing Model and Time-Adjusted Wing Model fitting
    - Parameter bounds and initial values
    - Calibration methods and settings
    - Arbitrage validation and quality control
    """
    
    @property
    def config_type(self) -> str:
        """Return the configuration type identifier."""
        return "volatility"
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path for volatility analysis."""
        return Path(__file__).parent / "volatility_config.yaml"
    
    def _validate_config(self):
        """Validate volatility configuration structure and required fields."""
        # Validate common sections first
        self._validate_common_sections()
        
        # Validate required sections for volatility analysis
        required_sections = ['data', 'models', 'calibration']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate models section
        models_config = self._config['models']
        if 'wing_model' not in models_config and 'time_adjusted_wing_model' not in models_config:
            raise ConfigurationError("At least one model (wing_model or time_adjusted_wing_model) must be configured")
        
        # Validate enabled models section
        if 'enabled_models' not in models_config:
            raise ConfigurationError("Missing required section: models.enabled_models")
        
        # Validate calibration section
        calibration_config = self._config['calibration']
        if 'unified_calibrator' not in calibration_config:
            raise ConfigurationError("Missing required section: calibration.unified_calibrator")
    
    # Market Data Processing Properties
    
    @property
    def min_strikes(self) -> int:
        """Get minimum number of strikes required for calibration."""
        return self.market_data.min_strikes
    
    @property
    def min_strike_ratio(self) -> float:
        """Get minimum strike ratio (as percentage of forward price)."""
        return self.market_data.min_strike_ratio
    
    @property
    def max_strike_ratio(self) -> float:
        """Get maximum strike ratio (as percentage of forward price)."""
        return self.market_data.max_strike_ratio
    
    @property
    def min_time_to_expiry(self) -> float:
        """Get minimum time to expiry (in years) for analysis."""
        return self.market_data.min_time_to_expiry
    
    @property
    def min_volatility(self) -> float:
        """Get minimum volatility threshold."""
        return self.market_data.min_volatility
    
    @property
    def max_volatility(self) -> float:
        """Get maximum volatility threshold."""
        return self.market_data.max_volatility
    
    # Model Configuration Properties
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a specific model is enabled."""
        return self.models.is_model_enabled(model_name)
    
    @property
    def wing_model_enabled(self) -> bool:
        """Check if Traditional Wing Model is enabled."""
        return self.models.wing_model_enabled
    
    @property
    def future_min_tick_size(self) -> float:
        """Get future minimum tick size."""
        return self.market_data.future_min_tick_size
    
    @property
    def time_adjusted_wing_model_enabled(self) -> bool:
        """Check if Time-Adjusted Wing Model is enabled."""
        return self.models.time_adjusted_wing_model_enabled
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names."""
        return self.models.get_enabled_models()
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.models.get_model_config(model_name)
    
    def get_initial_params(self, model_name: str) -> Dict[str, float]:
        """Get initial parameters for a model."""
        return self.models.get_initial_params(model_name)
    
    def get_parameter_bounds(self, model_name: str) -> List[Tuple[float, float]]:
        """Get parameter bounds for a model as list of tuples for optimization."""
        return self.models.get_parameter_bounds(model_name)
    
    @property
    def use_norm_term(self) -> bool:
        """Get whether to use normalization term for time-adjusted wing model."""
        return self.models.use_norm_term
    
    # Calibration Configuration Properties
    
    @property
    def calibration_method(self) -> str:
        """Get calibration optimization method."""
        return self.calibration.method
    
    @property
    def calibration_tolerance(self) -> float:
        """Get calibration tolerance."""
        return self.calibration.tolerance
    
    @property
    def max_calibration_iterations(self) -> int:
        """Get maximum calibration iterations."""
        return self.calibration.max_iterations
    
    @property
    def enable_bounds(self) -> bool:
        """Get whether to enable parameter bounds in calibration."""
        return self.calibration.enable_bounds
    
    @property
    def arbitrage_penalty(self) -> float:
        """Get arbitrage penalty factor."""
        return self.calibration.arbitrage_penalty
    
    @property
    def multi_start_enabled(self) -> bool:
        """Get whether multi-start optimization is enabled."""
        return self.calibration.multi_start_enabled
    
    @property
    def multi_start_num_starts(self) -> int:
        """Get number of multi-start optimization attempts."""
        return self.calibration.multi_start_num_starts
    
    @property
    def multi_start_random_seed(self) -> int:
        """Get random seed for multi-start optimization."""
        return self.calibration.multi_start_random_seed
    
    @property
    def enforce_arbitrage_free(self) -> bool:
        """Get whether to enforce arbitrage-free constraints."""
        return self.calibration.enforce_arbitrage_free
    
    @property
    def weighting_scheme(self) -> str:
        """Get weighting scheme for calibration."""
        return self.calibration.weighting_scheme
    
    @property
    def min_vega_threshold(self) -> float:
        """Get minimum vega threshold for vega weighting."""
        return self.calibration.min_vega_threshold
    
    @property
    def normalize_vega_weights(self) -> bool:
        """Get whether to normalize vega weights."""
        return self.calibration.normalize_vega_weights
    
    # Validation and Quality Control Properties
    
    @property
    def durrleman_num_points(self) -> int:
        """Get number of points for Durrleman condition check."""
        return self.validation.durrleman_num_points
    
    @property
    def min_g_value(self) -> float:
        """Get minimum g-value threshold for Durrleman condition."""
        return self.validation.min_g_value
    
    @property
    def arbitrage_lower_bound(self) -> float:
        """Get lower bound for arbitrage checking (as ratio of forward price)."""
        return self.validation.arbitrage_lower_bound
    
    @property
    def arbitrage_upper_bound(self) -> float:
        """Get upper bound for arbitrage checking (as ratio of forward price)."""
        return self.validation.arbitrage_upper_bound
    
    @property
    def max_rmse_threshold(self) -> float:
        """Get maximum acceptable RMSE for calibration."""
        return self.validation.max_rmse_threshold
    
    @property
    def min_r_squared(self) -> float:
        """Get minimum R-squared for goodness of fit."""
        return self.validation.min_r_squared
    
    @property
    def max_param_change(self) -> float:
        """Get maximum parameter change tolerance between iterations."""
        return self.validation.max_param_change
    
    # Output and Visualization Properties
    
    @property
    def surface_num_strikes(self) -> int:
        """Get number of strikes for surface generation."""
        return self.output.surface_num_strikes
    
    @property
    def surface_strike_range(self) -> List[float]:
        """Get strike range for surface generation (as ratio of forward price)."""
        return self.output.surface_strike_range
    
    @property
    def surface_precision(self) -> int:
        """Get output precision (decimal places) for surface generation."""
        return self.output.surface_precision
    
    @property
    def show_bid_ask_spread(self) -> bool:
        """Get whether to show bid/ask spread in plots."""
        return self.output.show_bid_ask_spread
    
    @property
    def show_market_data(self) -> bool:
        """Get whether to show market data in plots."""
        return self.output.show_market_data
    
    @property
    def show_fitted_curve(self) -> bool:
        """Get whether to show fitted curve in plots."""
        return self.output.show_fitted_curve
    
    @property
    def show_error_bars(self) -> bool:
        """Get whether to show error bars in plots."""
        return self.output.show_error_bars
    
    @property
    def plot_theme(self) -> str:
        """Get plot theme."""
        return self.output.plot_theme
    
    @property
    def plot_width(self) -> int:
        """Get default plot width."""
        return self.output.plot_width
    
    @property
    def plot_height(self) -> int:
        """Get default plot height."""
        return self.output.plot_height
    
    @property
    def error_bar_opacity(self) -> float:
        """Get error bar transparency."""
        return self.output.error_bar_opacity
    
    @property
    def export_formats(self) -> List[str]:
        """Get list of export file formats."""
        return self.output.export_formats
    
    @property
    def volatility_export_dir(self) -> str:
        """Get volatility export directory."""
        return self.output.export_dir
    
    # Performance and Development Properties
    
    @property
    def parallel_enabled(self) -> bool:
        """Get whether parallel processing is enabled."""
        return self.performance.parallel_enabled
    
    @property
    def num_processes(self) -> int:
        """Get number of processes for parallel processing."""
        return self.performance.num_processes
    
    @property
    def cache_enabled(self) -> bool:
        """Get whether caching is enabled."""
        return self.performance.cache_enabled
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory."""
        return self.performance.cache_dir
    
    @property
    def debug_enabled(self) -> bool:
        """Get whether debug mode is enabled."""
        return self.performance.debug_enabled
    
    @property
    def debug_verbose(self) -> bool:
        """Get debug verbosity setting."""
        return self.performance.debug_verbose
    
    @property
    def save_intermediate(self) -> bool:
        """Get whether to save intermediate results."""
        return self.performance.save_intermediate
        
    def get_synthetic_data_config(self) -> Dict[str, Any]:
        """Get synthetic data configuration for testing."""
        return self.performance.get_synthetic_data_config()
    
    @property
    def baseline_dir(self) -> str:
        """Get baseline directory for regression tests."""
        return self.performance.baseline_dir
        
    def get_calibration_config(self) -> Dict[str, Any]:
        """Get complete calibration configuration."""
        return self.calibration.get_all_settings()
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get complete validation configuration.""" 
        return self.validation.get_all_settings()