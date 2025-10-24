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
        return self.get('market_data.min_strikes', 5)
    
    @property
    def min_strike_ratio(self) -> float:
        """Get minimum strike ratio (as percentage of forward price)."""
        return self.get('market_data.min_strike_ratio', 0.7)
    
    @property
    def max_strike_ratio(self) -> float:
        """Get maximum strike ratio (as percentage of forward price)."""
        return self.get('market_data.max_strike_ratio', 1.3)
    
    @property
    def min_time_to_expiry(self) -> float:
        """Get minimum time to expiry (in years) for analysis."""
        return self.get('market_data.min_time_to_expiry', 0.01)
    
    @property
    def min_volatility(self) -> float:
        """Get minimum volatility threshold."""
        return self.get('market_data.min_volatility', 0.05)
    
    @property
    def max_volatility(self) -> float:
        """Get maximum volatility threshold."""
        return self.get('market_data.max_volatility', 5.0)
    
    # Model Configuration Properties
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a specific model is enabled."""
        return self.get(f'models.enabled_models.{model_name}', True)
    
    @property
    def wing_model_enabled(self) -> bool:
        """Check if Traditional Wing Model is enabled."""
        return self.is_model_enabled('wing_model')
    
    @property
    def future_min_tick_size(self) -> float:
        """Get future minimum tick size."""
        return self.get('market_data.future_min_tick_size', 0.0001)
    
    @property
    def time_adjusted_wing_model_enabled(self) -> bool:
        """Check if Time-Adjusted Wing Model is enabled."""
        return self.is_model_enabled('time_adjusted_wing_model')
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names."""
        enabled_models = []
        models_config = self.get('models.enabled_models', {})
        for model_name, enabled in models_config.items():
            if enabled:
                enabled_models.append(model_name)
        return enabled_models
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.get(f'models.{model_name}', {})
    
    def get_initial_params(self, model_name: str) -> Dict[str, float]:
        """Get initial parameters for a model."""
        return self.get(f'models.{model_name}.initial_params', {})
    
    def get_parameter_bounds(self, model_name: str) -> List[Tuple[float, float]]:
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
    
    @property
    def use_norm_term(self) -> bool:
        """Get whether to use normalization term for time-adjusted wing model."""
        return self.get('models.time_adjusted_wing_model.use_norm_term', True)
    
    # Calibration Configuration Properties
    
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
    
    @property
    def enable_bounds(self) -> bool:
        """Get whether to enable parameter bounds in calibration."""
        return self.get('calibration.unified_calibrator.enable_bounds', True)
    
    @property
    def arbitrage_penalty(self) -> float:
        """Get arbitrage penalty factor."""
        return self.get('calibration.unified_calibrator.arbitrage_penalty', 1e5)
    
    @property
    def multi_start_enabled(self) -> bool:
        """Get whether multi-start optimization is enabled."""
        return self.get('calibration.unified_calibrator.multi_start.enabled', False)
    
    @property
    def multi_start_num_starts(self) -> int:
        """Get number of multi-start optimization attempts."""
        return self.get('calibration.unified_calibrator.multi_start.num_starts', 5)
    
    @property
    def multi_start_random_seed(self) -> int:
        """Get random seed for multi-start optimization."""
        return self.get('calibration.unified_calibrator.multi_start.random_seed', 42)
    
    @property
    def enforce_arbitrage_free(self) -> bool:
        """Get whether to enforce arbitrage-free constraints."""
        return self.get('calibration.model_specific.enforce_arbitrage_free', True)
    
    @property
    def weighting_scheme(self) -> str:
        """Get weighting scheme for calibration."""
        return self.get('calibration.weighting.scheme', 'vega')
    
    @property
    def min_vega_threshold(self) -> float:
        """Get minimum vega threshold for vega weighting."""
        return self.get('calibration.weighting.vega.min_vega', 0.01)
    
    @property
    def normalize_vega_weights(self) -> bool:
        """Get whether to normalize vega weights."""
        return self.get('calibration.weighting.vega.normalize', True)
    
    # Validation and Quality Control Properties
    
    @property
    def durrleman_num_points(self) -> int:
        """Get number of points for Durrleman condition check."""
        return self.get('validation.arbitrage.durrleman.num_points', 501)
    
    @property
    def min_g_value(self) -> float:
        """Get minimum g-value threshold for Durrleman condition."""
        return self.get('validation.arbitrage.durrleman.min_g_value', 0.0)
    
    @property
    def arbitrage_lower_bound(self) -> float:
        """Get lower bound for arbitrage checking (as ratio of forward price)."""
        return self.get('validation.arbitrage.arbitrage_bounds.lower_bound', 0.5)
    
    @property
    def arbitrage_upper_bound(self) -> float:
        """Get upper bound for arbitrage checking (as ratio of forward price)."""
        return self.get('validation.arbitrage.arbitrage_bounds.upper_bound', 2.0)
    
    @property
    def max_rmse_threshold(self) -> float:
        """Get maximum acceptable RMSE for calibration."""
        return self.get('validation.quality.max_rmse', 0.1)
    
    @property
    def min_r_squared(self) -> float:
        """Get minimum R-squared for goodness of fit."""
        return self.get('validation.quality.min_r_squared', 0.8)
    
    @property
    def max_param_change(self) -> float:
        """Get maximum parameter change tolerance between iterations."""
        return self.get('validation.quality.max_param_change', 1e-6)
    
    # Output and Visualization Properties
    
    @property
    def surface_num_strikes(self) -> int:
        """Get number of strikes for surface generation."""
        return self.get('output.surface.num_strikes', 100)
    
    @property
    def surface_strike_range(self) -> List[float]:
        """Get strike range for surface generation (as ratio of forward price)."""
        return self.get('output.surface.strike_range', [0.5, 2.0])
    
    @property
    def surface_precision(self) -> int:
        """Get output precision (decimal places) for surface generation."""
        return self.get('output.surface.precision', 6)
    
    @property
    def show_bid_ask_spread(self) -> bool:
        """Get whether to show bid/ask spread in plots."""
        return self.get('output.plotting.show_bid_ask_spread', True)
    
    @property
    def show_market_data(self) -> bool:
        """Get whether to show market data in plots."""
        return self.get('output.plotting.show_market_data', True)
    
    @property
    def show_fitted_curve(self) -> bool:
        """Get whether to show fitted curve in plots."""
        return self.get('output.plotting.show_fitted_curve', True)
    
    @property
    def show_error_bars(self) -> bool:
        """Get whether to show error bars in plots."""
        return self.get('output.plotting.show_error_bars', True)
    
    @property
    def plot_theme(self) -> str:
        """Get plot theme."""
        return self.get('output.plotting.theme', 'plotly_white')
    
    @property
    def plot_width(self) -> int:
        """Get default plot width."""
        return self.get('output.plotting.width', 800)
    
    @property
    def plot_height(self) -> int:
        """Get default plot height."""
        return self.get('output.plotting.height', 600)
    
    @property
    def error_bar_opacity(self) -> float:
        """Get error bar transparency."""
        return self.get('output.plotting.error_bars.opacity', 0.3)
    
    @property
    def export_formats(self) -> List[str]:
        """Get list of export file formats."""
        return self.get('output.export.formats', ['csv', 'json'])
    
    @property
    def volatility_export_dir(self) -> str:
        """Get volatility export directory."""
        return self.get('output.export.export_dir', 'volatility_results')
    
    # Performance and Development Properties
    
    @property
    def parallel_enabled(self) -> bool:
        """Get whether parallel processing is enabled."""
        return self.get('performance.parallel.enabled', False)
    
    @property
    def num_processes(self) -> int:
        """Get number of processes for parallel processing."""
        return self.get('performance.parallel.num_processes', -1)
    
    @property
    def cache_enabled(self) -> bool:
        """Get whether caching is enabled."""
        return self.get('performance.cache.enabled', False)
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory."""
        return self.get('performance.cache.cache_dir', '.volatility_cache')
    
    @property
    def debug_verbose(self) -> bool:
        """Get debug verbosity setting."""
        return self.get('performance.debug.verbose', False)
    
    @property
    def save_intermediate(self) -> bool:
        """Get whether to save intermediate results."""
        return self.get('performance.debug.save_intermediate', False)
        
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
        
    def get_calibration_config(self) -> Dict[str, Any]:
        """Get complete calibration configuration."""
        return self.get('calibration', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get complete validation configuration.""" 
        return self.get('validation', {})