#!/usr/bin/env python3
"""
Base Offset Configuration

Configuration module specifically for put-call parity and base offset analysis.
Handles all settings related to interest rate extraction and spread analysis.
"""

from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from .base_config import BaseConfig, ConfigurationError

class BaseOffsetConfig(BaseConfig):
    """
    Configuration manager for Bitcoin Options Base Offset Analysis.
    
    Handles configuration for:
    - Put-call parity regression analysis
    - Interest rate extraction (USD rate r, BTC rate q)
    - Constrained optimization with rate bounds
    - Time series analysis with exponential smoothing
    """
    
    @property
    def config_type(self) -> str:
        """Return the configuration type identifier."""
        return "base_offset"
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path for base offset analysis."""
        return Path(__file__).parent / "base_offset_config.yaml"
    
    def _validate_config(self):
        """Validate base offset configuration structure and required fields."""
        # Validate common sections first
        self._validate_common_sections()
        
        # Validate required sections for base offset analysis
        required_sections = ['data', 'market_data', 'analysis', 'fitting', 'results']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate data section
        data_config = self._config['data']
        if 'date_str' not in data_config:
            raise ConfigurationError("Missing required field: data.date_str")
        
        # Validate analysis section structure
        analysis_config = self._config['analysis']
        if 'time_series' not in analysis_config:
            raise ConfigurationError("Missing required section: analysis.time_series")
        
        # Validate fitting section
        fitting_config = self._config['fitting']
        if 'nonlinear' not in fitting_config:
            raise ConfigurationError("Missing required section: fitting.nonlinear")   
    
    # Analysis Configuration Properties
    
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
    
    # Fitting Algorithm Properties
    
    @property
    def future_spread_mult(self) -> float:
        """Get future spread multiplier."""
        return self.get('fitting.nonlinear.future_spread_mult', 0.0020)
    
    @property
    def future_spread_threshold(self) -> float:
        """Get future spread threshold."""
        return self.get('fitting.nonlinear.future_spread_threshold', 0.0200)
    
    @property
    def lambda_reg(self) -> float:
        """Get lambda regularization parameter."""
        return self.get('fitting.nonlinear.lambda_reg', 500.0)
    
    @property
    def wls_printable(self) -> bool:
        """Get WLS verbose output setting."""
        return self.get('fitting.wls.printable', False)
    
    @property
    def nonlinear_printable(self) -> bool:
        """Get nonlinear optimization verbose output setting."""
        return self.get('fitting.nonlinear.printable', False)
    
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
    
    # Results and Export Properties
    
    @property
    def old_weight(self) -> float:
        """Get exponential smoothing old weight."""
        return self.get('results.smoothing.old_weight', 0.95)
        
    @property
    def min_total_results(self) -> int:
        """Get minimum total results for export."""
        return self.get('results.export.min_total_results', 100)
    
    @property
    def export_dir(self) -> str:
        """Get export directory name."""
        return self.get('results.export.export_dir', 'exports')
    
    # Display and Development Properties
    
    @property
    def polars_max_rows(self) -> int:
        """Get Polars DataFrame maximum rows to display."""
        return self.get('display.polars.max_rows', 20)
    
    @property
    def plotly_renderer(self) -> str:
        """Get Plotly default renderer."""
        return self.get('display.plotly.renderer', 'notebook')
    
    @property
    def progress_verbose(self) -> bool:
        """Get progress reporting verbosity."""
        return self.get('display.progress.verbose', True)
    
    @property
    def progress_report_interval(self) -> int:
        """Get progress reporting interval."""
        return self.get('display.progress.report_interval', 100)
    
    @property
    def debug_mode(self) -> bool:
        """Get debug mode setting."""
        return self.get('development.debug_mode', False)
    
    @property
    def suppress_warnings(self) -> bool:
        """Get suppress warnings setting."""
        return self.get('development.suppress_warnings', True)
    