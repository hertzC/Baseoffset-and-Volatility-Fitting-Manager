#!/usr/bin/env python3
"""
Output Configuration Component

Handles the 'output' section of configuration, including plotting,
surface generation, export settings, and visualization parameters.
"""

from typing import Dict, Any, List
from .base_component import BaseConfigComponent


class OutputComponent(BaseConfigComponent):
    """
    Configuration component for output-related settings.
    
    Provides specialized access to output configuration including:
    - Surface generation parameters
    - Plotting and visualization settings
    - Export formats and options
    - File output settings
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize output component."""
        super().__init__(config_data, 'output')
    
    # Surface generation settings
    
    @property
    def surface_num_strikes(self) -> int:
        """Get number of strikes for surface generation."""
        return self.get('surface.num_strikes', 100)
    
    @property
    def surface_strike_range(self) -> List[float]:
        """Get strike range for surface generation (as ratio of forward price)."""
        return self.get('surface.strike_range', [0.5, 2.0])
    
    @property
    def surface_precision(self) -> int:
        """Get output precision (decimal places) for surface generation."""
        return self.get('surface.precision', 6)
    
    # Plotting settings
    
    @property
    def show_bid_ask_spread(self) -> bool:
        """Get whether to show bid/ask spread in plots."""
        return self.get('plotting.show_bid_ask_spread', True)
    
    @property
    def show_market_data(self) -> bool:
        """Get whether to show market data in plots."""
        return self.get('plotting.show_market_data', True)
    
    @property
    def show_fitted_curve(self) -> bool:
        """Get whether to show fitted curve in plots."""
        return self.get('plotting.show_fitted_curve', True)
    
    @property
    def show_error_bars(self) -> bool:
        """Get whether to show error bars in plots."""
        return self.get('plotting.show_error_bars', True)
    
    @property
    def plot_theme(self) -> str:
        """Get plot theme."""
        return self.get('plotting.theme', 'plotly_white')
    
    @property
    def plot_width(self) -> int:
        """Get default plot width."""
        return self.get('plotting.width', 800)
    
    @property
    def plot_height(self) -> int:
        """Get default plot height."""
        return self.get('plotting.height', 600)
    
    @property
    def error_bar_opacity(self) -> float:
        """Get error bar transparency."""
        return self.get('plotting.error_bars.opacity', 0.3)
    
    # Export settings
    
    @property
    def export_formats(self) -> List[str]:
        """Get list of export file formats."""
        return self.get('export.formats', ['csv', 'json'])
    
    @property
    def export_dir(self) -> str:
        """Get export directory."""
        return self.get('export.export_dir', 'results')
    
    @property
    def include_metadata(self) -> bool:
        """Get whether to include metadata in export."""
        return self.get('export.include_metadata', True)
    
    def get_surface_settings(self) -> Dict[str, Any]:
        """Get surface generation settings."""
        return {
            'num_strikes': self.surface_num_strikes,
            'strike_range': self.surface_strike_range,
            'precision': self.surface_precision
        }
    
    def get_plotting_settings(self) -> Dict[str, Any]:
        """Get plotting and visualization settings."""
        return {
            'show_bid_ask_spread': self.show_bid_ask_spread,
            'show_market_data': self.show_market_data,
            'show_fitted_curve': self.show_fitted_curve,
            'show_error_bars': self.show_error_bars,
            'theme': self.plot_theme,
            'width': self.plot_width,
            'height': self.plot_height,
            'error_bar_opacity': self.error_bar_opacity
        }
    
    def get_export_settings(self) -> Dict[str, Any]:
        """Get export settings."""
        return {
            'formats': self.export_formats,
            'export_dir': self.export_dir,
            'include_metadata': self.include_metadata
        }
    
    def get_file_output_settings(self) -> Dict[str, Any]:
        """Get file output settings."""
        return self.get('files', {
            'save_calibration_results': True,
            'save_fitted_surfaces': True,
            'save_plots': False,
            'compression': 'gzip'
        })
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all output configuration settings."""
        return {
            'surface': self.get_surface_settings(),
            'plotting': self.get_plotting_settings(),
            'export': self.get_export_settings(),
            'files': self.get_file_output_settings()
        }