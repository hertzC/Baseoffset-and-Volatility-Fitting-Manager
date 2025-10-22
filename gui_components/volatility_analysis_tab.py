"""
Volatility Analysis Tab - Handles volatility plots and analysis
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import numpy as np
import polars as pl


class VolatilityAnalysisTab:
    """Handles the volatility analysis tab with plots and statistics"""
    
    def __init__(self, parent_notebook, gui_instance):
        """
        Initialize the volatility analysis tab
        
        Args:
            parent_notebook: The main notebook widget to add this tab to
            gui_instance: Reference to the main GUI instance for data access
        """
        self.notebook = parent_notebook
        self.gui = gui_instance
        self.tab_frame = None
        
    def create_tab(self):
        """Create and configure the volatility analysis tab"""
        # Create tab frame
        self.tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_frame, text="Volatility Analysis")
        
        # Create plots
        self._create_volatility_plots()
        
        return self.tab_frame
    
    def _create_volatility_plots(self):
        """Create the volatility analysis plots"""
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Volatility Analysis - {self.gui.my_expiry} @ {self.gui.snapshot_time}', 
                    fontsize=14, fontweight='bold')
        
        # Get best calibration result
        best_result = min([r for r in self.gui.calibration_results 
                          if r.optimization_method != "Initial Setup"], 
                         key=lambda x: x.error)
        best_model = self.gui.TimeAdjustedWingModel(best_result.parameters)
        
        # Create main volatility plot
        self._create_main_volatility_plot(ax1, best_result, best_model)
        
        # Create residuals plot with price impact
        self._create_residuals_plot(ax2, best_model)
        
        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, self.tab_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.tab_frame)
        toolbar.update()
    
    def _create_main_volatility_plot(self, ax, best_result, best_model):
        """Create the main volatility smile plot"""
        strikes_array = np.array(self.gui.strikes_list)
        market_vols_array = np.array(self.gui.market_vols)
        fitted_vols_array = np.array([best_model.calculate_volatility_from_strike(strike) * 100 
                                    for strike in strikes_array])
        
        # Plot market data
        ax.scatter(strikes_array, market_vols_array, color='blue', s=50, alpha=0.8, 
                  label='Market Vols', zorder=5)
        
        # Plot smooth fitted curve
        extended_strikes = np.linspace(min(strikes_array) * 0.9, max(strikes_array) * 1.1, 100)
        fitted_smooth = [best_model.calculate_volatility_from_strike(strike) * 100 
                        for strike in extended_strikes]
        
        ax.plot(extended_strikes, fitted_smooth, 'r-', linewidth=2, 
               label=f'Fitted Model ({best_result.optimization_method})')
        ax.scatter(strikes_array, fitted_vols_array, color='red', s=30, alpha=0.7, 
                  label='Fitted Points')
        
        # Forward price line
        ax.axvline(x=self.gui.forward_price, color='purple', linestyle='--', alpha=0.7, 
                  label=f'Forward: ${self.gui.forward_price:.0f}')
        
        # Labels and formatting
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title('Volatility Smile/Skew with Wing Model Parameters')
        ax.legend()
        
        # Enhanced gridlines
        ax.minorticks_on()
        ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5, color='gray')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.3, color='lightgray', which='minor')
        
        # Add wing model parameters text box
        self._add_parameter_text_box(ax, best_result)
    
    def _add_parameter_text_box(self, ax, best_result):
        """Add wing model parameters text box to plot"""
        param_text = f"Wing Model Parameters:\\n"
        param_text += f"Method: {best_result.optimization_method}\\n"
        param_text += f"Error: {best_result.error:.4f}\\n"
        param_text += f"Parameters: {str(best_result.parameters)[:50]}..."
        
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9, fontfamily='monospace')
    
    def _create_residuals_plot(self, ax, best_model):
        """Create residuals plot with secondary y-axis for price impact"""
        strikes_array = np.array(self.gui.strikes_list)
        market_vols_array = np.array(self.gui.market_vols)
        fitted_vols_array = np.array([best_model.calculate_volatility_from_strike(strike) * 100 
                                    for strike in strikes_array])
        
        vol_diffs = fitted_vols_array - market_vols_array
        
        # Get vega values for price impact calculation
        vega_values = []
        for strike in strikes_array:
            matching_rows = self.gui.df.filter(pl.col('strike') == strike)
            if len(matching_rows) > 0:
                vega = matching_rows['vega'][0]
                vega_values.append(vega)
            else:
                vega_values.append(0)
        
        vega_array = np.array(vega_values)
        price_diffs = vol_diffs * vega_array / 100  # Convert vol % to price impact
        
        # Primary y-axis (vol differences)
        colors = ['red' if abs(diff) > 2 else 'orange' if abs(diff) > 1 else 'green' 
                 for diff in vol_diffs]
        ax.scatter(strikes_array, vol_diffs, c=colors, s=50, alpha=0.7, 
                  label='Vol Difference', zorder=5)
        
        # Secondary y-axis (price impact)
        ax2 = ax.twinx()
        ax2.plot(strikes_array, price_diffs, 'purple', marker='x', linestyle='--', 
                alpha=0.8, label='Price Impact ($)', markersize=8, linewidth=2)
        
        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='±1% vol threshold')
        ax.axhline(y=-1, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='±2% vol threshold')
        ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        
        ax2.axhline(y=0, color='purple', linestyle='-', alpha=0.3)
        
        # Labels and legend
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Vol Difference (Fitted - Market) %', color='black')
        ax2.set_ylabel('Price Impact (Vol Diff × Vega) $', color='purple')
        ax.set_title('Model Residuals: Volatility & Price Impact')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Enhanced gridlines
        ax.minorticks_on()
        ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5, color='gray')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.3, color='lightgray', which='minor')
        
        # Add statistics
        self._add_statistics_text_box(ax, vol_diffs, price_diffs)
    
    def _add_statistics_text_box(self, ax, vol_diffs, price_diffs):
        """Add statistics text box to residuals plot"""
        mae = np.mean(np.abs(vol_diffs))
        rmse = np.sqrt(np.mean(vol_diffs**2))
        max_error = np.max(np.abs(vol_diffs))
        
        mae_price = np.mean(np.abs(price_diffs))
        max_price_error = np.max(np.abs(price_diffs))
        
        stats_text = (f'Vol: MAE {mae:.2f}% | RMSE {rmse:.2f}% | Max {max_error:.2f}%\\n'
                     f'Price: MAE ${mae_price:.2f} | Max ${max_price_error:.2f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))