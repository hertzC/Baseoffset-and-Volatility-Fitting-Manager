# Interactive Option Analysis GUI - Class-Based Architecture
"""
ENHANCED CLASS-BASED OPTION ANALYSIS GUI
=======================================

Features:
- Clean class architecture for better organization
- Table with alternating row gridlines and BTC toggle
- Enhanced volatility plot with wing model parameters display
- Residuals plot with secondary y-axis showing price impact
- Improved maintainability and extensibility

Usage:
    app = OptionAnalysisGUI(df, expiry, snapshot_time, strikes, vols, forward, calibration, model)
    app.run()
"""

import pickle
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import numpy as np
import polars as pl
from datetime import datetime

from utils.volatility_fitter.time_adjusted_wing_model.time_adjusted_wing_model import TimeAdjustedWingModel


class OptionAnalysisGUI:
    """Main Option Analysis GUI class with enhanced features"""
    
    def __init__(self, df, my_expiry, snapshot_time, strikes_list, market_vols, 
                 forward_price, calibration_results, TimeAdjustedWingModel):
        """
        Initialize the Option Analysis GUI
        
        Args:
            df: Polars DataFrame with option data
            my_expiry: Expiry identifier (e.g., "8MAR24")
            snapshot_time: Timestamp of the data snapshot
            strikes_list: List of strike prices
            market_vols: List of market volatilities
            forward_price: Forward price
            calibration_results: List of calibration results
            TimeAdjustedWingModel: Wing model class
        """
        self.df = df
        self.my_expiry = my_expiry
        self.snapshot_time = snapshot_time
        self.strikes_list = strikes_list
        self.market_vols = market_vols
        self.forward_price = forward_price
        self.calibration_results = calibration_results
        self.TimeAdjustedWingModel = TimeAdjustedWingModel
        
        # Extract spot price from data
        self.spot_price = df['S'][0] if 'S' in df.columns else forward_price
        
        # GUI state variables
        self.btc_mode = False
        self.root = None
        self.tree = None
        
        # Configure styles
        self._configure_styles()
    
    def _configure_styles(self):
        """Configure TTK styles for enhanced appearance"""
        self.style = ttk.Style()
        self.style.configure("Small.Treeview", font=('Arial', 8), rowheight=22)
        self.style.configure("Small.Treeview.Heading", font=('Arial', 9, 'bold'))
        self.style.configure("Small.Treeview", fieldbackground='white', borderwidth=1, relief='solid')
    
    def run(self):
        """Launch the GUI application"""
        self._create_main_window()
        self._create_notebook()
        self._create_tabs()
        self.root.mainloop()
    
    def _create_main_window(self):
        """Create and configure the main window"""
        self.root = tk.Tk()
        self.root.title("Enhanced Option Analysis Dashboard")
        self.root.geometry("1500x900")
        self.root.configure(bg='white')
        
        # Initialize BTC variable after root is created
        self.btc_var = tk.BooleanVar(value=False)
    
    def _create_notebook(self):
        """Create the main notebook widget"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    def _create_tabs(self):
        """Create all tabs"""
        self._create_table_tab()
        self._create_volatility_tab()
    
    def _create_table_tab(self):
        """Create the option data table tab with BTC toggle"""
        table_frame = ttk.Frame(self.notebook)
        self.notebook.add(table_frame, text="Option Chain & Arbitrage")
        
        # Title
        title_text = (f"Option Chain Analysis - {self.my_expiry} @ {self.snapshot_time}  "
                     f"S=${self.spot_price:.0f} F=${self.forward_price:.0f}")
        title_label = tk.Label(table_frame, text=title_text, 
                              font=('Arial', 10, 'bold'), bg='white')
        title_label.pack(pady=10)
        
        # Control frame for toggle
        self._create_control_frame(table_frame)
        
        # Create table
        self._create_data_table(table_frame)
        
        # Create legend
        self._create_legend(table_frame)
    
    def _create_control_frame(self, parent):
        """Create control frame with BTC toggle"""
        control_frame = tk.Frame(parent, bg='white')
        control_frame.pack(pady=5)
        
        # Toggle button - store reference for direct state checking
        self.toggle_button = tk.Checkbutton(control_frame, text="₿ BTC Denomination", variable=self.btc_var, font=('Arial', 9, 'bold'), 
                                       bg='white', fg='orange', command=self._toggle_currency)
        self.toggle_button.pack(side='left', padx=10)
        
        # Info label
        info_label = tk.Label(
            control_frame, 
            text="(Toggle to show prices in BTC instead of USD)", 
            font=('Arial', 8), fg='gray', bg='white'
        )
        info_label.pack(side='left', padx=5)
    
    def _create_data_table(self, parent):
        """Create and configure the data table"""
        # Get currency symbol
        curr_symbol = "₿" if self.btc_mode else "$"
        
        # Define column configuration
        self.column_config = [
            ('Delta', 'delta', 60, self._format_delta),
            ('BidQty_Call', 'bq0_C', 70, self._format_qty),
            (f'BidPx_Call({curr_symbol})', 'bp0_C_usd', 90, self._format_price),
            (f'TV_Call({curr_symbol})', 'tv_C', 90, self._format_price),
            (f'AskPx_Call({curr_symbol})', 'ap0_C_usd', 90, self._format_price),
            ('AskQty_Call', 'aq0_C', 70, self._format_qty),
            ('Strike', 'strike', 80, self._format_strike),
            ('BidQty_Put', 'bq0_P', 70, self._format_qty),
            (f'BidPx_Put({curr_symbol})', 'bp0_P_usd', 90, self._format_price),
            (f'TV_Put({curr_symbol})', 'tv_P', 90, self._format_price),
            (f'AskPx_Put({curr_symbol})', 'ap0_P_usd', 90, self._format_price),
            ('AskQty_Put', 'aq0_P', 70, self._format_qty),
            ('Fit IV%', 'fitVola', 80, self._format_percentage),
            ('Mkt IV%', 'midVola', 80, self._format_percentage),
            ('Gamma', 'gamma', 70, self._format_gamma),
            ('Vega', 'vega', 70, self._format_vega)
        ]
        
        columns = [col[0] for col in self.column_config]
        
        # Create treeview
        self.tree = ttk.Treeview(parent, columns=columns, show='headings', 
                                height=20, style="Small.Treeview")
        
        # Configure columns
        for i, (col_name, _, width, _) in enumerate(self.column_config):
            self.tree.heading(columns[i], text=col_name)
            self.tree.column(columns[i], width=width, anchor='center')
        
        # Configure alternating row colors for gridline effect
        self.tree.tag_configure('even', background='#f8f8f8')
        self.tree.tag_configure('odd', background='white')
        
        self.tree.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Populate table
        self._populate_table()
    
    def _populate_table(self):
        """Populate the table with data"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for i, row in enumerate(self.df.iter_rows()):
            values = []
            # print(f"current row: {row}")
            # Format each column value
            for _, data_source, _, format_func in self.column_config:
                col_idx = self.df.columns.index(data_source)
                raw_value = row[col_idx]
                formatted_value = format_func(raw_value)
                values.append(formatted_value)
            
            # Check for arbitrage conditions and add indicators
            self._add_arbitrage_indicators(values, row)
            
            # Insert row with alternating background
            tag = 'odd' if i % 2 else 'even'
            self.tree.insert('', 'end', values=values, tags=(tag,))
    
    def _add_arbitrage_indicators(self, values, row):
        """Add arbitrage indicators to price values"""
        # Get price values for arbitrage check (always use USD)
        call_bid = row[self.df.columns.index('bp0_C_usd')]
        call_tv = row[self.df.columns.index('tv_C')]
        call_ask = row[self.df.columns.index('ap0_C_usd')]
        put_bid = row[self.df.columns.index('bp0_P_usd')]
        put_tv = row[self.df.columns.index('tv_P')]
        put_ask = row[self.df.columns.index('ap0_P_usd')]
        
        # Find column indices for price columns
        data_sources = [col[1] for col in self.column_config]
        
        # Add arbitrage indicators
        if call_tv < call_bid:
            idx = data_sources.index('bp0_C_usd')
            values[idx] = f"🔴 {values[idx]}"
        elif call_tv > call_ask:
            idx = data_sources.index('ap0_C_usd')
            values[idx] = f"🔵 {values[idx]}"
        elif put_tv < put_bid:
            idx = data_sources.index('bp0_P_usd')
            values[idx] = f"🔴 {values[idx]}"
        elif put_tv > put_ask:
            idx = data_sources.index('ap0_P_usd')
            values[idx] = f"🔵 {values[idx]}"
    
    def _toggle_currency(self):
        """Toggle between USD and BTC display"""
        # Toggle the current mode manually (more reliable than depending on Tkinter variable sync)
        self.btc_mode = not self.btc_mode
        
        # Update the BooleanVar to match
        self.btc_var.set(self.btc_mode)
        
        # Force table rebuild with new currency
        self._rebuild_table_with_currency()
    
    def _rebuild_table_with_currency(self):
        """Rebuild the entire table with current currency setting"""
        # Get currency symbol
        curr_symbol = "₿" if self.btc_mode else "$"
        
        # Recreate column configuration with new currency symbol
        self.column_config = [
            ('Delta', 'delta', 60, self._format_delta),
            ('BidQty_Call', 'bq0_C', 70, self._format_qty),
            (f'BidPx_Call({curr_symbol})', 'bp0_C_usd', 90, self._format_price),
            (f'TV_Call({curr_symbol})', 'tv_C', 90, self._format_price),
            (f'AskPx_Call({curr_symbol})', 'ap0_C_usd', 90, self._format_price),
            ('AskQty_Call', 'aq0_C', 70, self._format_qty),
            ('Strike', 'strike', 80, self._format_strike),
            ('BidQty_Put', 'bq0_P', 70, self._format_qty),
            (f'BidPx_Put({curr_symbol})', 'bp0_P_usd', 90, self._format_price),
            (f'TV_Put({curr_symbol})', 'tv_P', 90, self._format_price),
            (f'AskPx_Put({curr_symbol})', 'ap0_P_usd', 90, self._format_price),
            ('AskQty_Put', 'aq0_P', 70, self._format_qty),
            ('Fit IV%', 'fitVola', 80, self._format_percentage),
            ('Mkt IV%', 'midVola', 80, self._format_percentage),
            ('Gamma', 'gamma', 70, self._format_gamma),
            ('Vega', 'vega', 70, self._format_vega)
        ]
        
        # Update column headers
        for i, (col_name, _, _, _) in enumerate(self.column_config):
            self.tree.heading(self.tree['columns'][i], text=col_name)
        
        # Repopulate table with new formatting
        self._populate_table()
    
    # Formatting methods
    def _format_delta(self, value):
        """Format delta value"""
        return f"{value:.3f}"
    
    def _format_qty(self, value):
        """Format quantity value"""
        return f"{value:.1f}"
    
    def _format_price(self, value):
        """Format price value (USD or BTC)"""
        if self.btc_mode:
            return f"₿{value/self.spot_price:.4f}"
        else:
            return f"${value:.2f}"
    
    def _format_strike(self, value):
        """Format strike price"""
        return f"{value:.0f}"
    
    def _format_percentage(self, value):
        """Format percentage value"""
        return f"{value:.2f}%"
    
    def _format_gamma(self, value):
        """Format gamma value"""
        return f"{value:.6f}"
    
    def _format_vega(self, value):
        """Format vega value"""
        return f"{value:.2f}"
    
    def _create_legend(self, parent):
        """Create legend for arbitrage indicators"""
        legend_frame = tk.Frame(parent, bg='white')
        legend_frame.pack(pady=10)
        
        tk.Label(legend_frame, text="Legend: ", font=('Arial', 8, 'bold'), bg='white').pack(side='left')
        tk.Label(legend_frame, text="🔴 Low BidPx", bg='#FFE5E5', fg='#8B0000', font=('Arial', 8)).pack(side='left', padx=5)
        tk.Label(legend_frame, text="🔵 High AskPx", bg='#E5E5FF', fg='#00008B', font=('Arial', 8)).pack(side='left', padx=5)
    
    def _create_volatility_tab(self):
        """Create the volatility analysis tab"""
        vol_frame = ttk.Frame(self.notebook)
        self.notebook.add(vol_frame, text="Volatility Analysis")
        
        self._create_volatility_plots(vol_frame)
    
    def _create_volatility_plots(self, parent):
        """Create the volatility analysis plots"""
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Volatility Analysis - {self.my_expiry} @ {self.snapshot_time}', fontsize=14, fontweight='bold')
        
        # Get best calibration result
        best_result = min([r for r in self.calibration_results 
                          if r.optimization_method != "Initial Setup"], 
                         key=lambda x: x.error)
        best_model = self.TimeAdjustedWingModel(best_result.parameters)
        
        # Create main volatility plot
        self._create_main_volatility_plot(ax1, best_result, best_model)
        
        # Create residuals plot with price impact
        self._create_residuals_plot(ax2, best_model)
        
        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
    
    def _create_main_volatility_plot(self, ax, best_result, best_model):
        """Create the main volatility smile plot"""
        strikes_array = np.array(self.strikes_list)
        market_vols_array = np.array(self.market_vols)
        fitted_vols_array = np.array([best_model.calculate_volatility_from_strike(strike) * 100 
                                    for strike in strikes_array])
        
        # Plot market data
        ax.scatter(strikes_array, market_vols_array, color='blue', s=50, alpha=0.8, 
                  label='Market Vols', zorder=5)
        
        # Plot smooth fitted curve
        extended_strikes = np.linspace(min(strikes_array) * 0.9, max(strikes_array) * 1.1, 100)
        fitted_smooth = [best_model.calculate_volatility_from_strike(strike) * 100 for strike in extended_strikes]
        
        ax.plot(extended_strikes, fitted_smooth, 'r-', linewidth=2, label=f'Fitted Model ({best_result.optimization_method})')
        ax.scatter(strikes_array, fitted_vols_array, color='red', s=30, alpha=0.7, label='Fitted Points')
        
        # Forward price line
        ax.axvline(x=self.forward_price, color='purple', linestyle='--', alpha=0.7, 
                  label=f'Forward: ${self.forward_price:.0f}')
        
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
        strikes_array = np.array(self.strikes_list)
        market_vols_array = np.array(self.market_vols)
        fitted_vols_array = np.array([best_model.calculate_volatility_from_strike(strike) * 100 
                                    for strike in strikes_array])
        
        vol_diffs = fitted_vols_array - market_vols_array
        
        # Get vega values for price impact calculation
        vega_values = []
        for strike in strikes_array:
            matching_rows = self.df.filter(pl.col('strike') == strike)
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


def main():
    """Main function to load data and launch GUI"""
    print("🚀 Enhanced Option Analysis GUI (Class-Based) Created!")
    
    # Load option chain data
    option_df = pl.read_csv("notebooks/option_data.csv")
    print(f"📥 Loaded option chain with {len(option_df)} rows")
    
    # Load calibration results
    with open("notebooks/calibration_results.pkl", "rb") as f:
        calibration_results = pickle.load(f)
    print(f"📥 Loaded calibration results with {len(calibration_results)} entries")
    
    # Extract parameters
    my_expiry = "8MAR24"
    snapshot_time = (option_df['datetime'][0] if 'datetime' in option_df.columns 
                    else "2024-02-29T20:22:00.000000000")
    strikes_list = option_df['strike'].to_list()
    market_vols = option_df['midVola'].to_list()
    forward_price = option_df['F'][0] if 'F' in option_df.columns else 63000
    
    print(f"🚀 Launching Enhanced Class-Based GUI for {my_expiry} at {snapshot_time}...")
    print(f"📊 DataFrame shape: {option_df.shape}")
    
    # Create and launch GUI
    app = OptionAnalysisGUI(
        df=option_df,
        my_expiry=my_expiry,
        snapshot_time=snapshot_time,
        strikes_list=strikes_list,
        market_vols=market_vols,
        forward_price=forward_price,
        calibration_results=calibration_results,
        TimeAdjustedWingModel=TimeAdjustedWingModel
    )
    
    app.run()
    print("✅ Enhanced Class-Based GUI launched successfully!")


if __name__ == "__main__":
    main()