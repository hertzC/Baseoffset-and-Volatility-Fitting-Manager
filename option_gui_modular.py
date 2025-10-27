# Interactive Option Analysis GUI - Modular Class-Based Architecture
"""
MODULAR CLASS-BASED OPTION ANALYSIS GUI
======================================

Features:
- Modular design with separate classes for each tab
- Clean separation of concerns
- Enhanced maintainability and extensibility
- BTC toggle functionality
- Arbitrage detection
- Volatility analysis with wing model parameters

Usage:
    app = OptionAnalysisGUI(df, expiry, snapshot_time, strikes, vols, forward, calibration, model)
    app.run()
"""

import pickle
import tkinter as tk
from tkinter import ttk
import polars as pl
from datetime import datetime

from utils.volatility_fitter.time_adjusted_wing_model.time_adjusted_wing_model import TimeAdjustedWingModel
from gui_components import OptionChainTab, VolatilityAnalysisTab, DataFormatter


class OptionAnalysisGUI:
    """Main Option Analysis GUI class with modular architecture"""
    
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
        # Core data
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
        self.notebook = None
        self.btc_var = None
        self.tree = None  # Reference to the main table
        self.column_config = []  # Current column configuration
        
        # Initialize components
        self.data_formatter = DataFormatter(self)
        self.option_chain_tab = None
        self.volatility_tab = None
        
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
        self.root.title("Enhanced Option Analysis Dashboard - Modular Architecture")
        self.root.geometry("1500x900")
        self.root.configure(bg='white')
        
        # Initialize BTC variable after root is created
        self.btc_var = tk.BooleanVar(value=False)
    
    def _create_notebook(self):
        """Create the main notebook widget"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    def _create_tabs(self):
        """Create all tabs using modular components"""
        # Create option chain tab
        self.option_chain_tab = OptionChainTab(self.notebook, self)
        self.option_chain_tab.create_tab()
        
        # Create volatility analysis tab
        self.volatility_tab = VolatilityAnalysisTab(self.notebook, self)
        self.volatility_tab.create_tab()
    
    def _toggle_currency(self):
        """Toggle between USD and BTC display"""
        # Toggle the current mode manually (more reliable than depending on Tkinter variable sync)
        self.btc_mode = not self.btc_mode
        
        # Update the BooleanVar to match
        self.btc_var.set(self.btc_mode)
        
        # Force table rebuild with new currency
        if self.option_chain_tab:
            self.option_chain_tab.rebuild_with_currency()
    
    # Data formatting methods (delegated to DataFormatter)
    def _format_text(self, value):
        """Format text value"""
        return self.data_formatter.format_text(value)
    
    def _format_delta(self, value):
        """Format delta value"""
        return self.data_formatter.format_delta(value)
    
    def _format_qty(self, value):
        """Format quantity value"""
        return self.data_formatter.format_qty(value)
    
    def _format_price(self, value):
        """Format price value (USD or BTC)"""
        return self.data_formatter.format_price(value)
    
    def _format_strike(self, value):
        """Format strike price"""
        return self.data_formatter.format_strike(value)
    
    def _format_percentage(self, value):
        """Format percentage value"""
        return self.data_formatter.format_percentage(value)
    
    def _format_gamma(self, value):
        """Format gamma value"""
        return self.data_formatter.format_gamma(value)
    
    def _format_vega(self, value):
        """Format vega value"""
        return self.data_formatter.format_vega(value)


def main():
    """Main function to load data and launch GUI"""
    print("🚀 Enhanced Option Analysis GUI (Modular Architecture) Created!")
    
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
    
    print(f"🚀 Launching Enhanced Modular GUI for {my_expiry} at {snapshot_time}...")
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
    print("✅ Enhanced Modular GUI launched successfully!")


if __name__ == "__main__":
    main()