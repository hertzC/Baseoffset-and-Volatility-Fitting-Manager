"""
Data Formatting Utilities - Handles all data formatting functions
"""


class DataFormatter:
    """Handles all data formatting for the GUI components"""
    
    def __init__(self, gui_instance):
        """
        Initialize the data formatter
        
        Args:
            gui_instance: Reference to the main GUI instance
        """
        self.gui = gui_instance

    def format_text(self, value):
        """Format text value"""
        return str(value)
    
    def format_delta(self, value):
        """Format delta value"""
        return f"{value:.3f}"
    
    def format_qty(self, value):
        """Format quantity value"""
        return f"{value:.1f}"
    
    def format_price(self, value):
        """Format price value (USD or BTC)"""
        if self.gui.btc_mode:
            return f"₿{value/self.gui.spot_price:.4f}"
        else:
            return f"${value:.2f}"
    
    def format_strike(self, value):
        """Format strike price"""
        return f"{value:.0f}"
    
    def format_percentage(self, value):
        """Format percentage value"""
        return f"{value:.2f}%"
    
    def format_gamma(self, value):
        """Format gamma value"""
        return f"{value:.6f}"
    
    def format_vega(self, value):
        """Format vega value"""
        return f"{value:.2f}"