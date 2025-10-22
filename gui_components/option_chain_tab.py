"""
Option Chain Table Tab - Handles the option data table with BTC toggle functionality
"""

import tkinter as tk
from tkinter import ttk


class OptionChainTab:
    """Handles the option chain table tab with arbitrage detection and BTC toggle"""
    
    def __init__(self, parent_notebook, gui_instance):
        """
        Initialize the option chain tab
        
        Args:
            parent_notebook: The main notebook widget to add this tab to
            gui_instance: Reference to the main GUI instance for data access
        """
        self.notebook = parent_notebook
        self.gui = gui_instance
        self.tab_frame = None
        self.tree = None
        self.toggle_button = None
        
    def create_tab(self):
        """Create and configure the option chain tab"""
        # Create tab frame
        self.tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_frame, text="Option Chain & Arbitrage")
        
        # Create tab components
        self._create_title()
        self._create_control_frame()
        self._create_data_table()
        self._create_legend()
        
        return self.tab_frame
    
    def _create_title(self):
        """Create the title label"""
        title_text = (f"Option Chain Analysis - {self.gui.my_expiry} @ {self.gui.snapshot_time}  "
                     f"S=${self.gui.spot_price:.0f} F=${self.gui.forward_price:.0f}")
        title_label = tk.Label(self.tab_frame, text=title_text, 
                              font=('Arial', 10, 'bold'), bg='white')
        title_label.pack(pady=10)
    
    def _create_control_frame(self):
        """Create control frame with BTC toggle"""
        control_frame = tk.Frame(self.tab_frame, bg='white')
        control_frame.pack(pady=5)
        
        # Toggle button - store reference for direct state checking
        self.toggle_button = tk.Checkbutton(
            control_frame, 
            text="₿ BTC Denomination", 
            variable=self.gui.btc_var, 
            font=('Arial', 9, 'bold'), 
            bg='white', 
            fg='orange', 
            command=self.gui._toggle_currency
        )
        self.toggle_button.pack(side='left', padx=10)
        
        # Info label
        info_label = tk.Label(
            control_frame, 
            text="(Toggle to show prices in BTC instead of USD)", 
            font=('Arial', 8), fg='gray', bg='white'
        )
        info_label.pack(side='left', padx=5)
    
    def _create_data_table(self):
        """Create and configure the data table"""
        # Get currency symbol
        curr_symbol = "₿" if self.gui.btc_mode else "$"
        
        # Define column configuration
        self.gui.column_config = [
            ('Delta', 'delta', 60, self.gui._format_delta),
            ('BidQty_Call', 'bq0_C', 70, self.gui._format_qty),
            (f'BidPx_Call({curr_symbol})', 'bp0_C_usd', 90, self.gui._format_price),
            (f'TV_Call({curr_symbol})', 'tv_C', 90, self.gui._format_price),
            (f'AskPx_Call({curr_symbol})', 'ap0_C_usd', 90, self.gui._format_price),
            ('AskQty_Call', 'aq0_C', 70, self.gui._format_qty),
            ('Strike', 'strike', 80, self.gui._format_strike),
            ('BidQty_Put', 'bq0_P', 70, self.gui._format_qty),
            (f'BidPx_Put({curr_symbol})', 'bp0_P_usd', 90, self.gui._format_price),
            (f'TV_Put({curr_symbol})', 'tv_P', 90, self.gui._format_price),
            (f'AskPx_Put({curr_symbol})', 'ap0_P_usd', 90, self.gui._format_price),
            ('AskQty_Put', 'aq0_P', 70, self.gui._format_qty),
            ('Fit IV%', 'fitVola', 80, self.gui._format_percentage),
            ('Mkt IV%', 'midVola', 80, self.gui._format_percentage),
            ('Gamma', 'gamma', 70, self.gui._format_gamma),
            ('Vega', 'vega', 70, self.gui._format_vega)
        ]
        
        columns = [col[0] for col in self.gui.column_config]
        
        # Create treeview
        self.tree = ttk.Treeview(self.tab_frame, columns=columns, show='headings', 
                                height=20, style="Small.Treeview")
        
        # Configure columns
        for i, (col_name, _, width, _) in enumerate(self.gui.column_config):
            self.tree.heading(columns[i], text=col_name)
            self.tree.column(columns[i], width=width, anchor='center')
        
        # Configure alternating row colors for gridline effect
        self.tree.tag_configure('even', background='#f8f8f8')
        self.tree.tag_configure('odd', background='white')
        
        self.tree.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Store tree reference in main GUI
        self.gui.tree = self.tree
        
        # Populate table
        self.populate_table()
    
    def populate_table(self):
        """Populate the table with data"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for i, row in enumerate(self.gui.df.iter_rows()):
            values = []
            
            # Format each column value
            for _, data_source, _, format_func in self.gui.column_config:
                col_idx = self.gui.df.columns.index(data_source)
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
        call_bid = row[self.gui.df.columns.index('bp0_C_usd')]
        call_tv = row[self.gui.df.columns.index('tv_C')]
        call_ask = row[self.gui.df.columns.index('ap0_C_usd')]
        put_bid = row[self.gui.df.columns.index('bp0_P_usd')]
        put_tv = row[self.gui.df.columns.index('tv_P')]
        put_ask = row[self.gui.df.columns.index('ap0_P_usd')]
        
        # Find column indices for price columns
        data_sources = [col[1] for col in self.gui.column_config]
        
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
    
    def rebuild_with_currency(self):
        """Rebuild the table with current currency setting"""
        # Get currency symbol
        curr_symbol = "₿" if self.gui.btc_mode else "$"
        
        # Recreate column configuration with new currency symbol
        self.gui.column_config = [
            ('Delta', 'delta', 60, self.gui._format_delta),
            ('BidQty_Call', 'bq0_C', 70, self.gui._format_qty),
            (f'BidPx_Call({curr_symbol})', 'bp0_C_usd', 90, self.gui._format_price),
            (f'TV_Call({curr_symbol})', 'tv_C', 90, self.gui._format_price),
            (f'AskPx_Call({curr_symbol})', 'ap0_C_usd', 90, self.gui._format_price),
            ('AskQty_Call', 'aq0_C', 70, self.gui._format_qty),
            ('Strike', 'strike', 80, self.gui._format_strike),
            ('BidQty_Put', 'bq0_P', 70, self.gui._format_qty),
            (f'BidPx_Put({curr_symbol})', 'bp0_P_usd', 90, self.gui._format_price),
            (f'TV_Put({curr_symbol})', 'tv_P', 90, self.gui._format_price),
            (f'AskPx_Put({curr_symbol})', 'ap0_P_usd', 90, self.gui._format_price),
            ('AskQty_Put', 'aq0_P', 70, self.gui._format_qty),
            ('Fit IV%', 'fitVola', 80, self.gui._format_percentage),
            ('Mkt IV%', 'midVola', 80, self.gui._format_percentage),
            ('Gamma', 'gamma', 70, self.gui._format_gamma),
            ('Vega', 'vega', 70, self.gui._format_vega)
        ]
        
        # Update column headers
        for i, (col_name, _, _, _) in enumerate(self.gui.column_config):
            self.tree.heading(self.tree['columns'][i], text=col_name)
        
        # Repopulate table with new formatting
        self.populate_table()
    
    def _create_legend(self):
        """Create legend for arbitrage indicators"""
        legend_frame = tk.Frame(self.tab_frame, bg='white')
        legend_frame.pack(pady=10)
        
        tk.Label(legend_frame, text="Legend: ", font=('Arial', 8, 'bold'), bg='white').pack(side='left')
        tk.Label(legend_frame, text="🔴 Low BidPx", bg='#FFE5E5', fg='#8B0000', font=('Arial', 8)).pack(side='left', padx=5)
        tk.Label(legend_frame, text="🔵 High AskPx", bg='#E5E5FF', fg='#00008B', font=('Arial', 8)).pack(side='left', padx=5)