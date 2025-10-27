"""
Option Chain Table Tab - Handles the option data table with BTC toggle functionality
"""

from random import random
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
            text="BTC Denomination", 
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

    def _create_column_config(self) -> list:
        """Create column configuration for the data table"""
        # Get currency symbol
        curr_symbol = "B" if self.gui.btc_mode else "$"
        
        # Define column configuration
        return [
            ('Exch', 'exchange', 40, self.gui._format_text),
            ('Expiry', 'expiry', 40, self.gui._format_text),
            ('OI_C', 'open_interest_C', 40, self.gui._format_qty),
            ('TO_C', 'turnover_C', 40, self.gui._format_qty),
            ('Delta', 'delta', 40, self.gui._format_delta),
            ('B#_C', 'bq0_C', 60, self.gui._format_qty),
            (f'Bid_C({curr_symbol})', 'bp0_C_usd', 80, self.gui._format_price),
            (f'TV_C({curr_symbol})', 'tv_C', 80, self.gui._format_price),
            (f'Ask_C({curr_symbol})', 'ap0_C_usd', 80, self.gui._format_price),
            ('A#_C', 'aq0_C', 60, self.gui._format_qty),
            ('Pos_C', 'position_C', 40, self.gui._format_qty),
            ('Strike', 'strike', 80, self.gui._format_strike),
            ('Fit IV%', 'fitVola', 70, self.gui._format_percentage),
            ('Pos_P', 'position_P', 40, self.gui._format_qty),
            ('B#_P', 'bq0_P', 60, self.gui._format_qty),
            (f'Bid_P({curr_symbol})', 'bp0_P_usd', 80, self.gui._format_price),
            (f'TV_P({curr_symbol})', 'tv_P', 80, self.gui._format_price),
            (f'Ask_P({curr_symbol})', 'ap0_P_usd', 80, self.gui._format_price),
            ('A#_P', 'aq0_P', 60, self.gui._format_qty),
            ('TO_P', 'turnover_P', 40, self.gui._format_qty),
            ('OI_P', 'open_interest_P', 40, self.gui._format_qty),
            ('Mkt IV%', 'midVola', 70, self.gui._format_percentage),
            ('SynPos', 'synthetic_position', 40, self.gui._format_qty),
            ('Gamma', 'gamma', 70, self.gui._format_gamma),
            ('Vega', 'vega', 70, self.gui._format_vega)
        ]
    
    def _create_data_table(self):
        """Create and configure the data table"""
        self.gui.column_config = self._create_column_config()
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
        
        # Configure arbitrage color tags
        self.tree.tag_configure('call_bid_arb', background='#FFE5E5', foreground='#8B0000')
        self.tree.tag_configure('call_ask_arb', background='#E5E5FF', foreground='#00008B')
        self.tree.tag_configure('put_bid_arb', background='#FFE5E5', foreground='#8B0000')
        self.tree.tag_configure('put_ask_arb', background='#E5E5FF', foreground='#00008B')
        
        self.tree.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Store tree reference in main GUI
        self.gui.tree = self.tree
        
        # Populate table
        self.populate_table()
    
    def populate_table(self):
        """Populate the table with data"""
        # Clear existing items
        columns = [col[1] for col in self.gui.column_config]
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for i, row in enumerate(self.gui.df.iter_rows()):
            values = []
            
            # Format each column value
            for _, data_source, _, format_func in self.gui.column_config:
                if data_source == 'exchange':
                    raw_value = 'deribit'
                elif data_source in ['open_interest_C','open_interest_P']:
                    raw_value = random() * 1000 / 10 if random() > 0.5 else 0  # Example placeholder for open interest
                elif data_source in ['turnover_C','turnover_P']:
                    raw_value = random() * 100 / 10 if random() > 0.5 else 0  # Example placeholder for turnover
                elif data_source in ['position_C','position_P']:
                    raw_value = (random() * 10 / 10 - 1) if random() > 0.5 else 0  # Example placeholder for position
                elif data_source == 'synthetic_position':
                    raw_value = float(values[columns.index('position_C')]) + float(values[columns.index('position_P')])
                else:
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
        """Add arbitrage indicators to price values when TV is outside bid-ask spread"""
        # Get price values for arbitrage check (always use USD)
        call_bid = row[self.gui.df.columns.index('bp0_C_usd')]
        call_tv = row[self.gui.df.columns.index('tv_C')]
        call_ask = row[self.gui.df.columns.index('ap0_C_usd')]
        put_bid = row[self.gui.df.columns.index('bp0_P_usd')]
        put_tv = row[self.gui.df.columns.index('tv_P')]
        put_ask = row[self.gui.df.columns.index('ap0_P_usd')]
        
        # Debug print for first few rows
        if len(self.tree.get_children()) < 3:
            print(f"🔍 DEBUG Row: Call TV={call_tv:.2f}, Bid={call_bid:.2f}, Ask={call_ask:.2f}")
            print(f"🔍 DEBUG Row: Put TV={put_tv:.2f}, Bid={put_bid:.2f}, Ask={put_ask:.2f}")
        
        # Find column indices for price columns
        data_sources = [col[1] for col in self.gui.column_config]
        
        # Check Call options: TV outside bid-ask spread
        if call_tv < call_bid:
            # TV is below bid - highlight both TV and bid price
            bid_idx = data_sources.index('bp0_C_usd')
            values[bid_idx] = f"[LOW] {values[bid_idx]}"
        elif call_tv > call_ask:
            # TV is above ask - highlight both TV and ask price  
            ask_idx = data_sources.index('ap0_C_usd')
            values[ask_idx] = f"[HIGH] {values[ask_idx]}"        
        # Check Put options: TV outside bid-ask spread
        if put_tv < put_bid:
            # TV is below bid - highlight both TV and bid price
            bid_idx = data_sources.index('bp0_P_usd')
            values[bid_idx] = f"[LOW] {values[bid_idx]}"
        elif put_tv > put_ask:
            # TV is above ask - highlight both TV and ask price
            ask_idx = data_sources.index('ap0_P_usd')
            values[ask_idx] = f"[HIGH] {values[ask_idx]}"
    
    def rebuild_with_currency(self):
        """Rebuild the table with current currency setting"""
        
        # Recreate column configuration with new currency symbol
        self.gui.column_config = self._create_column_config()
        
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
        tk.Label(legend_frame, text="[LOW] TV < Bid (Arbitrage)", bg='#FFE5E5', fg='#8B0000', font=('Arial', 8)).pack(side='left', padx=5)
        tk.Label(legend_frame, text="[HIGH] TV > Ask (Arbitrage)", bg='#E5E5FF', fg='#00008B', font=('Arial', 8)).pack(side='left', padx=5)