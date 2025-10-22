#!/usr/bin/env python3
"""
Test script to debug BTC toggle functionality
"""

import tkinter as tk
from tkinter import ttk

class BTCToggleTest:
    def __init__(self):
        self.btc_mode = False
        self.spot_price = 63000
        
        self.root = tk.Tk()
        self.root.title("BTC Toggle Test")
        self.root.geometry("400x200")
        
        # Initialize BTC variable AFTER root window
        self.btc_var = tk.BooleanVar(value=False)
        
        self.create_gui()
        
    def create_gui(self):
        # Test checkbox
        self.checkbox = tk.Checkbutton(
            self.root, 
            text="₿ BTC Mode", 
            variable=self.btc_var,
            command=self.toggle_currency
        )
        self.checkbox.pack(pady=10)
        
        # Status labels
        self.status_label = tk.Label(self.root, text="Status: USD Mode")
        self.status_label.pack(pady=5)
        
        self.var_label = tk.Label(self.root, text="btc_var: False")
        self.var_label.pack(pady=5)
        
        self.mode_label = tk.Label(self.root, text="btc_mode: False")
        self.mode_label.pack(pady=5)
        
        # Test price formatting
        self.price_label = tk.Label(self.root, text="Test Price: $100.00")
        self.price_label.pack(pady=5)
        
        # Manual test button
        test_button = tk.Button(self.root, text="Manual Test", command=self.manual_test)
        test_button.pack(pady=10)
        
    def toggle_currency(self):
        """Test the toggle functionality"""
        print("=" * 50)
        print("TOGGLE CALLED!")
        
        # Check checkbox state
        checkbox_state = self.btc_var.get()
        print(f"Checkbox state (btc_var.get()): {checkbox_state}")
        
        # Update internal state
        self.btc_mode = checkbox_state
        print(f"Updated btc_mode to: {self.btc_mode}")
        
        # Update UI
        self.update_display()
        
    def manual_test(self):
        """Manual test to verify state"""
        print("\n" + "=" * 50)
        print("MANUAL TEST:")
        print(f"btc_var.get(): {self.btc_var.get()}")
        print(f"btc_mode: {self.btc_mode}")
        print(f"Checkbox selected: {self.checkbox.instate(['selected'])}")
        
        # Test formatting
        test_price = 100.0
        formatted = self.format_price(test_price)
        print(f"Formatted price: {formatted}")
        
    def format_price(self, value):
        """Test price formatting"""
        if self.btc_mode:
            return f"₿{value/self.spot_price:.6f}"
        else:
            return f"${value:.2f}"
            
    def update_display(self):
        """Update all display elements"""
        curr_symbol = "₿" if self.btc_mode else "$"
        mode_text = "BTC" if self.btc_mode else "USD"
        
        self.status_label.config(text=f"Status: {mode_text} Mode")
        self.var_label.config(text=f"btc_var: {self.btc_var.get()}")
        self.mode_label.config(text=f"btc_mode: {self.btc_mode}")
        
        # Test price formatting
        test_price = 100.0
        formatted_price = self.format_price(test_price)
        self.price_label.config(text=f"Test Price: {formatted_price}")
        
    def run(self):
        print("Starting BTC Toggle Test...")
        print(f"Initial state - btc_var: {self.btc_var.get()}, btc_mode: {self.btc_mode}")
        self.root.mainloop()

def main():
    """Run the test"""
    print("🧪 BTC Toggle Test Starting...")
    test = BTCToggleTest()
    test.run()

if __name__ == "__main__":
    main()