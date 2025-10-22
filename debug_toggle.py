#!/usr/bin/env python3
"""
Debug script to find the exact issue with BTC toggle
"""

import tkinter as tk

class DebugToggle:
    def __init__(self):
        self.btc_mode = False
        self.root = tk.Tk()
        self.root.title("Debug Toggle")
        self.root.geometry("300x200")
        
        # Create BTC var AFTER root
        self.btc_var = tk.BooleanVar(value=False)
        
        # Add debug labels
        self.mode_label = tk.Label(self.root, text=f"btc_mode: {self.btc_mode}")
        self.mode_label.pack(pady=10)
        
        self.var_label = tk.Label(self.root, text=f"btc_var: {self.btc_var.get()}")
        self.var_label.pack(pady=10)
        
        # Checkbox
        self.checkbox = tk.Checkbutton(
            self.root, 
            text="₿ Toggle", 
            variable=self.btc_var,
            command=self.toggle_debug
        )
        self.checkbox.pack(pady=10)
        
        # Refresh button
        refresh_btn = tk.Button(self.root, text="Refresh Labels", command=self.refresh_labels)
        refresh_btn.pack(pady=5)
        
    def toggle_debug(self):
        print("=" * 40)
        print("TOGGLE DEBUG:")
        print(f"  Before: btc_mode={self.btc_mode}, btc_var={self.btc_var.get()}")
        
        # Update mode
        self.btc_mode = self.btc_var.get()
        
        print(f"  After:  btc_mode={self.btc_mode}, btc_var={self.btc_var.get()}")
        
        # Update labels
        self.refresh_labels()
        
    def refresh_labels(self):
        self.mode_label.config(text=f"btc_mode: {self.btc_mode}")
        self.var_label.config(text=f"btc_var: {self.btc_var.get()}")
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    debug = DebugToggle()
    debug.run()