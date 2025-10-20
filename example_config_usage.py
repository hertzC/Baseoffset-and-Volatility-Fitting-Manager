#!/usr/bin/env python3
"""
Example demonstrating the new configuration toggle functionality
for enabling/disabling wing models in volatility fitting.
"""

from config.config_loader import load_volatility_config

def main():
    # Load configuration
    vol_config = load_volatility_config()
    
    print("🔧 Volatility Model Configuration Toggles Demo")
    print("=" * 50)
    
    # Check individual model status
    print(f"Traditional Wing Model:     {'✅ ENABLED' if vol_config.wing_model_enabled else '❌ DISABLED'}")
    print(f"Time-Adjusted Wing Model:   {'✅ ENABLED' if vol_config.time_adjusted_wing_model_enabled else '❌ DISABLED'}")
    
    # Get list of enabled models
    enabled_models = vol_config.get_enabled_models()
    print(f"\nEnabled Models: {enabled_models}")
    
    # Check if specific models are enabled
    print(f"\nModel Status Checks:")
    print(f"  - Can run Traditional Wing Model: {vol_config.is_model_enabled('wing_model')}")
    print(f"  - Can run Time-Adjusted Wing Model: {vol_config.is_model_enabled('time_adjusted_wing_model')}")
    
    print(f"\n💡 To change these settings, edit config/volatility_config.yaml")
    print(f"   Under the 'enabled_models' section:")
    print(f"   wing_model: true/false")
    print(f"   time_adjusted_wing_model: true/false")

if __name__ == "__main__":
    main()