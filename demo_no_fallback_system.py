#!/usr/bin/env python3
"""
Demo: No-Fallback Configuration System

Demonstrates the enhanced configuration system that relies entirely on
component-based architecture without any fallback mechanisms.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import VolatilityConfig

def main():
    print("=" * 80)
    print("Configuration System Demo - No Fallbacks")
    print("=" * 80)
    
    # Load configuration
    config = VolatilityConfig()
    print(f"✅ Loaded: {config}")
    
    print(f"\n📦 Available Components:")
    for name, component in config.get_all_components().items():
        print(f"   {name}: {type(component).__name__}")
    
    print(f"\n🔧 Component-Based Property Access:")
    print(f"   📅 Date: {config.date_str}")
    print(f"   📊 Min Strikes: {config.min_strikes}")
    print(f"   🎯 Calibration Method: {config.calibration_method}")
    print(f"   🧠 Models Enabled: {config.get_enabled_models()}")
    print(f"   📈 Plot Theme: {config.plot_theme}")
    
    print(f"\n🎨 Direct Component Access:")
    print(f"   Data Component: {config.data.date_str}")
    print(f"   Market Data Component: {config.market_data.min_strikes}")
    print(f"   Calibration Component: {config.calibration.method}")
    print(f"   Models Component: {config.models.get_enabled_models()}")
    print(f"   Output Component: {config.output.plot_theme}")
    
    print(f"\n🚫 Demonstrating No-Fallback Behavior:")
    try:
        # Store original component
        original_data = config._components['data']
        config._components['data'] = None
        
        # This will now fail without fallbacks
        date_str = config.date_str
        print(f"   ❌ Unexpected: Got {date_str}")
        
    except AttributeError as e:
        print(f"   ✅ Expected: {e}")
        
    except Exception as e:
        print(f"   ✅ Expected error: {type(e).__name__}: {e}")
        
    finally:
        # Restore component
        config._components['data'] = original_data
        print(f"   🔄 Restored data component")
        print(f"   ✅ Date accessible again: {config.date_str}")
    
    print(f"\n=" * 80)
    print("✅ Demo Complete: System now relies entirely on components!")
    print("✅ No fallback mechanisms - Clean component-based architecture")
    print("=" * 80)

if __name__ == "__main__":
    main()