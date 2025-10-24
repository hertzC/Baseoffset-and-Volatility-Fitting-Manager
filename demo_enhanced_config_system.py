#!/usr/bin/env python3
"""
Demo: Enhanced Configuration System with Component Decomposition

This script demonstrates the improved configuration system that provides
structured access to different configuration components.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.config_loader import (
    load_config,
    get_data_component,
    get_market_data_component, 
    get_analysis_component,
    get_calibration_component,
    get_models_component,
    get_validation_component,
    get_output_component,
    get_performance_component,
    get_component_summary
)

def demo_enhanced_config_system():
    """Demonstrate the enhanced configuration system with components."""
    
    print("=" * 80)
    print("Enhanced Configuration System Demo")
    print("=" * 80)
    
    # Load configuration
    print("\n1. Loading Volatility Configuration...")
    config = load_config(config_type="volatility")
    print(f"✅ Configuration loaded: {config}")
    print(f"📁 Config path: {config.config_path}")
    print(f"🏷️ Config type: {config.config_type}")
    
    # Demonstrate component access via main config object
    print("\n2. Accessing Components via Main Config Object...")
    print("-" * 60)
    
    if config.data:
        print(f"📊 Data Component: {config.data}")
        print(f"   - Date: {config.data.date_str}")
        print(f"   - Use orderbook data: {config.data.use_orderbook_data}")
        print(f"   - Data file path: {config.data.get_data_file_path()}")
    
    if config.market_data:
        print(f"📈 Market Data Component: {config.market_data}")
        print(f"   - Min strikes: {config.market_data.min_strikes}")
        print(f"   - Strike range: {config.market_data.get_strike_range()}")
        print(f"   - Volatility range: {config.market_data.get_volatility_range()}")
    
    if config.calibration:
        print(f"🎯 Calibration Component: {config.calibration}")
        print(f"   - Method: {config.calibration.method}")
        print(f"   - Tolerance: {config.calibration.tolerance}")
        print(f"   - Weighting scheme: {config.calibration.weighting_scheme}")
    
    if config.models:
        print(f"🧠 Models Component: {config.models}")
        print(f"   - Enabled models: {config.models.get_enabled_models()}")
        print(f"   - Wing model enabled: {config.models.wing_model_enabled}")
        print(f"   - Time-adjusted enabled: {config.models.time_adjusted_wing_model_enabled}")
    
    # Demonstrate direct component access functions
    print("\n3. Direct Component Access Functions...")
    print("-" * 60)
    
    # Market data component
    market_data = get_market_data_component()
    if market_data:
        print("📊 Market Data Settings:")
        settings = market_data.get_all_settings()
        for key, value in settings.items():
            print(f"   - {key}: {value}")
    
    # Calibration component  
    calibration = get_calibration_component()
    if calibration:
        print("\n🎯 Calibration Settings:")
        settings = calibration.get_all_settings()
        for section, section_settings in settings.items():
            print(f"   {section}:")
            for key, value in section_settings.items():
                print(f"     - {key}: {value}")
    
    # Models component
    models = get_models_component()
    if models:
        print("\n🧠 Models Settings:")
        wing_settings = models.get_wing_model_settings()
        ta_wing_settings = models.get_time_adjusted_wing_model_settings()
        print(f"   Wing Model: {wing_settings}")
        print(f"   Time-Adjusted Wing Model: {ta_wing_settings}")
    
    # Output component
    output = get_output_component()
    if output:
        print("\n📊 Output Settings:")
        plotting_settings = output.get_plotting_settings()
        export_settings = output.get_export_settings()
        print(f"   Plotting: {plotting_settings}")
        print(f"   Export: {export_settings}")
    
    # Component summary
    print("\n4. Complete Component Summary...")
    print("-" * 60)
    
    summary = get_component_summary()
    for component_name, component_settings in summary.items():
        print(f"🔧 {component_name.upper()} Component Summary:")
        if isinstance(component_settings, dict) and component_settings:
            for key, value in component_settings.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"     - {sub_key}: {sub_value}")
                else:
                    print(f"   - {key}: {value}")
        print()
    
    # Demonstrate backward compatibility
    print("\n5. Backward Compatibility Check...")
    print("-" * 60)
    
    # Old style access should still work
    print("✅ Old style access still works:")
    print(f"   - config.time_adjusted_wing_model_enabled: {config.time_adjusted_wing_model_enabled}")
    print(f"   - config.calibration_method: {config.calibration_method}")
    print(f"   - config.min_strikes: {config.min_strikes}")
    
    # New style access
    print("✅ New style component access:")
    print(f"   - config.models.time_adjusted_wing_model_enabled: {config.models.time_adjusted_wing_model_enabled}")
    print(f"   - config.calibration.method: {config.calibration.method}")
    print(f"   - config.market_data.min_strikes: {config.market_data.min_strikes}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully! 🎉")
    print("=" * 80)

if __name__ == "__main__":
    try:
        demo_enhanced_config_system()
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()