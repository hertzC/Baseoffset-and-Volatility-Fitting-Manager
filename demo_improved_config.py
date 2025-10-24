#!/usr/bin/env python3
"""
Configuration System Demonstration

This script demonstrates the improved configuration system with
separate modules for different analysis types.

Shows both the new modular approach and legacy compatibility.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def demonstrate_new_configuration_system():
    """Demonstrate the new modular configuration system."""
    
    print("🔧 Improved Configuration System Demonstration")
    print("=" * 60)
    
    # Method 1: Direct class usage (recommended)
    print("\n📋 Method 1: Direct Configuration Classes (Recommended)")
    print("-" * 50)
    
    try:
        from config import BaseOffsetConfig, VolatilityConfig
        
        # Load base offset configuration
        base_config = BaseOffsetConfig()
        print(f"✅ Base Offset Config: {base_config}")
        print(f"   📅 Date: {base_config.date_str}")
        print(f"   ⚙️ Constrained Optimization: {base_config.use_constrained_optimization}")
        print(f"   📊 Data Source: {'OrderBook' if base_config.use_orderbook_data else 'BBO'}")
        print(f"   🔧 Conflation: {base_config.conflation_every}/{base_config.conflation_period}")
        
        # Show rate constraints
        rate_constraints = base_config.get_rate_constraints()
        print(f"   📈 Rate Constraints: r∈[{rate_constraints['r_min']:.1%}, {rate_constraints['r_max']:.1%}]")
        print(f"                        q∈[{rate_constraints['q_min']:.1%}, {rate_constraints['q_max']:.1%}]")
        
        # Load volatility configuration
        vol_config = VolatilityConfig()
        print(f"\n✅ Volatility Config: {vol_config}")
        print(f"   📅 Date: {vol_config.date_str}")
        print(f"   🎯 Min Strikes: {vol_config.min_strikes}")
        print(f"   📊 Strike Ratio: {vol_config.min_strike_ratio:.1%} - {vol_config.max_strike_ratio:.1%}")
        print(f"   ⚙️ Calibration Method: {vol_config.calibration_method}")
        print(f"   🔧 Max RMSE: {vol_config.max_rmse_threshold:.1%}")
        
        # Show enabled models
        enabled_models = vol_config.get_enabled_models()
        print(f"   🧮 Enabled Models: {', '.join(enabled_models)}")
        
        # Show model parameters for time-adjusted wing model
        if vol_config.time_adjusted_wing_model_enabled:
            bounds = vol_config.get_parameter_bounds('time_adjusted_wing_model')
            print(f"   📏 Time-Adjusted Wing Model Bounds: {len(bounds)} parameters")
            
    except Exception as e:
        print(f"❌ Error with direct classes: {e}")
    
    # Method 2: Factory pattern
    print(f"\n🏭 Method 2: Configuration Factory")
    print("-" * 50)
    
    try:
        from config import ConfigFactory
        
        # Create configurations using factory
        base_config_factory = ConfigFactory.create_config("base_offset")
        vol_config_factory = ConfigFactory.create_config("volatility")
        
        print(f"✅ Factory Base Config: {type(base_config_factory).__name__}")
        print(f"✅ Factory Volatility Config: {type(vol_config_factory).__name__}")
        
        # Show auto-detection
        auto_type = ConfigFactory.auto_detect_config_type("volatility_config.yaml")
        print(f"🔍 Auto-detected type for 'volatility_config.yaml': {auto_type}")
        
    except Exception as e:
        print(f"❌ Error with factory: {e}")
    
    # Method 3: Legacy compatibility (deprecated but still works)
    print(f"\n⚠️  Method 3: Legacy Compatibility (Deprecated)")
    print("-" * 50)
    
    try:
        from config.config_loader import load_config, Config
        
        # Legacy way still works
        legacy_config = load_config(config_type="base_offset")
        print(f"✅ Legacy Config: {type(legacy_config).__name__}")
        print(f"   📅 Date: {legacy_config.date_str}")
        print("   ⚠️  This method shows deprecation warning")
        
    except Exception as e:
        print(f"❌ Error with legacy: {e}")
    
    # Method 4: Configuration comparison
    print(f"\n🔄 Method 4: Configuration Type Comparison")
    print("-" * 50)
    
    try:
        from config import BaseOffsetConfig, VolatilityConfig
        
        base_config = BaseOffsetConfig()
        vol_config = VolatilityConfig()
        
        print("Base Offset specific properties:")
        print(f"  • future_spread_mult: {base_config.future_spread_mult}")
        print(f"  • lambda_reg: {base_config.lambda_reg}")
        print(f"  • old_weight: {base_config.old_weight}")
        print(f"  • cutoff_hour_for_0DTE: {base_config.cutoff_hour_for_0DTE}")
        
        print("\nVolatility specific properties:")
        print(f"  • calibration_tolerance: {vol_config.calibration_tolerance}")
        print(f"  • arbitrage_penalty: {vol_config.arbitrage_penalty}")
        print(f"  • durrleman_num_points: {vol_config.durrleman_num_points}")
        print(f"  • use_norm_term: {vol_config.use_norm_term}")
        
        print("\nCommon properties (both inherit from BaseConfig):")
        print(f"  • date_str: {base_config.date_str} == {vol_config.date_str}")
        print(f"  • use_orderbook_data: {base_config.use_orderbook_data} == {vol_config.use_orderbook_data}")
        print(f"  • default_interest_rate: {base_config.default_interest_rate} == {vol_config.default_interest_rate}")
        
    except Exception as e:
        print(f"❌ Error with comparison: {e}")

def demonstrate_benefits():
    """Demonstrate the benefits of the new system."""
    
    print(f"\n🎯 Benefits of the New Configuration System")
    print("=" * 60)
    
    print("✅ Type Safety:")
    print("   • Each config type has its own class with proper type hints")
    print("   • IDE autocomplete works properly for config-specific properties")
    print("   • Compile-time detection of invalid property access")
    
    print("\n✅ Separation of Concerns:")
    print("   • Base offset analysis configs separated from volatility configs")
    print("   • Each module handles its own validation logic")
    print("   • Easier to maintain and extend")
    
    print("\n✅ Backward Compatibility:")
    print("   • Existing code continues to work unchanged")
    print("   • Gradual migration path available")
    print("   • Legacy Config class delegated to new system")
    
    print("\n✅ Better Organization:")
    print("   • Common functionality in BaseConfig")
    print("   • Specialized properties in subclasses")
    print("   • Factory pattern for dynamic creation")
    
    print("\n✅ Enhanced Validation:")
    print("   • Type-specific validation rules")
    print("   • Better error messages")
    print("   • Modular validation logic")

def show_migration_examples():
    """Show how to migrate existing code."""
    
    print(f"\n🔄 Migration Examples")
    print("=" * 60)
    
    print("Old way (still works but deprecated):")
    print("```python")
    print("from config.config_loader import load_config")
    print("config = load_config(config_type='base_offset')")
    print("rate_constraints = config.get_rate_constraints()")
    print("```")
    
    print("\nNew way (recommended):")
    print("```python")
    print("from config import BaseOffsetConfig")
    print("config = BaseOffsetConfig()")
    print("rate_constraints = config.get_rate_constraints()")
    print("```")
    
    print("\nFor volatility analysis:")
    print("```python")
    print("from config import VolatilityConfig")
    print("config = VolatilityConfig()")
    print("bounds = config.get_parameter_bounds('wing_model')")
    print("```")
    
    print("\nUsing factory for dynamic creation:")
    print("```python")
    print("from config import ConfigFactory")
    print("config = ConfigFactory.create_config('volatility')")
    print("enabled_models = config.get_enabled_models()")
    print("```")

if __name__ == "__main__":
    try:
        demonstrate_new_configuration_system()
        demonstrate_benefits()
        show_migration_examples()
        
        print(f"\n🎉 Configuration System Demonstration Complete!")
        print("   New modular system is ready to use alongside existing code.")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()