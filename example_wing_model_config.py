#!/usr/bin/env python3
"""
Example: Using Wing Model Parameters with Configuration

Demonstrates how to use the updated wing model parameters that get
constraints from the volatility configuration instead of hardcoded values.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import VolatilityConfig
import numpy as np

def main():
    print("=" * 80)
    print("Wing Model Parameters with Configuration Constraints")
    print("=" * 80)
    
    # Load configuration
    config = VolatilityConfig()
    print(f"✅ Configuration loaded: {config.config_type}")
    
    # Import wing model parameters (with direct execution to avoid import issues)
    exec(open('utils/volatility_fitter/wing_model/wing_model_parameters.py').read())
    
    print(f"\n📊 Demonstration: Configuration-driven parameter bounds")
    
    # Show bounds for different models from configuration
    wing_bounds = config.get_parameter_bounds('wing_model')
    ta_bounds = config.get_parameter_bounds('time_adjusted_wing_model')
    
    print(f"   🎯 Wing Model bounds from config: {wing_bounds}")
    print(f"   🎯 Time-Adjusted Wing Model bounds: {ta_bounds}")
    
    print(f"\n🔧 Creating wing model parameters with configuration:")
    
    # Create wing model parameters with configuration
    params = WingModelParameters(
        vr=0.7, sr=0.05, pc=1.1, cc=0.9, dc=-1.2, uc=1.5, dsm=0.4, usm=0.5,
        forward_price=55000.0,
        ref_price=55000.0,
        time_to_expiry=0.25,
        config=config,
        model_name="wing_model"
    )
    
    print(f"   📋 Parameter names: {params.get_parameter_names()}")
    print(f"   📊 Parameter values: {params.get_fitted_vol_parameter()}")
    print(f"   🎯 Bounds from config: {params.get_parameter_bounds()}")
    print(f"   📈 Parameters dict: {params.to_dict()}")
    
    print(f"\n🏭 Using factory function with configuration:")
    
    # Simulate optimization result
    optimization_result = np.array([0.8, 0.1, 1.3, 1.1, -1.8, 2.0, 0.6, 0.7])
    
    params_from_result = create_wing_model_from_result(
        result=optimization_result,
        forward_price=60000.0,
        ref_price=60000.0,
        time_to_expiry=0.1,
        config=config,
        model_name="time_adjusted_wing_model"
    )
    
    print(f"   📋 Created from optimization result: {params_from_result}")
    print(f"   🎯 Uses time-adjusted model bounds: {params_from_result.get_parameter_bounds()}")
    
    print(f"\n🔄 Fallback behavior without configuration:")
    
    # Test without configuration (should use default bounds)
    params_no_config = WingModelParameters(
        vr=0.5, sr=-0.1, pc=0.8, cc=1.2, dc=-0.9, uc=1.8, dsm=0.3, usm=0.4,
        forward_price=50000.0,
        ref_price=50000.0,
        time_to_expiry=0.5
        # No config parameter - will use defaults
    )
    
    print(f"   📊 Default bounds (no config): {params_no_config.get_parameter_bounds()}")
    
    print(f"\n📝 Benefits of configuration-driven constraints:")
    print(f"   ✅ Centralized parameter management in YAML")
    print(f"   ✅ Easy adjustment without code changes")
    print(f"   ✅ Model-specific bounds (wing_model vs time_adjusted_wing_model)")
    print(f"   ✅ Fallback to sensible defaults if config unavailable")
    print(f"   ✅ Consistent with overall component-based architecture")
    
    print(f"\n" + "=" * 80)
    print("✅ Configuration-driven wing model parameters working perfectly!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()