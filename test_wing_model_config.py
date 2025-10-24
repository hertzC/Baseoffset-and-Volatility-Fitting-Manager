#!/usr/bin/env python3
"""
Test updated wing model parameters with configuration
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import VolatilityConfig
from utils.volatility_fitter.wing_model.wing_model_parameters import WingModelParameters, create_wing_model_from_result
import numpy as np

def test_wing_model_with_config():
    print("=" * 60)
    print("Testing Wing Model Parameters with Configuration")
    print("=" * 60)
    
    # Load configuration
    config = VolatilityConfig()
    print(f"✅ Configuration loaded: {config.config_type}")
    
    # Test parameter bounds from configuration
    print(f"\n📊 Testing parameter bounds from configuration...")
    
    # Create a wing model parameters instance with config
    params = WingModelParameters(
        vr=0.8, sr=0.1, pc=1.2, cc=1.0, dc=-1.5, uc=1.8, dsm=0.5, usm=0.6,
        forward_price=50000.0, ref_price=50000.0, time_to_expiry=0.25,
        config=config, model_name="wing_model"
    )
    
    print(f"   📋 Parameter names: {params.get_parameter_names()}")
    print(f"   📊 Parameter values: {params.get_fitted_vol_parameter()}")
    
    # Get bounds from configuration
    bounds = params.get_parameter_bounds()
    print(f"   🎯 Parameter bounds: {bounds}")
    
    # Test time-adjusted wing model bounds
    params_ta = WingModelParameters(
        vr=0.8, sr=0.1, pc=1.2, cc=1.0, dc=-1.5, uc=1.8, dsm=0.5, usm=0.6,
        forward_price=50000.0, ref_price=50000.0, time_to_expiry=0.25,
        config=config, model_name="time_adjusted_wing_model"
    )
    
    bounds_ta = params_ta.get_parameter_bounds()
    print(f"   🎯 Time-adjusted bounds: {bounds_ta}")
    
    # Test factory function with config
    print(f"\n🏭 Testing factory function with configuration...")
    
    result = np.array([0.7, 0.05, 1.1, 0.9, -1.2, 1.5, 0.4, 0.5])
    
    params_from_result = create_wing_model_from_result(
        result=result,
        forward_price=55000.0,
        ref_price=55000.0,
        time_to_expiry=0.1,
        config=config,
        model_name="time_adjusted_wing_model"
    )
    
    print(f"   📋 Created from result: {params_from_result}")
    print(f"   🎯 Bounds: {params_from_result.get_parameter_bounds()}")
    
    # Test without configuration (fallback)
    print(f"\n🔄 Testing fallback without configuration...")
    
    params_no_config = WingModelParameters(
        vr=0.8, sr=0.1, pc=1.2, cc=1.0, dc=-1.5, uc=1.8, dsm=0.5, usm=0.6,
        forward_price=50000.0, ref_price=50000.0, time_to_expiry=0.25
        # No config provided
    )
    
    bounds_default = params_no_config.get_parameter_bounds()
    print(f"   📊 Default bounds: {bounds_default}")
    
    print(f"\n✅ All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_wing_model_with_config()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()