'''
@Time: 2024/10/15
@Author: Demo Example
@Contact: 
@File: time_adjusted_wing_model_example.py
@Desc: Example usage of Time-Adjusted Wing Model with class structure
'''

import numpy as np
from .time_adjusted_wing_model import TimeAdjustedWingModel, TimeAdjustedWingModelParameters
from .time_adjusted_wing_model_calibrator import TimeAdjustedWingModelCalibrator


def demo_time_adjusted_wing_model():
    """Demonstrate usage of the Time-Adjusted Wing Model class structure"""
    
    print("🚀 Time-Adjusted Wing Model Demo")
    print("=" * 50)
    
    # Example 1: Create model with default parameters
    print("\n1️⃣ Creating model with default parameters:")
    model_default = TimeAdjustedWingModel()
    print(f"   Default ATM Vol: {model_default.parameters.atm_vol}")
    print(f"   Default Forward: {model_default.parameters.forward_price}")
    
    # Example 2: Create model with custom parameters
    print("\n2️⃣ Creating model with custom parameters:")
    custom_params = TimeAdjustedWingModelParameters(
        atm_vol=0.75,           # 75% ATM volatility
        slope=-0.1,             # Negative skew
        call_curve=0.3,           # Upside curvature
        put_curve=0.5,         # Downside curvature  
        up_cutoff=1.2,             # Upside cutoff
        down_cutoff=-1.0,            # Downside cutoff
        up_smoothing=0.4,              # Upside smoothing
        down_smoothing=0.6,              # Downside smoothing
        forward_price=60000.0,  # BTC forward price
        time_to_expiry=0.25     # 3 months to expiry
    )
    
    model_custom = TimeAdjustedWingModel(custom_params)
    print(f"   Custom ATM Vol: {model_custom.parameters.atm_vol}")
    print(f"   Custom Forward: {model_custom.parameters.forward_price}")
    
    # Example 3: Calculate volatilities for different strikes
    print("\n3️⃣ Calculating volatilities for different strikes:")
    strikes = [50000, 55000, 60000, 65000, 70000]  # Range of strikes
    
    for strike in strikes:
        vol = model_custom.calculate_volatility_from_strike(strike)
        moneyness = model_custom.calculate_moneyness(
            model_custom.parameters.forward_price, strike,
            model_custom.parameters.time_to_expiry, model_custom.parameters.atm_vol
        )
        print(f"   Strike {strike:>6}: Vol {vol:.3f} | Moneyness {moneyness:.3f}")
    
    # Example 4: Generate volatility surface
    print("\n4️⃣ Generating volatility surface:")
    strikes_array, vols_array = model_custom.generate_volatility_surface(
        strike_range=(45000, 75000), num_strikes=11
    )
    
    print("   Strike -> Volatility:")
    for strike, vol in zip(strikes_array, vols_array):
        print(f"   {strike:>7.0f} -> {vol:.3f}")
    
    # Example 5: Check arbitrage condition
    print("\n5️⃣ Checking Durrleman arbitrage condition:")
    log_moneyness, g_values = model_custom.calculate_durrleman_condition()
    min_g = np.min(g_values)
    arbitrage_free = min_g >= 0
    
    print(f"   Min g-value: {min_g:.6f}")
    print(f"   Arbitrage-free: {'✅ Yes' if arbitrage_free else '❌ No'}")
    
    # Example 6: Model calibration (conceptual - would need market data)
    print("\n6️⃣ Model calibration example:")
    print("   # Synthetic market data for demo")
    market_strikes = [50000, 55000, 60000, 65000, 70000]
    market_vols = [0.82, 0.78, 0.75, 0.73, 0.71]  # Decreasing vol with strike
    market_vegas = [100, 120, 140, 120, 100]      # Vega profile
    
    calibrator = TimeAdjustedWingModelCalibrator(enable_bounds=True)
    
    print("   Calibration setup ready with:")
    print(f"   - {len(market_strikes)} market data points")
    print(f"   - Vol range: {min(market_vols):.3f} - {max(market_vols):.3f}")
    print(f"   - Forward: {custom_params.forward_price}")
    print(f"   - Time to expiry: {custom_params.time_to_expiry}")
    
    # Note: Actual calibration would be:
    # result = calibrator.calibrate(
    #     strike_list=market_strikes,
    #     market_vol_list=market_vols,
    #     market_vega_list=market_vegas,
    #     forward_price=60000.0,
    #     time_to_expiry=0.25
    # )
    
    print("\n✅ Demo completed! Time-Adjusted Wing Model is ready for use.")


if __name__ == "__main__":
    demo_time_adjusted_wing_model()