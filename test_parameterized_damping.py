#!/usr/bin/env python3
"""
Test the parameterized damping configuration in find_vol function
"""

import sys
import os
sys.path.append('/home/user/Python/Baseoffset-Fitting-Manager')

from utils.pricer.pricer_helper import find_vol
import numpy as np

def test_parameterized_damping():
    """Test the parameterized damping configuration."""
    print("⚙️ Testing Parameterized Damping Configuration")
    print("=" * 55)
    
    # Test data
    F = 100
    K = np.array([80, 100, 150])  # ITM, ATM, OTM
    T = 0.1  # ~36 days
    r = 0.05
    prices = np.array([19.5, 5.8, 0.8])
    
    print("Base Test Case:")
    print(f"  Forward: ${F}")
    print(f"  Strikes: {K}")
    print(f"  Prices: {prices}")
    print(f"  Time: {T:.3f}y ({T*365:.0f} days)")
    print()
    
    # Test 1: Default parameters
    print("1. Default Damping Parameters:")
    default_ivs = find_vol(prices, F, K, T, r, 'C')
    print(f"   Result IVs: {default_ivs}")
    print(f"   Range: {np.min(default_ivs):.1%} - {np.max(default_ivs):.1%}")
    print()
    
    # Test 2: Conservative damping (more stable, slower convergence)
    print("2. Conservative Damping (Heavy damping everywhere):")
    conservative_ivs = find_vol(
        prices, F, K, T, r, 'C',
        base_damping=0.3,                    # Reduced from 0.7
        extreme_moneyness_damping=0.1,       # Reduced from 0.3
        large_update_damping=0.2,            # Reduced from 0.4
        max_update_per_iter=0.1              # Reduced from 0.3
    )
    print(f"   Result IVs: {conservative_ivs}")
    print(f"   Range: {np.min(conservative_ivs):.1%} - {np.max(conservative_ivs):.1%}")
    print()
    
    # Test 3: Aggressive damping (less stable, faster convergence)
    print("3. Aggressive Damping (Light damping for speed):")
    aggressive_ivs = find_vol(
        prices, F, K, T, r, 'C',
        base_damping=0.9,                    # Increased from 0.7
        extreme_moneyness_damping=0.6,       # Increased from 0.3
        large_update_damping=0.8,            # Increased from 0.4
        max_update_per_iter=0.5,             # Increased from 0.3
        min_iteration_factor=0.5             # Increased from 0.3
    )
    print(f"   Result IVs: {aggressive_ivs}")
    print(f"   Range: {np.min(aggressive_ivs):.1%} - {np.max(aggressive_ivs):.1%}")
    print()
    
    # Test 4: Custom moneyness bounds
    print("4. Custom Moneyness Bounds (Tighter extreme detection):")
    tight_bounds_ivs = find_vol(
        prices, F, K, T, r, 'C',
        moneyness_bounds=(0.8, 1.2),         # Tighter bounds (was 0.5, 2.0)
        extreme_moneyness_damping=0.4        # Moderate damping for "extreme" cases
    )
    print(f"   Moneyness values: {F/K}")
    print(f"   Result IVs: {tight_bounds_ivs}")
    print(f"   Range: {np.min(tight_bounds_ivs):.1%} - {np.max(tight_bounds_ivs):.1%}")
    print()
    
    # Test 5: Custom volatility bounds
    print("5. Custom Volatility Bounds (Crypto-friendly range):")
    crypto_bounds_ivs = find_vol(
        prices, F, K, T, r, 'C',
        vol_bounds=(0.1, 3.0),              # 10% to 300% (was 1% to 500%)
        initial_vola=0.8                     # Start with higher guess for crypto
    )
    print(f"   Result IVs: {crypto_bounds_ivs}")
    print(f"   Range: {np.min(crypto_bounds_ivs):.1%} - {np.max(crypto_bounds_ivs):.1%}")
    print()
    
    # Test 6: Short expiry scenario
    print("6. Short Expiry with Custom Parameters:")
    short_T = 2/365  # 2 days
    short_prices = np.array([19.8, 0.2, 0.01])
    
    short_ivs = find_vol(
        short_prices, F, K, short_T, r, 'C',
        short_expiry_days=1.0,               # Consider < 1 day as "short" (was 7)
        short_expiry_damping=0.1,            # Very heavy damping (was 0.2)
        min_time_hours=0.5                   # Allow very short expiries (was 2.0)
    )
    print(f"   Time: {short_T:.4f}y ({short_T*365:.1f} days)")
    print(f"   Prices: {short_prices}")
    print(f"   Result IVs: {short_ivs}")
    print()
    
    print("=" * 55)
    print("📋 Summary of Parameterized Controls:")
    print("")
    print("Core Algorithm:")
    print("  • max_iterations: Maximum Newton-Raphson iterations")
    print("  • precision: Convergence tolerance")
    print("  • initial_vola: Starting volatility guess")
    print("  • vol_bounds: (min, max) volatility bounds")
    print("  • min_time_hours: Minimum time for processing")
    print("")
    print("Damping Control:")
    print("  • base_damping: Standard damping factor")
    print("  • extreme_moneyness_damping: For deep ITM/OTM")
    print("  • short_expiry_damping: For near-expiry options")
    print("  • large_update_damping: For big volatility jumps")
    print("")
    print("Thresholds:")
    print("  • moneyness_bounds: (min, max) for normal moneyness")
    print("  • short_expiry_days: Days threshold for short expiry")
    print("  • max_update_per_iter: Maximum vol change per step")
    print("  • large_update_threshold: Threshold for large updates")
    print("  • min_iteration_factor: Minimum progressive damping")
    print("")
    print("✅ All damping parameters are now configurable!")

if __name__ == "__main__":
    test_parameterized_damping()