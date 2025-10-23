"""
Option Price Constraints Module

This module provides utilities for applying no-arbitrage constraints and monotonicity 
to option prices. These constraints are essential for maintaining data quality and 
realistic implied volatility calculations.

Key Features:
- Monotonicity constraints (call bids decrease, put bids increase with strike)
- No-arbitrage bounds between adjacent strikes
- Support for both Polars DataFrame and numpy array operations
- Preservation of original data with tightened alternatives

Mathematical Foundation:
1. Monotonicity: 
   - Call bids: C_bid(K1) >= C_bid(K2) for K1 < K2
   - Put bids: P_bid(K1) <= P_bid(K2) for K1 < K2
   
2. No-arbitrage bounds:
   - |C(K1) - C(K2)| <= PV(|K2 - K1| / S) for adjacent strikes
   - Similar bounds apply to puts

Author: GitHub Copilot
Date: October 2025
"""

import math
import numpy as np
import polars as pl
from typing import Tuple


def apply_option_constraints(bid_price_call: np.ndarray, ask_price_call: np.ndarray, bid_price_put: np.ndarray, ask_price_put: np.ndarray,
                             strike: np.ndarray, spot: float, interest_rate: float, time_to_expiry: float,
                             bid_size_call: np.ndarray, ask_size_call: np.ndarray, bid_size_put: np.ndarray, ask_size_put: np.ndarray,
                             volume_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply monotonicity and no-arbitrage constraints to option prices.
    
    Ensures call bids decrease, asks increase with strike; put bids increase, asks decrease.
    Applies no-arbitrage bounds between adjacent strikes.
    Constraints applied only where volume > threshold.
            
    Returns:
        Tuple of adjusted (bid_call, ask_call, bid_put, ask_put)
    """
    bid_call = bid_price_call.astype(float)
    ask_call = ask_price_call.astype(float)
    bid_put = bid_price_put.astype(float)
    ask_put = ask_price_put.astype(float)

    # Monotonicity constraints
    for i in range(len(strike)-2, -1, -1):
        if bid_size_call[i+1] >= volume_threshold:
            bid_call[i] = max(bid_call[i], bid_call[i+1])
        if ask_size_put[i+1] >= volume_threshold:
            ask_put[i] = min(ask_put[i], ask_put[i+1])
    
    for i in range(1, len(strike)):
        if ask_size_call[i-1] >= volume_threshold:
            ask_call[i] = min(ask_call[i], ask_call[i-1])
        if bid_size_put[i-1] >= volume_threshold:
            bid_put[i] = max(bid_put[i], bid_put[i-1])
    
    # No-arbitrage constraints
    for i in range(1, len(strike)):
        strike_diff = (strike[i] - strike[i-1]) / spot  # need to divided by spot as the bid/ask price is in BTC terms
        strike_diff *= math.exp(-interest_rate * time_to_expiry)  # discount factor
        if bid_size_call[i-1] >= volume_threshold:
            bid_call[i] = max(bid_call[i], bid_call[i-1] - strike_diff)
        if ask_size_put[i-1] >= volume_threshold:
            ask_put[i] = min(ask_put[i], ask_put[i-1] + strike_diff)
    
    for i in range(len(strike)-2, -1, -1):
        strike_diff = (strike[i+1] - strike[i]) / spot
        strike_diff *= math.exp(-interest_rate * time_to_expiry)  # discount factor
        if ask_size_call[i+1] >= volume_threshold:
            ask_call[i] = min(ask_call[i], ask_call[i+1] + strike_diff)
        if bid_size_put[i+1] >= volume_threshold:
            bid_put[i] = max(bid_put[i], bid_put[i+1] - strike_diff)
    
    return bid_call, ask_call, bid_put, ask_put


def tighten_option_spreads_separate_columns(
    option_df: pl.DataFrame,
    spot_col: str = 'S',
    strike_col: str = 'strike',
    call_bid_col: str = 'bid_price',
    call_ask_col: str = 'ask_price', 
    put_bid_col: str = 'bid_price_P',
    put_ask_col: str = 'ask_price_P',
    call_bid_size_col: str = 'bid_size',
    call_ask_size_col: str = 'ask_size',
    put_bid_size_col: str = 'bid_size_P',
    put_ask_size_col: str = 'ask_size_P',
    volume_threshold: float = 0.0,
    interest_rate: float = 0.0,
    time_to_expiry_col: str = 'tau',
) -> pl.DataFrame:
    """
    Apply option spread tightening for separate call/put columns format.
    
    Handles DataFrames with call/put prices in separate columns.
    Applies constraints conditionally based on volume.
            
    Returns:
        DataFrame with tightened prices and preserved originals.
    """
    # Remove rows with nulls in critical columns
    required_cols = [call_bid_col, call_ask_col, put_bid_col, put_ask_col, spot_col, strike_col, 
                     call_bid_size_col, call_ask_size_col, put_bid_size_col, put_ask_size_col, time_to_expiry_col]
    df = option_df.drop_nulls(required_cols)
    if df.is_empty():
        return option_df
    
    # Sort by strike
    df = df.sort(strike_col)

    # Apply constraints
    tightened_bid_call, tightened_ask_call, tightened_bid_put, tightened_ask_put = apply_option_constraints(
        bid_price_call=df[call_bid_col].to_numpy(),
        ask_price_call=df[call_ask_col].to_numpy(),
        bid_price_put=df[put_bid_col].to_numpy(),
        ask_price_put=df[put_ask_col].to_numpy(),
        strike=df[strike_col].to_numpy(),
        spot=df[spot_col][0],
        interest_rate=interest_rate,
        bid_size_call=df[call_bid_size_col].to_numpy(),
        ask_size_call=df[call_ask_size_col].to_numpy(),
        bid_size_put=df[put_bid_size_col].to_numpy(),
        ask_size_put=df[put_ask_size_col].to_numpy(),
        volume_threshold=volume_threshold,
        time_to_expiry=df[time_to_expiry_col][0]
    )
    
    # Add tightened prices, preserve originals
    result = df.with_columns([
        pl.col(call_bid_col).alias(f'original_{call_bid_col}'),
        pl.col(call_ask_col).alias(f'original_{call_ask_col}'),
        pl.col(put_bid_col).alias(f'original_{put_bid_col}'),
        pl.col(put_ask_col).alias(f'original_{put_ask_col}'),
        pl.Series(f'tightened_{call_bid_col}', tightened_bid_call),
        pl.Series(f'tightened_{call_ask_col}', tightened_ask_call),
        pl.Series(f'tightened_{put_bid_col}', tightened_bid_put),
        pl.Series(f'tightened_{put_ask_col}', tightened_ask_put),
        pl.Series(call_bid_col, tightened_bid_call),
        pl.Series(call_ask_col, tightened_ask_call),
        pl.Series(put_bid_col, tightened_bid_put),
        pl.Series(put_ask_col, tightened_ask_put)
    ])
    
    # Check for bid-ask breaches and warn
    call_breach = result[call_ask_col] <= result[call_bid_col]
    put_breach = result[put_ask_col] <= result[put_bid_col]
    if call_breach.any():
        timestamp = df['timestamp'][0] if 'timestamp' in df.columns else "Unknown"
        expiry = df['expiry'][0] if 'expiry' in df.columns else "Unknown"
        print(f"❌ {timestamp}: Call bid >= ask at S: {df[spot_col][0]}, expiry: {expiry}, strikes: {result.filter(call_breach)[strike_col].to_list()}")
    if put_breach.any():
        timestamp = df['timestamp'][0] if 'timestamp' in df.columns else "Unknown"
        expiry = df['expiry'][0] if 'expiry' in df.columns else "Unknown"
        print(f"❌ {timestamp}: Put bid >= ask at S: {df[spot_col][0]}, expiry: {expiry}, strikes: {result.filter(put_breach)[strike_col].to_list()}")
    
    return result


def tighten_option_spreads_with_volume_filter(
    option_df: pl.DataFrame,
    volume_threshold: float = 10.0,
    **kwargs
) -> pl.DataFrame:
    """
    Apply option spread tightening with volume-based filtering.
    
    This is a convenience function that applies reasonable defaults for volume-based
    constraint filtering to avoid being impacted by very small lots.
    
    Args:
        option_df: DataFrame with option data
        volume_threshold: Minimum volume to apply tightening logic (default: 10.0)
        **kwargs: Additional arguments passed to tighten_option_spreads_separate_columns
        
    Returns:
        DataFrame with volume-filtered constraints applied
    """
    return tighten_option_spreads_separate_columns(
        option_df,
        volume_threshold=volume_threshold,
        **kwargs
    )


# Export main functions
__all__ = [
    'apply_option_constraints',
    'tighten_option_spreads_separate_columns',
    'tighten_option_spreads_with_volume_filter',  # convenience function
]