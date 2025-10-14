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
   - |C(K1) - C(K2)| <= |K2 - K1| / S for adjacent strikes
   - Similar bounds apply to puts

Author: GitHub Copilot
Date: October 2025
"""

import numpy as np
import polars as pl
from typing import Tuple


def apply_option_constraints(
    bid_price_call: np.ndarray, 
    ask_price_call: np.ndarray, 
    bid_price_put: np.ndarray, 
    ask_price_put: np.ndarray,
    strike: np.ndarray,
    spot: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply monotonicity and no-arbitrage constraints to option prices.
    
    This function ensures that option prices follow theoretical constraints:
    1. Call bids decrease with strike, asks increase
    2. Put bids increase with strike, asks decrease  
    3. No-arbitrage bounds between adjacent strikes
    
    Args:
        bid_price_call: Array of call bid prices
        ask_price_call: Array of call ask prices
        bid_price_put: Array of put bid prices
        ask_price_put: Array of put ask prices
        strike: Array of strike prices (must be sorted ascending)
        spot: Current spot price
        
    Returns:
        Tuple of (adjusted_bid_call, adjusted_ask_call, adjusted_bid_put, adjusted_ask_put)
        
    Example:
        >>> strikes = np.array([90, 95, 100, 105, 110])
        >>> call_bids = np.array([12, 8, 5, 3, 1])
        >>> call_asks = np.array([13, 9, 6, 4, 2])
        >>> put_bids = np.array([1, 3, 5, 8, 12])
        >>> put_asks = np.array([2, 4, 6, 9, 13])
        >>> spot = 100.0
        >>> 
        >>> adj_cb, adj_ca, adj_pb, adj_pa = apply_option_constraints(
        ...     call_bids, call_asks, put_bids, put_asks, strikes, spot
        ... )
    """
    # Make copies to avoid modifying original data
    bid_call = bid_price_call.copy()
    ask_call = ask_price_call.copy()
    bid_put = bid_price_put.copy()
    ask_put = ask_price_put.copy()
    
    # Step 1: Apply monotonicity constraints
    # Call prices: bids should decrease, asks should increase with strike
    bid_call = np.maximum.accumulate(bid_call[::-1], axis=0)[::-1]
    
    # Handle zero asks by treating them as infinite, then restoring zeros
    ask_call[ask_call == 0] = np.inf
    ask_call = np.minimum.accumulate(ask_call, axis=0)
    ask_call[ask_call == np.inf] = 0
    
    # Put prices: bids should increase, asks should decrease with strike  
    bid_put = np.maximum.accumulate(bid_put, axis=0)
    
    # Handle zero asks similarly for puts
    ask_put[ask_put == 0] = np.inf
    ask_put = np.minimum.accumulate(ask_put[::-1], axis=0)[::-1]
    ask_put[ask_put == np.inf] = 0
    
    # Step 2: Apply no-arbitrage spread constraints
    # Forward pass (increasing strike) - tighten lower bounds
    for i in range(1, len(strike)):
        strike_diff = strike[i] - strike[i-1]
        
        # Call bid constraint: C_bid(K_i) >= C_bid(K_{i-1}) - ΔK/S
        bid_call[i] = max(bid_call[i], bid_call[i-1] - strike_diff / spot)
        
        # Put ask constraint: P_ask(K_i) <= P_ask(K_{i-1}) + ΔK/S
        if ask_put[i] > 0:  # Only apply if ask price is valid
            ask_put[i] = min(ask_put[i], ask_put[i-1] + strike_diff / spot)
        
    # Backward pass (decreasing strike) - tighten upper bounds
    for i in range(len(strike)-2, -1, -1):
        strike_diff = strike[i+1] - strike[i]
        
        # Call ask constraint: C_ask(K_i) <= C_ask(K_{i+1}) + ΔK/S
        if ask_call[i] > 0:  # Only apply if ask price is valid
            ask_call[i] = min(ask_call[i], ask_call[i+1] + strike_diff / spot)
        
        # Put bid constraint: P_bid(K_i) >= P_bid(K_{i+1}) - ΔK/S
        bid_put[i] = max(bid_put[i], bid_put[i+1] - strike_diff / spot)
        
    return bid_call, ask_call, bid_put, ask_put


def tighten_option_spreads_mixed_format(
    option_df: pl.DataFrame,
    spot_col: str = 'S',
    strike_col: str = 'strike',
    option_type_col: str = 'option_type',
    bid_col: str = 'bid_price',
    ask_col: str = 'ask_price'
) -> pl.DataFrame:
    """
    Apply option spread tightening for mixed call/put format data.
    
    This function handles DataFrames where calls and puts are in separate rows
    with an 'option_type' column indicating 'C' or 'P'.
    
    Args:
        option_df: DataFrame with mixed call/put rows
        spot_col: Column name for spot price
        strike_col: Column name for strike prices
        option_type_col: Column name for option type ('C'/'P')
        bid_col: Column name for bid prices
        ask_col: Column name for ask prices
        
    Returns:
        DataFrame with additional columns for tightened prices
        
    Example:
        >>> df = pl.DataFrame({
        ...     'strike': [90, 90, 95, 95, 100, 100],
        ...     'option_type': ['C', 'P', 'C', 'P', 'C', 'P'],
        ...     'bid_price': [12, 1, 8, 3, 5, 5],
        ...     'ask_price': [13, 2, 9, 4, 6, 6],
        ...     'S': [100, 100, 100, 100, 100, 100]
        ... })
        >>> tightened_df = tighten_option_spreads_mixed_format(df)
    """
    if option_df.is_empty():
        return option_df
    
    # Get spot price (should be constant across all options)
    spot_price = option_df[spot_col][0]
    
    # Separate calls and puts, sort by strike
    calls = option_df.filter(pl.col(option_type_col) == 'C').sort(strike_col)
    puts = option_df.filter(pl.col(option_type_col) == 'P').sort(strike_col)
    
    if calls.is_empty() or puts.is_empty():
        print("⚠️ Warning: Missing calls or puts data - cannot apply full constraints")
        return option_df.with_columns([
            pl.col(bid_col).alias('original_bid_price'),
            pl.col(ask_col).alias('original_ask_price'),
            pl.col(bid_col).alias('tightened_bid_price'),
            pl.col(ask_col).alias('tightened_ask_price')
        ])
    
    # Ensure we have matching strikes for calls and puts
    call_strikes = calls[strike_col].to_numpy()
    put_strikes = puts[strike_col].to_numpy()
    
    if not np.array_equal(call_strikes, put_strikes):
        print("⚠️ Warning: Call and put strikes don't match - using intersection")
        common_strikes = np.intersect1d(call_strikes, put_strikes)
        calls = calls.filter(pl.col(strike_col).is_in(common_strikes))
        puts = puts.filter(pl.col(strike_col).is_in(common_strikes))
    
    if calls.is_empty() or puts.is_empty():
        return option_df.with_columns([
            pl.col(bid_col).alias('original_bid_price'),
            pl.col(ask_col).alias('original_ask_price'),
            pl.col(bid_col).alias('tightened_bid_price'),
            pl.col(ask_col).alias('tightened_ask_price')
        ])
    
    # Apply constraints
    tightened_bid_call, tightened_ask_call, tightened_bid_put, tightened_ask_put = apply_option_constraints(
        bid_price_call=calls[bid_col].to_numpy(),
        ask_price_call=calls[ask_col].to_numpy(),
        bid_price_put=puts[bid_col].to_numpy(),
        ask_price_put=puts[ask_col].to_numpy(),
        strike=calls[strike_col].to_numpy(),
        spot=spot_price
    )
    
    # Add tightened prices to dataframes
    calls_tightened = calls.with_columns([
        pl.col(bid_col).alias('original_bid_price'),
        pl.col(ask_col).alias('original_ask_price'),
        pl.Series('tightened_bid_price', tightened_bid_call),
        pl.Series('tightened_ask_price', tightened_ask_call)
    ])
    
    puts_tightened = puts.with_columns([
        pl.col(bid_col).alias('original_bid_price'),
        pl.col(ask_col).alias('original_ask_price'),  
        pl.Series('tightened_bid_price', tightened_bid_put),
        pl.Series('tightened_ask_price', tightened_ask_put)
    ])
    
    # Combine back together and sort by original order
    result = pl.concat([calls_tightened, puts_tightened]).sort([strike_col, option_type_col])
    
    return result


def tighten_option_spreads_separate_columns(
    option_df: pl.DataFrame,
    spot_col: str = 'S',
    strike_col: str = 'strike',
    call_bid_col: str = 'bid_price',
    call_ask_col: str = 'ask_price', 
    put_bid_col: str = 'bid_price_P',
    put_ask_col: str = 'ask_price_P'
) -> pl.DataFrame:
    """
    Apply option spread tightening for separate call/put columns format.
    
    This function handles DataFrames where call and put prices are in separate columns
    on the same row (typical format from option chain data).
    
    Args:
        option_df: DataFrame with separate call/put columns
        spot_col: Column name for spot price
        strike_col: Column name for strike prices
        call_bid_col: Column name for call bid prices
        call_ask_col: Column name for call ask prices
        put_bid_col: Column name for put bid prices
        put_ask_col: Column name for put ask prices
        
    Returns:
        DataFrame with additional columns for original and tightened prices
        
    Example:
        >>> df = pl.DataFrame({
        ...     'strike': [90, 95, 100, 105, 110],
        ...     'bid_price': [12, 8, 5, 3, 1],      # Call bids
        ...     'ask_price': [13, 9, 6, 4, 2],      # Call asks
        ...     'bid_price_P': [1, 3, 5, 8, 12],    # Put bids
        ...     'ask_price_P': [2, 4, 6, 9, 13],    # Put asks
        ...     'S': [100, 100, 100, 100, 100]
        ... })
        >>> tightened_df = tighten_option_spreads_separate_columns(df)
    """
    # Remove rows with null values in critical columns
    df = option_df.drop_nulls([call_bid_col, call_ask_col, put_bid_col, put_ask_col, spot_col])
    
    if df.is_empty():
        return option_df
    
    # Sort by strike to ensure proper ordering
    df = df.sort(strike_col)
    
    # Apply constraints
    tightened_bid_call, tightened_ask_call, tightened_bid_put, tightened_ask_put = apply_option_constraints(
        bid_price_call=df[call_bid_col].to_numpy(),
        ask_price_call=df[call_ask_col].to_numpy(),
        bid_price_put=df[put_bid_col].to_numpy(),
        ask_price_put=df[put_ask_col].to_numpy(),
        strike=df[strike_col].to_numpy(),
        spot=df[spot_col].drop_nulls()[0]
    )
    
    # Add tightened prices while preserving original values
    result = df.with_columns([
        # Preserve original values
        pl.col(call_bid_col).alias(f'original_{call_bid_col}'),
        pl.col(call_ask_col).alias(f'original_{call_ask_col}'),
        pl.col(put_bid_col).alias(f'original_{put_bid_col}'),
        pl.col(put_ask_col).alias(f'original_{put_ask_col}'),
        
        # Add tightened values
        pl.Series(f'tightened_{call_bid_col}', tightened_bid_call),
        pl.Series(f'tightened_{call_ask_col}', tightened_ask_call),
        pl.Series(f'tightened_{put_bid_col}', tightened_bid_put),
        pl.Series(f'tightened_{put_ask_col}', tightened_ask_put),
        
        # Update original columns with tightened values
        pl.Series(call_bid_col, tightened_bid_call),
        pl.Series(call_ask_col, tightened_ask_call),
        pl.Series(put_bid_col, tightened_bid_put),
        pl.Series(put_ask_col, tightened_ask_put)
    ])
    # assert (result[call_ask_col] > result[call_bid_col]).all(), f"Call ask prices are not greater than bid prices, {result.filter(pl.col(call_ask_col) <= pl.col(call_bid_col))}"
    # assert (result[put_ask_col] > result[put_bid_col]).all(), f"Put ask prices are not greater than bid prices, {result.filter(pl.col(put_ask_col) <= pl.col(put_bid_col))}"

    call_breach = result[call_ask_col] <= result[call_bid_col]
    put_breach = result[put_ask_col] <= result[put_bid_col]

    if call_breach.any():
        print(f"❌ {df['timestamp'][0]}: Call prices violate bid < ask constraints, S: {df[spot_col][0]}, expiry: {df['expiry'][0]} strikes: {result.filter(call_breach)['strike'].to_list()}")
    if put_breach.any():
        print(f"❌ {df['timestamp'][0]}: Put prices violate bid < ask constraints, S: {df[spot_col][0]}, expiry: {df['expiry'][0]} strikes: {result.filter(put_breach)['strike'].to_list()}")
    return result.filter(~call_breach & ~put_breach)


# Backward compatibility aliases
def tighten_option_spread(option_df: pl.DataFrame) -> pl.DataFrame:
    """
    Legacy function name for backward compatibility.
    Automatically detects DataFrame format and applies appropriate tightening.
    """
    # Check if it's mixed format (has option_type column) or separate columns format
    if 'option_type' in option_df.columns:
        return tighten_option_spreads_mixed_format(option_df)
    else:
        # Assume separate columns format
        return tighten_option_spreads_separate_columns(option_df)


# Export main functions
__all__ = [
    'apply_option_constraints',
    'tighten_option_spreads_mixed_format',
    'tighten_option_spreads_separate_columns',
    'tighten_option_spread',  # backward compatibility
]