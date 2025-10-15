"""
Volatility Calculation Module

This module provides functions to calculate implied volatilities from option prices
and perform synthetic arbitrage checks.
"""

import polars as pl
import numpy as np
from typing import Tuple
from utils.pricer.pricer_helper import find_vol


def calculate_bid_ask_volatilities(
    df: pl.DataFrame, 
    interest_rate: float, 
    is_call: bool, 
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate implied volatilities for bid and ask prices.
    
    Args:
        df: DataFrame with option market data containing price columns
        interest_rate: Risk-free interest rate
        is_call: True for calls, False for puts
        **kwargs: Additional arguments passed to find_vol function
        
    Returns:
        Tuple of (bid_volatilities, ask_volatilities) as numpy arrays
    """
    # Select appropriate price columns based on option type
    bid_col = 'bp0_C_usd' if is_call else 'bp0_P_usd'
    ask_col = 'ap0_C_usd' if is_call else 'ap0_P_usd'
    
    # Extract required data for volatility calculation
    input_data = df[bid_col, ask_col, 'F', 'strike', 'tau']
    
    # Calculate bid volatilities
    bid_vola = find_vol(
        target_value=input_data[:, 0], 
        F=input_data[:, 2], 
        K=input_data[:, 3], 
        T=input_data[:, 4], 
        r=interest_rate, 
        option_type='C' if is_call else 'P',
        kwargs=kwargs
    )
    
    # Calculate ask volatilities
    ask_vola = find_vol(
        target_value=input_data[:, 1], 
        F=input_data[:, 2], 
        K=input_data[:, 3], 
        T=input_data[:, 4], 
        r=interest_rate, 
        option_type='C' if is_call else 'P',
        kwargs=kwargs
    )

    return bid_vola, ask_vola


def process_option_chain_with_volatilities(
    df_option_chain: pl.DataFrame, 
    interest_rate: float, 
    **kwargs
) -> pl.DataFrame:
    """
    Process option chain by calculating implied volatilities and performing arbitrage checks.
    
    This function:
    1. Converts tightened bid/ask prices to USD terms
    2. Calculates implied volatilities for calls and puts
    3. Performs synthetic arbitrage validation
    
    Args:
        df_option_chain: DataFrame with tightened option chain data
        interest_rate: Risk-free interest rate
        **kwargs: Additional arguments passed to volatility calculation
        
    Returns:
        DataFrame with added volatility columns and proper naming conventions
        
    Raises:
        ValueError: If synthetic arbitrage violations are detected
    """
    # Convert tightened prices to USD terms
    df = df_option_chain.with_columns([
        (pl.col('tightened_bid_price') * pl.col('S')).round(2).alias('bp0_C_usd'),
        (pl.col('tightened_ask_price') * pl.col('S')).round(2).alias('ap0_C_usd'),
        (pl.col('tightened_bid_price_P') * pl.col('S')).round(2).alias('bp0_P_usd'),
        (pl.col('tightened_ask_price_P') * pl.col('S')).round(2).alias('ap0_P_usd'),
    ]).select([
        'timestamp', 'bid_size', 'tightened_bid_price', 'bp0_C_usd', 'ap0_C_usd', 
        'tightened_ask_price', 'ask_size', 'strike', 'bid_size_P',
        'tightened_bid_price_P', 'bp0_P_usd', 'ap0_P_usd', 'tightened_ask_price_P', 
        'ask_size_P', 'S', 'bid_price_fut', 'expiry', 'tau'
    ]).rename({
        'bid_size': 'bq0_C', 
        'tightened_bid_price': 'bp0_C', 
        'tightened_ask_price': 'ap0_C', 
        'ask_size': 'aq0_C',
        'bid_size_P': 'bq0_P', 
        'tightened_bid_price_P': 'bp0_P', 
        'tightened_ask_price_P': 'ap0_P', 
        'ask_size_P': 'aq0_P', 
        'bid_price_fut': 'F'
    })
    
    # Calculate implied volatilities
    call_bid_vola, call_ask_vola = calculate_bid_ask_volatilities(
        df, interest_rate, is_call=True, **kwargs
    )
    put_bid_vola, put_ask_vola = calculate_bid_ask_volatilities(
        df, interest_rate, is_call=False, **kwargs
    )
    
    # Add volatility columns to DataFrame (convert to percentage)
    df = df.with_columns([
        pl.lit(interest_rate).alias('r'),
        (pl.Series('bidVola_C', call_bid_vola) * 100).alias('bidVola_C'),
        (pl.Series('askVola_C', call_ask_vola) * 100).alias('askVola_C'),
        (pl.Series('bidVola_P', put_bid_vola) * 100).alias('bidVola_P'),
        (pl.Series('askVola_P', put_ask_vola) * 100).alias('askVola_P'),
    ])
    
    # Perform synthetic arbitrage checks
    _validate_synthetic_arbitrage(df)
    
    return df


def _validate_synthetic_arbitrage(df: pl.DataFrame) -> None:
    """
    Validate synthetic arbitrage conditions.
    
    Synthetic arbitrage conditions:
    - Ask vol of Put >= Bid vol of Call  
    - Ask vol of Call >= Bid vol of Put
    
    Args:
        df: DataFrame with volatility columns
        
    Raises:
        ValueError: If synthetic arbitrage violations are detected
    """
    # Check for cross-violations: ask vol of put < bid vol of call
    mask_cross_cp = (df['askVola_P'] < df['bidVola_C'])
    if mask_cross_cp.any():
        violating_strikes = df.filter(mask_cross_cp)[['strike', 'bidVola_C', 'askVola_P']]
        raise ValueError(
            f"Synthetic arbitrage violation: Ask vol of Put < Bid vol of Call at strikes:\n"
            f"{violating_strikes}"
        )

    # Check for cross-violations: ask vol of call < bid vol of put  
    mask_cross_pc = (df['askVola_C'] < df['bidVola_P'])
    if mask_cross_pc.any():
        violating_strikes = df.filter(mask_cross_pc)[['strike', 'bidVola_P', 'askVola_C']]
        raise ValueError(
            f"Synthetic arbitrage violation: Ask vol of Call < Bid vol of Put at strikes:\n"
            f"{violating_strikes}"
        )


def add_volatility_summary_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add summary volatility columns for easier analysis.
    
    Args:
        df: DataFrame with bid/ask volatility columns
        
    Returns:
        DataFrame with additional columns: bidVola, askVola, midVola, volSpread
    """
    return df.with_columns([
        # Take maximum of call and put bid volatilities
        pl.max_horizontal('bidVola_C', 'bidVola_P').round(2).alias('bidVola'),
        # Take minimum of call and put ask volatilities  
        pl.min_horizontal('askVola_C', 'askVola_P').round(2).alias('askVola'),
    ]).with_columns([
        # Calculate mid volatility
        ((pl.col('bidVola') + pl.col('askVola')) / 2).round(2).alias('midVola'),
        # Calculate volatility spread
        (pl.col('askVola') - pl.col('bidVola')).round(2).alias('volSpread'),
    ])


def get_volatility_statistics(df: pl.DataFrame) -> dict:
    """
    Get summary statistics for volatility data.
    
    Args:
        df: DataFrame with volatility columns
        
    Returns:
        Dictionary with volatility statistics
    """
    stats = {}
    
    # Check if required columns exist
    required_cols = ['bidVola_C', 'askVola_C', 'bidVola_P', 'askVola_P']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate statistics
    stats['call_bid_vol_range'] = (df['bidVola_C'].min(), df['bidVola_C'].max())
    stats['call_ask_vol_range'] = (df['askVola_C'].min(), df['askVola_C'].max())
    stats['put_bid_vol_range'] = (df['bidVola_P'].min(), df['bidVola_P'].max())
    stats['put_ask_vol_range'] = (df['askVola_P'].min(), df['askVola_P'].max())
    
    # If summary columns exist, add their statistics
    if 'midVola' in df.columns:
        stats['mid_vol_range'] = (df['midVola'].min(), df['midVola'].max())
        stats['avg_mid_vol'] = df['midVola'].mean()
    
    if 'volSpread' in df.columns:
        stats['vol_spread_range'] = (df['volSpread'].min(), df['volSpread'].max())
        stats['avg_vol_spread'] = df['volSpread'].mean()
    
    stats['num_strikes'] = len(df)
    
    return stats