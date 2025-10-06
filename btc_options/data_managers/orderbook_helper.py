import polars as pl


def _calculate_vwap(row: dict, target_volume: float, price_widening_factor: float, side: str) -> tuple[float, float]:
    """
    Generic VWAP calculation for both buy and sell operations.
    
    Args:
        row: Single row from orderbook DataFrame
        target_volume: Target USD volume to trade
        price_widening_factor: Price adjustment when insufficient liquidity
        side: 'sell' for hitting bids, 'buy' for hitting asks
        
    Returns:
        tuple: (vwap_price, achieved_volume)
    """
    if side == 'sell':
        # Selling BTC (hitting bids)
        price_cols = [f"bids[{level}].price" for level in range(5)]
        amount_cols = [f"bids[{level}].amount" for level in range(5)]
        widening_multiplier = 1 - price_widening_factor  # Downward price adjustment
    elif side == 'buy':
        # Buying BTC (hitting asks)
        price_cols = [f"asks[{level}].price" for level in range(5)]
        amount_cols = [f"asks[{level}].amount" for level in range(5)]
        widening_multiplier = 1 + price_widening_factor  # Upward price adjustment
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
    
    cumulative_volume = 0
    last_valid_level = None
    
    # Accumulate volume across orderbook levels
    for level in range(5):
        if row[amount_cols[level]] is None:
            break
        cumulative_volume += row[amount_cols[level]]
        last_valid_level = level
        if cumulative_volume >= target_volume:
            break
    
    # Determine VWAP based on available liquidity
    if cumulative_volume < target_volume:
        # Insufficient liquidity - apply price widening
        if last_valid_level is not None:
            vwap = row[price_cols[last_valid_level]] * widening_multiplier
        else:
            vwap = row[price_cols[0]] * widening_multiplier  # Fallback to best level
    else:
        # Sufficient liquidity - use current level price
        vwap = row[price_cols[level]]
    
    achieved_volume = min(target_volume, cumulative_volume)
    return vwap, achieved_volume


def _calculate_sell_vwap(row: dict, target_volume: float, price_widening_factor: float) -> tuple[float, float]:
    """Calculate VWAP for selling BTC (hitting bids)."""
    return _calculate_vwap(row, target_volume, price_widening_factor, 'sell')


def _calculate_buy_vwap(row: dict, target_volume: float, price_widening_factor: float) -> tuple[float, float]:
    """Calculate VWAP for buying BTC (hitting asks)."""
    return _calculate_vwap(row, target_volume, price_widening_factor, 'buy')


def get_volume_targeted_price(df: pl.DataFrame, target_btc: int, price_widening_factor: float) -> tuple[list, list, list, list]:
    """
    Calculate volume-weighted average prices for target BTC volume on both sides.
    
    Args:
        df: DataFrame with orderbook data (bids/asks levels 0-4)
        target_btc: Target volume in BTC
        price_widening_factor: Price adjustment when insufficient liquidity
        
    Returns:
        tuple: (bid_vwap, ask_vwap, bid_size, ask_size) lists
    """
    bid_vwap, ask_vwap = [], []
    bid_size, ask_size = [], []
    
    try:
        for row in df.iter_rows(named=True):
            # Handle invalid/missing data
            if (row['index_price'] is None or 
                row['bids[0].price'] is None or 
                row['asks[0].price'] is None):
                bid_vwap.append(None)
                ask_vwap.append(None)
                bid_size.append(None)
                ask_size.append(None)
                continue
            
            # Convert target BTC to USD volume
            target_volume = target_btc * row['index_price']
            
            # Calculate sell side (hitting bids)
            sell_vwap, sell_volume = _calculate_sell_vwap(row, target_volume, price_widening_factor)
            bid_vwap.append(sell_vwap)
            bid_size.append(sell_volume)
            
            # Calculate buy side (hitting asks)
            buy_vwap, buy_volume = _calculate_buy_vwap(row, target_volume, price_widening_factor)
            ask_vwap.append(buy_vwap)
            ask_size.append(buy_volume)
            
    except Exception as e:
        print(f"⚠️ Error calculating volume-targeted prices: {e}")
        return [], [], [], []
    
    return bid_vwap, ask_vwap, bid_size, ask_size