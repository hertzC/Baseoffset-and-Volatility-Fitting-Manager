import polars as pl


def get_volume_targeted_price(df: pl.DataFrame, target_btc: int, price_widening_factor: float) -> tuple[list, list, list, list]:
    """ given the target number of coins, how much depth can we reach on bid and ask side """
    bid_vwap, ask_vwap = [], []
    bid_size, ask_size = [], []
    bid_price_cols = [f"bids[{level}].price" for level in range(5)]
    bid_amount_cols = [f"bids[{level}].amount" for level in range(5)]
    ask_price_cols = [f"asks[{level}].price" for level in range(5)]
    ask_amount_cols = [f"asks[{level}].amount" for level in range(5)]
    try:
        for row in df.iter_rows(named=True):
            if row['index_price'] is None or row['bids[0].price'] is None or row['asks[0].price'] is None:
                bid_vwap.append(None)
                ask_vwap.append(None)
                bid_size.append(None)
                ask_size.append(None)
            else:
                target_volume = target_btc * row['index_price']
                cumulative_bid_volume = 0
                bid_invalid_level = None
                for i in range(5):
                    if row[bid_amount_cols[i]] is None:
                        bid_invalid_level = i
                        break
                    cumulative_bid_volume += row[bid_amount_cols[i]]
                    if cumulative_bid_volume >= target_volume:
                        break
                
                if cumulative_bid_volume < target_volume:
                    if bid_invalid_level:
                        bid_vwap.append(row[bid_price_cols[bid_invalid_level-1]] * (1 - price_widening_factor))
                    else:
                        bid_vwap.append(row[bid_price_cols[i]] * (1 - price_widening_factor))  # row[i] is None
                else:
                    bid_vwap.append(row[bid_price_cols[i]])
                bid_size.append(min(target_volume, cumulative_bid_volume))

                cumulative_ask_volume = 0
                ask_invalid_level = None
                for i in range(5):
                    if row[ask_amount_cols[i]] is None:
                        ask_invalid_level = i
                        break
                    cumulative_ask_volume += row[ask_amount_cols[i]]
                    if cumulative_ask_volume >= target_volume:
                        break

                if cumulative_ask_volume < target_volume:
                    if ask_invalid_level:
                        ask_vwap.append(row[ask_price_cols[ask_invalid_level-1]] * (1 + price_widening_factor))
                    else:
                        ask_vwap.append(row[ask_price_cols[i]] * (1 + price_widening_factor))
                else:
                    ask_vwap.append(row[ask_price_cols[i]])
                ask_size.append(min(target_volume, cumulative_ask_volume))
    except Exception as e:
        print(f"⚠️  Error calculating volume-targeted prices: {e}, row={i}")
        return [], [], [], []
    return bid_vwap, ask_vwap, bid_size, ask_size