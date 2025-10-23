"""
HTML Table Generator for Bitcoin Options Price Comparison

This module provides functionality to generate styled HTML tables for 
displaying Bitcoin options price comparisons with highlighting for changes.
"""

import polars as pl


def format_price_change(old_val, new_val, precision=4):
    """
    Format price with red highlighting if changed
    
    Args:
        old_val (float): Original price value
        new_val (float): New price value  
        precision (int): Number of decimal places to display
        
    Returns:
        str: Formatted HTML string with conditional highlighting
    """
    if abs(old_val - new_val) > 1e-6:  # If changed (accounting for floating point precision)
        return f'<span style="color: red; font-weight: bold;">{new_val:.{precision}f}</span>'
    else:
        return f'{new_val:.{precision}f}'


def enrich_comparison_df(comparison_df):
    """
    Enrich comparison DataFrame with spread calculations
    
    Args:
        comparison_df: Polars DataFrame with price comparison data
        
    Returns:
        pl.DataFrame: Enriched DataFrame with spread columns
    """
    return comparison_df.rename(
        {'original_bid_price': 'old_bid_price', 'original_ask_price': 'old_ask_price', 
         'original_bid_price_P': 'old_bid_price_P', 'original_ask_price_P': 'old_ask_price_P'}
    ).with_columns([
        (pl.col('ask_price') - pl.col('bid_price')).alias('final_call_spread'),
        (pl.col('old_ask_price') - pl.col('old_bid_price')).alias('orig_call_spread'),
        (pl.col('ask_price_P') - pl.col('bid_price_P')).alias('final_put_spread'),
        (pl.col('old_ask_price_P') - pl.col('old_bid_price_P')).alias('orig_put_spread'),
    ]).sort('strike')


def generate_price_comparison_table(comparison_df, table_width="70%", font_size="10px", volume_threshold=None):
    """
    Generate HTML table for price comparison with styling
    
    Args:
        comparison_df: Polars DataFrame with price comparison data
        table_width (str): CSS width for the table
        font_size (str): CSS font size for the table
        
    Returns:
        str: Complete HTML content with CSS styling and table
    """
    print("📝 Generating HTML price comparison table for the tightened bid/ask spread on call and put options..." + (f" (Volume Threshold: {volume_threshold})" if volume_threshold else ""))
    # Get spot price for display
    comparison_df = enrich_comparison_df(comparison_df)
    spot_price = comparison_df['S'][0]
    
    # CSS styling for the table
    html_content = f"""
    <div style="color: white; font-family: monospace; margin-bottom: 10px; background-color: #333; padding: 5px; width: {table_width};">
        <strong>📊 BTC Spot Price: ${spot_price:.2f}</strong>
    </div>
    <style>
    .price-table {{
        border-collapse: collapse;
        width: {table_width};
        font-family: monospace;
        font-size: {font_size};
        background-color: black;
        color: white;
    }}
    .price-table th, .price-table td {{
        border: 1px solid #444;
        padding: 2px;
        text-align: center;
        background-color: black;
    }}
    .price-table th {{
        background-color: #222;
        font-weight: bold;
        color: white;
    }}
    .price-table tr:nth-child(even) td {{
        background-color: #111;
    }}
    .price-table tr:nth-child(odd) td {{
        background-color: black;
    }}
    </style>
    <table class="price-table">
    <thead>
    <tr>
        <th>Strike</th>
        <th>Bid Size</th>
        <th colspan="2">Call Bid</th>
        <th colspan="2">Call Ask</th>
        <th>Ask Size</th>
        <th colspan="2">Call Spread</th>
        <th>Bid Size</th>
        <th colspan="2">Put Bid</th>
        <th colspan="2">Put Ask</th>
        <th>Ask Size</th>
        <th colspan="2">Put Spread</th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th>Old</th><th>New</th>
        <th>Old</th><th>New</th>
        <th></th>
        <th>Old</th><th>New</th>
        <th></th>
        <th>Old</th><th>New</th>
        <th>Old</th><th>New</th>
        <th></th>
        <th>Old</th><th>New</th>
    </tr>
    </thead>
    <tbody>
    """
    
    # Add rows for each strike
    for row in comparison_df.iter_rows(named=True):
        html_content += f"""
        <tr>
            <td>{row['strike']}</td>
            <td>{row['bid_size']:.1f}</td>
            <td>{row['old_bid_price']:.4f}</td>
            <td>{format_price_change(row['old_bid_price'], row['bid_price'])}</td>
            <td>{row['old_ask_price']:.4f}</td>
            <td>{format_price_change(row['old_ask_price'], row['ask_price'])}</td>
            <td>{row['ask_size']:.1f}</td>
            <td>{row['orig_call_spread']:.4f}</td>
            <td>{format_price_change(row['orig_call_spread'], row['final_call_spread'])}</td>
            <td>{row['bid_size_P']:.1f}</td>
            <td>{row['old_bid_price_P']:.4f}</td>
            <td>{format_price_change(row['old_bid_price_P'], row['bid_price_P'])}</td>
            <td>{row['old_ask_price_P']:.4f}</td>
            <td>{format_price_change(row['old_ask_price_P'], row['ask_price_P'])}</td>
            <td>{row['ask_size_P']:.1f}</td>
            <td>{row['orig_put_spread']:.4f}</td>
            <td>{format_price_change(row['orig_put_spread'], row['final_put_spread'])}</td>
        </tr>
        """
    
    html_content += """
    </tbody>
    </table>
    """
    
    return html_content


def calculate_tightening_stats(comparison_df):
    """
    Calculate summary statistics for price tightening effectiveness
    
    Args:
        comparison_df: Polars DataFrame with price comparison data
        
    Returns:
        dict: Dictionary containing various statistics about price changes
    """
    comparison_df = enrich_comparison_df(comparison_df)
    
    stats = {
        'call_bid_changes': len(comparison_df.filter(pl.col('old_bid_price') != pl.col('bid_price'))),
        'call_ask_changes': len(comparison_df.filter(pl.col('old_ask_price') != pl.col('ask_price'))),
        'put_bid_changes': len(comparison_df.filter(pl.col('old_bid_price_P') != pl.col('bid_price_P'))),
        'put_ask_changes': len(comparison_df.filter(pl.col('old_ask_price_P') != pl.col('ask_price_P'))),
        'spread_improved': len(comparison_df.filter(pl.col('final_call_spread') < pl.col('orig_call_spread'))),
        'put_spread_improved': len(comparison_df.filter(pl.col('final_put_spread') < pl.col('orig_put_spread'))),
        'total_strikes': len(comparison_df),
        'avg_call_spread_change': (comparison_df['final_call_spread'] - comparison_df['orig_call_spread']).mean(),
        'avg_put_spread_change': (comparison_df['final_put_spread'] - comparison_df['orig_put_spread']).mean(),
    }
    
    return stats


def print_tightening_effectiveness(stats):
    """
    Print formatted summary of tightening effectiveness
    
    Args:
        stats (dict): Statistics dictionary from calculate_tightening_stats()
    """
    print(f"\n📈 TIGHTENING EFFECTIVENESS:")
    print(f"   Call bid changes: {stats['call_bid_changes']}/{stats['total_strikes']} strikes")
    print(f"   Call ask changes: {stats['call_ask_changes']}/{stats['total_strikes']} strikes")  
    print(f"   Put bid changes: {stats['put_bid_changes']}/{stats['total_strikes']} strikes")
    print(f"   Put ask changes: {stats['put_ask_changes']}/{stats['total_strikes']} strikes")
    print(f"   Call spreads improved: {stats['spread_improved']}/{stats['total_strikes']} ({100*stats['spread_improved']/stats['total_strikes']:.1f}%)")
    print(f"   Put spreads improved: {stats['put_spread_improved']}/{stats['total_strikes']} ({100*stats['put_spread_improved']/stats['total_strikes']:.1f}%)")
    print(f"   Avg call spread change: {stats['avg_call_spread_change']:.6f}")
    print(f"   Avg put spread change: {stats['avg_put_spread_change']:.6f}")