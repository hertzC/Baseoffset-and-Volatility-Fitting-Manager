"""
Orderbook Deribit Market Data Manager

Specialized manager for handling order book depth data from Deribit.
Automatically converts order book data to BBO format during initialization,
then behaves exactly like the standard DeribitMDManager.
"""

import numpy as np
import polars as pl

from config_loader import Config

from .orderbook_helper import get_volume_targeted_price
from .deribit_md_manager import DeribitMDManager


class OrderbookDeribitMDManager(DeribitMDManager):
    """
    Specialized DeribitMDManager for handling order book depth data.
    
    Automatically converts order book data to BBO format during initialization,
    then behaves exactly like the standard DeribitMDManager.
    
    Key features:
    - Converts flat orderbook structure (asks[i].price/bids[i].price) to BBO format
    - Configurable level selection (0=best, 1=second best, etc.)
    - Optional volume normalization: USD → BTC for futures/perpetuals using index_price
    - Symbol-aware processing: only normalizes futures and perpetuals, leaves options unchanged
    """
    CONFLATION_COLUMNS = ['bid_price', 'ask_price', 'bid_size', 'ask_size']

    def __init__(self, df_orderbook: pl.LazyFrame, date_str: str, config_loader: Config):
        """
        Initialize with order book depth data, converting to BBO format.
        
        Args:
            df_orderbook: LazyFrame containing order book depth data
            date_str: Date string in YYYYMMDD format
            level: Which orderbook level to use (0=best, 1=second best, etc.)
        """
        super().__init__(df_orderbook, date_str)
        self.num_of_level = config_loader.orderbook_level  
        self.future_min_tick = config_loader.future_min_tick_size
        self.price_widening_factor = config_loader.price_widening_factor
        self.target_coin_volume = config_loader.target_coin_volume

    @staticmethod
    def convert_orderbook_to_bbo(df_orderbook: pl.DataFrame, level: int = 0) -> pl.DataFrame:
        """
        Convert order book depth data to BBO format by extracting bid/ask prices from specified level.
        
        Expected orderbook schema (flat structure with multiple levels): 
        - symbol, timestamp, exchange, local_timestamp, index_price
        - asks[0].price, asks[0].amount, bids[0].price, bids[0].amount
        - asks[1].price, asks[1].amount, bids[1].price, bids[1].amount
        - ... (up to level 4)
        
        Output BBO schema:
        - timestamp, exchange_timestamp, symbol, bid_price, ask_price
        
        Args:
            df_orderbook: DataFrame with order book depth data (flat structure)
            level: Which level to extract (0=best, 1=second best, etc.)
            
        Returns:
            DataFrame with BBO format data
        """
        print(f"🔄 Converting order book depth data to BBO format (using level {level})...")
        
        # Validate level parameter
        if level < 0 or level > 4:
            raise ValueError(f"Level must be between 0 and 4, got {level}")
        
        # Check if the required columns exist
        ask_price_col = f'asks[{level}].price'
        ask_amount_col = f'asks[{level}].amount'
        bid_price_col = f'bids[{level}].price'
        bid_amount_col = f'bids[{level}].amount'
        
        required_cols = [ask_price_col, ask_amount_col, bid_price_col, bid_amount_col]
        missing_cols = [col for col in required_cols if col not in df_orderbook.columns]
        
        if missing_cols:
            available_levels = []
            for i in range(5):  # Check levels 0-4
                if f'asks[{i}].price' in df_orderbook.columns and f'bids[{i}].price' in df_orderbook.columns:
                    available_levels.append(i)
            raise ValueError(f"Missing columns for level {level}: {missing_cols}. Available levels: {available_levels}")
        
        # Convert to BBO format
        bbo_data = df_orderbook.select([
            'timestamp',
            'symbol',
            pl.col(bid_price_col).alias('bid_price'),
            pl.col(ask_price_col).alias('ask_price'),
            # Keep the amounts and index_price for volume normalization
            pl.col(bid_amount_col).alias('bid_size'),
            pl.col(ask_amount_col).alias('ask_size')
        ])
        # Note: Removed filtering for null/zero prices to preserve empty orderbook entries
        
        print(f"✅ Converted {len(df_orderbook)} orderbook rows to {len(bbo_data)} BBO rows")
        
        return bbo_data
    
    def normalize_volume_to_btc(self, df_bbo: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize volume from USD value to BTC value for futures and perpetuals.
        
        Uses the index_price column from the original orderbook data to convert USD amounts
        to BTC amounts. Only applies to futures and perpetual contracts, not options.
        
        Args:
            df_bbo: DataFrame with BBO data including bid_size, ask_size columns
            
        Returns:
            DataFrame with normalized volumes (USD → BTC for futures/perps)
        """
        if 'bid_size' not in df_bbo.columns or 'ask_size' not in df_bbo.columns:
            print("⚠️  No size columns found, skipping volume normalization")
            return df_bbo
            
        if 'index_price' not in df_bbo.columns:
            print("⚠️  No index_price column found, skipping volume normalization")
            return df_bbo
        
        print("🔄 Normalizing volume from USD to BTC for futures and perpetuals...")
        
        # Get symbol classification from parent class symbol lookup
        df_with_type = df_bbo.join(
            self.df_symbol.select(['symbol', 'is_future', 'is_perp']),
            on='symbol',
            how='left'
        )
        
        # Normalize volumes only for futures and perpetuals
        df_normalized = df_with_type.with_columns([
            pl.when(
                (pl.col('is_future') | pl.col('is_perp')) & 
                (pl.col('index_price').is_not_null()) & 
                (pl.col('index_price') > 0)
            )
            .then(((pl.col('bid_size') / pl.col('index_price')) / self.future_min_tick).round() * self.future_min_tick)  # Round to nearest tick
            .otherwise(pl.col('bid_size'))
            .alias('bid_size'),
            
            pl.when(
                (pl.col('is_future') | pl.col('is_perp')) & 
                (pl.col('index_price').is_not_null()) & 
                (pl.col('index_price') > 0)
            )
            .then(((pl.col('ask_size') / pl.col('index_price')) / self.future_min_tick).round() * self.future_min_tick)  # Round to nearest tick
            .otherwise(pl.col('ask_size'))
            .alias('ask_size')
        ]).drop(['is_future', 'is_perp'])  # Remove temporary classification columns
        
        # Count how many symbols were normalized
        normalized_count = df_with_type.filter(
            (pl.col('is_future') | pl.col('is_perp')) & 
            (pl.col('index_price').is_not_null()) & 
            (pl.col('index_price') > 0)
        ).select('symbol').n_unique()
        
        total_futures_perps = df_with_type.filter(
            pl.col('is_future') | pl.col('is_perp')
        ).select('symbol').n_unique()
        
        print(f"✅ Normalized volumes for {normalized_count}/{total_futures_perps} futures/perpetual symbols")
        
        return df_normalized  

    def get_conflated_md(self, freq, period) -> pl.DataFrame:
        """
        Override to add orderbook-to-BBO conversion and volume normalization.
        Reuses parent's conflation logic but with extended column set.
        """
        # Convert orderbook to BBO format

        # for option
        temp_option =\
        self.lazy_df.filter(
            pl.col("symbol").str.ends_with('-C') | pl.col("symbol").str.ends_with('-P')
        ).collect().pipe(lambda df: self.convert_orderbook_to_bbo(df, self.num_of_level))

        temp_delta_one = self.lazy_df.filter(
            ~pl.col("symbol").str.ends_with("-C") & ~pl.col("symbol").str.ends_with("-P")
        ).collect()
        bid_vwap, ask_vwap, bid_size, ask_size = get_volume_targeted_price(temp_delta_one, target_btc=self.target_coin_volume, price_widening_factor=self.price_widening_factor)
        temp_delta_one =\
        temp_delta_one.with_columns(
            bid_price = np.array(bid_vwap, dtype=np.float64),
            ask_price = np.array(ask_vwap, dtype=np.float64),
            bid_size = np.array(bid_size, dtype=np.float64),
            ask_size = np.array(ask_size, dtype=np.float64)
        ).select(temp_option.columns)

        # lazy_df = self.lazy_df.collect().pipe(lambda df: self.convert_orderbook_to_bbo(df, self.num_of_level)).lazy()
        lazy_df = pl.concat([temp_option, temp_delta_one]).lazy()
        
        # Parse timestamps (reuse parent's method)
        parsed_lazy_df = self.with_parsed_timestamps(lazy_df)
        
        # Use parent's conflation logic but with our extended columns
        df = super()._get_conflated_md(parsed_lazy_df, freq, period)

        # join with index_price on exchange_timestamp
        df_index_price = self.with_parsed_timestamps(self.lazy_df.select(['timestamp','index_price']).unique()).collect()
        df = df.join(
            df_index_price,
            on='timestamp',
            how='left'
        )

        # Join symbol information and enrich (inline the removed join_symbol_info method)
        df = self.normalize_volume_to_btc(df).join(self.df_symbol, on='symbol', how='left')
        return self.enrich_conflated_md(df)
    
    def with_parsed_timestamps(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Parse timestamps for orderbook data format.
        Override parent method to handle orderbook-specific timestamp format.
        """
        return lazy_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime(time_unit="us"))   
        )

    def enrich_conflated_md(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = ['timestamp','symbol','expiry'] + self.CONFLATION_COLUMNS
        return (
            df.filter(pl.col("is_option")).join(
                df.filter(pl.col("is_future"))[cols],
                on=['timestamp','expiry'],
                how='left',
                suffix='_fut'
            ).with_columns(
                tau = (pl.col('expiry_ts') - pl.col('timestamp')).dt.total_seconds() / (365*24*3600)
            ).rename({'index_price': 'S'}) 
        )