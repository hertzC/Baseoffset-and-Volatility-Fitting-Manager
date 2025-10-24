"""
Deribit Market Data Manager

Handles processing of Deribit Bitcoin options market data including:
- Symbol parsing and categorization
- Data conflation and enrichment
- Option chain construction
- Spread tigh            ]).with_columns([
                pl.col("bid").round(2),
                pl.col("ask").round(2), 
                pl.col("mid").round(2),
                pl.col("spread").round(2)
            ])
        )
        
        # Build column list dynamically based on available columns
        base_columns = ['strike','bid','ask','mid','spread', 'S', 'tau', 'bid_price_fut', 'ask_price_fut']
        
        # Add futures size columns if available
        if 'bid_size_fut' in df.columns and 'ask_size_fut' in df.columns:
            base_columns.extend(['bid_size_fut', 'ask_size_fut'])
            
        return result.select(base_columns).sort('strike')ith monotonicity and no-arbitrage constraints
"""

from datetime import datetime
import polars as pl

# Import shared option constraints module
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.base_offset_config import BaseOffsetConfig
from config.volatility_config import VolatilityConfig
from utils.pricer.option_constraints import tighten_option_spreads_separate_columns


class DeribitMDManager:
    """Manager for processing Deribit market data and constructing option chains."""
    CONFLATION_COLUMNS = ['bid_price', 'ask_price']

    def __init__(self, df: pl.LazyFrame, date_str: str, config_loader: VolatilityConfig|BaseOffsetConfig):
        """
        Initialize the market data manager.
        
        Args:
            df: LazyFrame containing raw Deribit market data
            date_str: Date string in YYYYMMDD format
        """
        self.lazy_df = df
        self.date_str = date_str
        self.df_symbol = self.get_symbol_lookup()
        self.fut_expiries = self.find_expiries(is_future=True)
        self.opt_expiries = self.find_expiries(is_future=False)
        self.config_loader = config_loader

    def get_symbol_lookup(self) -> pl.DataFrame:
        """Create symbol lookup table with categorization flags."""
        return self.lazy_df.select(pl.col("symbol")).unique().collect().with_columns(
            is_option=pl.col('symbol').str.ends_with('-C') | pl.col('symbol').str.ends_with('-P'),
            is_index=pl.col('symbol').str.starts_with('INDEX'),
            is_perp=pl.col('symbol').str.ends_with('-PERPETUAL'),
            is_call=pl.col('symbol').str.contains('-C'),
            is_put=pl.col('symbol').str.contains('-P'),
        ).with_columns(
            is_future=~(pl.col('is_option') | pl.col('is_index') | pl.col('is_perp'))
        ).with_columns(
            pl.when(pl.col('is_future') | pl.col("is_option"))
            .then(pl.col('symbol').str.extract(r'-([0-9]{1,2}[A-Z]{3}[0-9]{2})', 1))
            .otherwise(None)
            .alias("expiry"),
            pl.when(pl.col('is_option'))
            .then(pl.col('symbol').str.extract(r'-([0-9]+)-[CP]$', 1).cast(pl.Int64))
            .otherwise(None)
            .alias("strike"),
        ).with_columns(
            expiry_ts=(pl.col('expiry') + pl.lit(" 08:00:00")).str.strptime(pl.Datetime, "%d%B%y %H:%M:%S")
        )

    def find_expiries(self, is_future: bool) -> list[str]:
        """Find all available expiries for futures or options."""
        df = self.df_symbol.filter(pl.col("is_option"))
        if is_future:
            df = self.df_symbol.filter(pl.col("is_future") == is_future)
        return df.select(['expiry', 'expiry_ts']).unique().sort('expiry_ts')['expiry'].to_list()

    def get_sorted_expiries(self) -> list[str]:
        """Get sorted list of option expiries."""
        sorted_expiries = sorted(self.opt_expiries, 
                                  key=lambda x: self.df_symbol.filter(pl.col('expiry') == x)['expiry_ts'][0])
        return sorted_expiries

    def with_parsed_timestamps(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        """Return DataFrame with timestamps parsed as datetime objects.  (Use exchange timestamp instead of local timestamp)"""
        return lazy_df.with_columns(
            timestamp=(pl.lit(self.date_str) + " " + pl.col("exchange_timestamp")).str.strptime(pl.Datetime, "%Y%m%d %H:%M:%S%.f"),
        )

    def get_conflation_columns(self) -> list[str]:
        """Get list of columns used for conflation."""
        return self.CONFLATION_COLUMNS

    def get_conflated_md(self, freq: str, period: str) -> pl.DataFrame:
        """
        Conflate market data to regular time intervals and enrich with symbol info and derived columns.

        Args:
            freq: Frequency for conflation (e.g., "1m")
            period: Lookback period (e.g., "10m")

        Returns:
            Conflated and enriched market data DataFrame
        """
        parsed_lazy_df = self.with_parsed_timestamps(self.lazy_df)
        df = self._get_conflated_md(parsed_lazy_df, freq, period)
        
        # Join symbol information and enrich
        df = df.join(self.df_symbol, on='symbol', how='left')
        return self.enrich_conflated_md(df)

    def _get_conflated_md(self, parsed_lazy_df: pl.LazyFrame, freq: str, period: str) -> pl.DataFrame:
        """Internal method for conflation that can be overridden by subclasses."""
        return parsed_lazy_df.sort('timestamp').group_by_dynamic(
            every=freq,
            period=period,
            index_column="timestamp",
            group_by="symbol",
            label="right",
        ).agg(
            [pl.col(col).last().alias(col) for col in self.get_conflation_columns()]
        ).collect().sort(['timestamp', 'symbol'])

    def enrich_conflated_md(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Enrich conflated market data with spot prices, futures data, and time to expiry.
        
        Args:
            df: Conflated market data DataFrame
            
        Returns:
            Enriched DataFrame with additional columns for analysis
        """
        return (
            df.filter(pl.col("is_option")).join(
                df.filter(pl.col("is_index"))['timestamp','bid_price'].rename({'bid_price':'S'}),
                on='timestamp',
                how='left',
            ).join(
                df.filter(pl.col("is_future"))['timestamp','symbol','expiry','bid_price','ask_price'],
                on=['timestamp','expiry'],
                how='left',
                suffix='_fut'
            ).with_columns(
                tau = (pl.col('expiry_ts') - pl.col('timestamp')).dt.total_seconds() / (365*24*3600)
            ) 
        )

    def get_symbols(self, is_option=None, is_index=None, is_perp=None, is_future=None) -> list[str]:
        """
        Get symbols filtered by type.
        
        Args:
            is_option: Filter for options
            is_index: Filter for index symbols
            is_perp: Filter for perpetual contracts
            is_future: Filter for futures
            
        Returns:
            List of filtered symbol names
        """
        df = self.df_symbol
        if is_option is not None:
            df = df.filter(pl.col("is_option") == is_option)
        if is_index is not None:
            df = df.filter(pl.col("is_index") == is_index)
        if is_perp is not None:
            df = df.filter(pl.col("is_perp") == is_perp)
        if is_future is not None:
            df = df.filter(pl.col("is_future") == is_future)
        return df.select(pl.col("symbol")).to_series().to_list()

    def is_fut_expiry(self, expiry: str) -> bool:
        """Check if expiry corresponds to a futures contract."""
        return expiry in self.fut_expiries

    @staticmethod
    def get_option_chain(df: pl.DataFrame, expiry: str, timestamp: datetime) -> pl.DataFrame:
        """
        Construct option chain by joining calls and puts on strike.
        
        Args:
            df: Market data DataFrame
            expiry: Option expiry
            timestamp: Analysis timestamp
            
        Returns:
            Option chain DataFrame with call and put data joined by strike
        """
        # Filter and get calls and puts for this expiry/timestamp
        filtered_df = df.filter(pl.col("expiry").is_in([expiry]), pl.col("timestamp") == timestamp)
        calls = filtered_df.filter(pl.col("is_call"))
        puts = filtered_df.filter(pl.col("is_put"))

        # Build column lists dynamically based on available columns
        base_call_cols = ['timestamp','bid_price','ask_price','strike']
        base_put_cols = ['bid_price','ask_price','S','bid_price_fut','ask_price_fut','strike','expiry','tau']
        
        # Add size columns if available
        if 'bid_size' in calls.columns and 'ask_size' in calls.columns:
            base_call_cols.extend(['bid_size', 'ask_size'])
            
        if 'bid_size' in puts.columns and 'ask_size' in puts.columns:
            base_put_cols.extend(['bid_size', 'ask_size'])
            
        # Add futures size columns if available
        if 'bid_size_fut' in puts.columns and 'ask_size_fut' in puts.columns:
            base_put_cols.extend(['bid_size_fut', 'ask_size_fut'])

        return calls[base_call_cols].join(
            puts[base_put_cols],
            on=['strike'],
            suffix='_P',
            validate='1:1'
        ).with_columns(
            S = pl.col('S').mode().first()  # S can vary slightly, take the mode
        ).with_columns(
            future_basis = (pl.col('bid_price_fut') + pl.col('ask_price_fut'))/2 - pl.col('S'),
        ).sort('strike')

    @staticmethod
    def get_option_synthetic(df: pl.DataFrame) -> pl.DataFrame:
        """
        Create synthetic put-call difference data for regression.
        
        Args:
            df: Option chain DataFrame
            
        Returns:
            Synthetic DataFrame with P-C differences in USD terms
        """
        result = (
            df.with_columns(
                (pl.col('S') * (pl.col('bid_price_P') - pl.col('ask_price'))).alias('bid'),  # convert to USD
                (pl.col('S') * (pl.col('ask_price_P') - pl.col('bid_price'))).alias('ask'),
            ).with_columns(
                mid=(pl.col("bid") + pl.col("ask")) / 2,
                spread=pl.col("ask") - pl.col("bid"),
            ).with_columns([
                pl.col("bid").round(2),
                pl.col("ask").round(2), 
                pl.col("mid").round(2),
                pl.col("spread").round(2)
            ])
        )
        
        # Build column list dynamically based on available columns
        base_columns = ['strike','bid','ask','mid','spread', 'S', 'tau', 'bid_price_fut', 'ask_price_fut']
        
        # Add option size columns if available (call and put sizes)
        if 'bid_size' in df.columns and 'ask_size' in df.columns:
            base_columns.extend(['bid_size', 'ask_size'])  # Call sizes
            
        if 'bid_size_P' in df.columns and 'ask_size_P' in df.columns:
            base_columns.extend(['bid_size_P', 'ask_size_P'])  # Put sizes
        
        # Add futures size columns if available
        if 'bid_size_fut' in df.columns and 'ask_size_fut' in df.columns:
            base_columns.extend(['bid_size_fut', 'ask_size_fut'])
            
        return result.select(base_columns).sort('strike')

    def create_option_synthetic(self, df: pl.DataFrame, expiry: str, timestamp: datetime, volume_threshold: float = 0.0, interest_rate: float|None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Create synthetic option data for put-call parity analysis.
        
        Args:
            df: Market data DataFrame
            expiry: Option expiry to analyze
            timestamp: Timestamp for analysis
            
        Returns:
            Tuple of (modified_option_chain, option_synthetic)
        """
        # Filter for the specific expiry and timestamp
        df_filtered = df.filter(pl.col("expiry") == expiry, pl.col("timestamp") == timestamp)
        
        if df_filtered.is_empty():
            return df_filtered, df_filtered
        
        # Get the option chain
        df_option_chain = self.get_option_chain(df_filtered, expiry=expiry, timestamp=timestamp)
        
        if df_option_chain.is_empty():
            return df_option_chain, df_option_chain
        
        # Apply spread tightening
        # print(f"📊 Modifying option chain size....")
        interest_rate = interest_rate or self.config_loader.default_interest_rate
        df_modified_option_chain = tighten_option_spreads_separate_columns(df_option_chain, volume_threshold=volume_threshold, interest_rate=interest_rate)
        
        # Create synthetic data, filtering out invalid options
        df_option_synthetic = self.get_option_synthetic(
            df_modified_option_chain.filter(
                pl.col('bid_price') > 0,
                pl.col('ask_price') > 0,
                pl.col('bid_price_P') > 0,
                pl.col('ask_price_P') > 0
            )
        )
        
        return df_modified_option_chain, df_option_synthetic    

    def is_expiry_today(self, expiry: str) -> bool:
        """Check if the given expiry is today."""
        return expiry == datetime.strptime(self.date_str, "%Y%m%d").strftime("%d%b%y").lstrip('0').upper()
    