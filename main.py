#!/usr/bin/env python3
"""
Base Offset Fitter - Cryptocurrency Options Trading Analytics

This project analyzes Bitcoin options data from Deribit to extract forward pricing
and basis calculations using put-call parity regression.

Main components:
- Market data processing from Deribit
- Option chain construction and spread tightening
- Put-call parity regression (WLS and constrained optimization)
- Interactive visualization of results
"""

import polars as pl
from datetime import datetime, timedelta
import os
import sys

# Import project modules from btc_options
from btc_options.data_managers.deribit_md_manager import DeribitMDManager
from btc_options.data_managers.orderbook_deribit_md_manager import OrderbookDeribitMDManager
from btc_options.visualization.plotly_manager import PlotlyManager
from btc_options.analytics.weight_least_square_regressor import WLSRegressor
from btc_options.analytics.nonlinear_minimization import NonlinearMinimization


def load_market_data(date_str: str, data_file: str, use_orderbook_data: bool = False) -> pl.LazyFrame:
    """
    Load Deribit market data from CSV file.
    
    Args:
        date_str: Date string in YYYYMMDD format
        data_file: Path to the data file
        use_orderbook_data: If True, load as orderbook depth data, otherwise as BBO data
        
    Returns:
        LazyFrame with market data or sample data if file not found
    """
    try:
        # Check if file exists first
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        # Load data - both types use the same lazy loading approach
        if use_orderbook_data:
            print(f"📖 Loading order book depth data from {data_file}")
        else:
            print(f"📈 Loading BBO data from {data_file}")
            
        df_market_updates = pl.scan_csv(data_file)
        
        # Validate data structure
        _ = df_market_updates.select("symbol").unique().collect()
        print(f"✅ Successfully loaded data from {data_file}")
        return df_market_updates
        
    except Exception as e:
        print(f"⚠️  Error loading data from {data_file}: {e}")
        print("📝 Creating sample data for demonstration...")
        
        # Create realistic sample data structure
        sample_data = pl.LazyFrame({
            "symbol": [
                "BTC-29FEB24-60000-C", "BTC-29FEB24-60000-P",
                "BTC-29FEB24-65000-C", "BTC-29FEB24-65000-P", 
                "INDEX", "BTC-29FEB24"
            ],
            "timestamp": [
                "08:00:00.000", "08:00:00.000", "08:00:00.000", 
                "08:00:00.000", "08:00:00.000", "08:00:00.000"
            ],
            "bid_price": [0.05, 0.02, 0.03, 0.04, 62000.0, 62500.0],
            "ask_price": [0.06, 0.03, 0.04, 0.05, 62000.0, 62600.0]
        })
        print("📊 Using sample data for demonstration")
        return sample_data


def analyze_single_expiry(symbol_manager: DeribitMDManager, 
                         plotly_manager: PlotlyManager,
                         wls_regressor: WLSRegressor,
                         nonlinear_minimizer: NonlinearMinimization,
                         df_conflated_md: pl.DataFrame, 
                         expiry: str, 
                         timestamp: datetime,
                         verbose: bool = True) -> dict:
    """
    Analyze a single expiry at a specific timestamp.
    
    Args:
        symbol_manager: Market data manager
        plotly_manager: Visualization manager
        wls_regressor: WLS regression fitter
        nonlinear_minimizer: Constrained optimization fitter
        df_conflated_md: Conflated market data
        expiry: Option expiry to analyze
        timestamp: Analysis timestamp
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    if verbose:
        print(f"📈 Analyzing expiry: {expiry} at time: {timestamp}")
    
    try:
        # Create option synthetic data
        df_option_chain, df_modified_option_chain, df_option_synthetic = symbol_manager.create_option_synthetic(
            df_conflated_md, expiry=expiry, timestamp=timestamp
        )
        
        if df_option_synthetic.is_empty():
            if verbose:
                print(f"⚠️  No synthetic data available for {expiry} at {timestamp}")
            return None
            
        if verbose:
            print(f"📊 Option chain size: {len(df_option_chain)}")
            print(f"🔧 Synthetic data size: {len(df_option_synthetic)}")
        
        # WLS Regression
        wls_result = wls_regressor.fit(df_option_synthetic)
        
        if verbose:
            print(f"💰 USD Interest Rate (r): {wls_result['r']:.4f}")
            print(f"₿  BTC Funding Rate (q): {wls_result['q']:.4f}")
            print(f"📊 Forward Price (F): {wls_result['F']:.2f}")
            print(f"📈 R-squared: {wls_result['r2']:.4f}")
        
        # Try constrained optimization
        try:
            constrained_result = nonlinear_minimizer.fit(
                df_option_synthetic, wls_result['const'], wls_result['coef']
            )
            
            if verbose:
                print(f"🎯 Constrained Forward Price: {constrained_result['F']:.2f}")
                print(f"📊 Constrained R-squared: {constrained_result['r2']:.4f}")
                
            # Use constrained result if better
            final_result = constrained_result if constrained_result['r2'] > wls_result['r2'] else wls_result
            
        except ValueError as e:
            if verbose:
                print(f"⚠️  Constrained optimization failed: {e}")
            final_result = wls_result
        
        # Add metadata
        result = {
            'expiry': expiry,
            'timestamp': timestamp,
            'synthetic_count': len(df_option_synthetic),
            **final_result
        }
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"❌ Error analyzing {expiry}: {e}")
        return None


def run_time_series_analysis(symbol_manager: DeribitMDManager,
                           plotly_manager: PlotlyManager,
                           wls_regressor: WLSRegressor, 
                           nonlinear_minimizer: NonlinearMinimization,
                           df_conflated_md: pl.DataFrame,
                           start_time: datetime = None,
                           end_time: datetime = None,
                           interval_minutes: int = 5) -> pl.DataFrame:
    """
    Run time series analysis across multiple timestamps.
    
    Args:
        symbol_manager: Market data manager
        plotly_manager: Visualization manager  
        wls_regressor: WLS regression fitter
        nonlinear_minimizer: Constrained optimization fitter
        df_conflated_md: Conflated market data
        start_time: Analysis start time
        end_time: Analysis end time  
        interval_minutes: Minutes between analysis points
        
    Returns:
        DataFrame with time series results
    """
    if not symbol_manager.opt_expiries:
        print("⚠️  No option expiries available for time series analysis")
        return pl.DataFrame()
    
    # Default time range
    if start_time is None:
        start_time = datetime(2024, 2, 29, 8, 0, 0)
    if end_time is None:
        end_time = datetime(2024, 2, 29, 16, 0, 0)
    
    print(f"🕒 Running time series analysis from {start_time} to {end_time}")
    print(f"📊 Analyzing expiries: {symbol_manager.opt_expiries}")
    
    results = []
    minimum_strikes = 3
    
    # Disable verbose output for time series
    wls_regressor.set_printable(False)
    nonlinear_minimizer.set_printable(False)
    
    for expiry in symbol_manager.opt_expiries:
        print(f"\n📈 Processing expiry: {expiry}")
        prev_const, prev_coef = -62000, 1.0  # Initial guess
        
        current_time = start_time
        expiry_results = 0
        
        while current_time <= end_time:
            result = analyze_single_expiry(
                symbol_manager, plotly_manager, wls_regressor, nonlinear_minimizer,
                df_conflated_md, expiry, current_time, verbose=False
            )
            
            if result and result['synthetic_count'] >= minimum_strikes:
                results.append(result)
                prev_const, prev_coef = result['const'], result['coef']
                expiry_results += 1
            
            current_time += timedelta(minutes=interval_minutes)
        
        print(f"✅ Collected {expiry_results} valid results for {expiry}")
    
    if results:
        df_results = pl.DataFrame(results).with_columns(
            (pl.col('r') - pl.col('q')).alias('r-q'),
            base_offset=pl.col('F') - pl.col('S')
        )
        print(f"🎉 Time series analysis complete: {len(results)} total results")
        return df_results
    else:
        print("❌ No valid results collected")
        return pl.DataFrame()


def main():
    """Main entry point for the base offset fitter application."""
    print("=" * 60)
    print("🚀 Base Offset Fitter - Cryptocurrency Options Analytics")
    print("=" * 60)
    print("📊 Analyzing Bitcoin options using put-call parity regression")
    print()
    
    # Configuration
    date_str = "20240229"  # YYYYMMDD format
    use_orderbook_data = False  # Set True to use order book depth data, False for BBO data
    conflation_every = "1m"   # 1-minute intervals
    conflation_period = "10m" # 10-minute lookback
    
    # Determine data source based on configuration
    if use_orderbook_data:
        data_dir = "data_orderbook"
        data_type = "Order Book Depth"
    else:
        data_dir = "data_bbo"
        data_type = "Best Bid/Offer (BBO)"
    
    data_file = f'{data_dir}/{date_str}.market_updates-1318071-20250916.log'
    
    try:
        # 1. Load market data
        print("🔄 Step 1: Loading market data...")
        print(f"📂 Data source: {data_type} from {data_dir}/")
        df_market_updates = load_market_data(date_str, data_file)
        
        # 2. Initialize components
        print("\n🔧 Step 2: Initializing analysis components...")
        # Initialize symbol manager with appropriate class
        if use_orderbook_data:
            orderbook_level = 0  # Use best bid/ask (level 0)
            print(f"🔧 Using OrderbookDeribitMDManager for orderbook data conversion")
            print(f"   - Level: {orderbook_level} ({'best' if orderbook_level == 0 else f'{orderbook_level}th best'})")
            symbol_manager = OrderbookDeribitMDManager(df_market_updates, date_str, level=orderbook_level)
        else:
            print("🔧 Using standard DeribitMDManager for BBO data")
            symbol_manager = DeribitMDManager(df_market_updates, date_str)
        wls_regressor = WLSRegressor()
        nonlinear_minimizer = NonlinearMinimization()  # Rate constraints: r ∈ [-5%, +10%], q ∈ [-30%, +100%]
        
        # For custom rate constraints in optimization, use:
        # nonlinear_minimizer = NonlinearMinimization(r_min=-0.02, r_max=0.08, q_min=-0.10, q_max=0.50)
        plotly_manager = PlotlyManager(date_str, symbol_manager.fut_expiries)
        
        print(f"📊 Available option expiries: {symbol_manager.opt_expiries}")
        print(f"🔮 Available future expiries: {symbol_manager.fut_expiries}")
        
        # 3. Process market data
        print(f"\n⚙️  Step 3: Conflating market data (freq={conflation_every}, period={conflation_period})...")
        df_conflated_md = symbol_manager.get_conflated_md(
            freq=conflation_every, 
            period=conflation_period
        )
        print(f"📈 Conflated data shape: {df_conflated_md.shape}")
        
        # 4. Single timestamp analysis (demonstration)
        if symbol_manager.opt_expiries:
            print(f"\n🎯 Step 4: Single timestamp analysis...")
            expiry = symbol_manager.opt_expiries[0]
            timestamp = datetime(2024, 2, 29, 8, 21, 0)
            
            wls_regressor.set_printable(True)
            nonlinear_minimizer.set_printable(True)
            
            result = analyze_single_expiry(
                symbol_manager, plotly_manager, wls_regressor, nonlinear_minimizer,
                df_conflated_md, expiry, timestamp
            )
            
            if result:
                print(f"✅ Single analysis completed successfully")
            else:
                print(f"⚠️  Single analysis returned no results")
        
        # 5. Time series analysis (optional)
        print(f"\n📊 Step 5: Time series analysis...")
        print("💡 For full time series analysis with real data:")
        print("   - Uncomment the time series section below")
        print("   - Ensure you have sufficient market data")
        print("   - Adjust time ranges as needed")
        
        # Uncomment for time series analysis:
        # df_time_series = run_time_series_analysis(
        #     symbol_manager, plotly_manager, wls_regressor, nonlinear_minimizer,
        #     df_conflated_md,
        #     start_time=datetime(2024, 2, 29, 8, 0, 0),
        #     end_time=datetime(2024, 2, 29, 16, 0, 0),
        #     interval_minutes=5
        # )
        
        print(f"\n🎉 Analysis completed successfully!")
        print("📝 Next steps:")
        print("   1. Add real Deribit data files to data_bbo/ or data_orderbook/ directory")
        print("   2. Run the Jupyter notebook for interactive analysis")
        print("   3. Modify parameters for your specific use case")
        print("   4. Enable time series analysis for comprehensive results")
        
    except Exception as e:
        print(f"\n❌ Error in main analysis: {e}")
        print("💡 Make sure all dependencies are installed and data is available")
        sys.exit(1)


if __name__ == "__main__":
    main()