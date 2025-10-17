#!/usr/bin/env python3
"""
Base Offset Fitter - Cryptocurrency Options Trading Analytics

This project analyzes Bitcoin options data from Deribit to extract forward pricing
and basis calculations using put-call parity regression.

Main components:
- Market data processing from Deribit
- Option chain construction and spread tightening
    # Load market data
    data_file = config.get_data_file_path()
    
    df_md = load_market_data(config, data_file)
    print(f"📊 Loaded market data with {df_md.collect().height:,} records")
    
    # Initialize managers
    if config.use_orderbook_data:
        symbol_manager = OrderbookDeribitMDManager()
        print("🔧 Using OrderBook data manager")
    else:
        symbol_manager = DeribitMDManager()
        print("🔧 Using BBO data manager")
    
    wls_regressor = WLSRegressor(symbol_manager)
    nonlinear_minimizer = NonlinearMinimization(symbol_manager)
    
    # Apply configuration to fitting algorithms
    wls_regressor.set_printable(config.get('fitting.wls.printable', False))
    nonlinear_minimizer.set_printable(config.get('fitting.nonlinear.printable', False))
    nonlinear_minimizer.future_spread_mult = config.future_spread_mult
    nonlinear_minimizer.lambda_reg = config.lambda_reg
    
    # Apply rate constraints if configured
    rate_constraints = config.get_rate_constraints()
    print(f"📋 Rate constraints: r ∈ [{rate_constraints['r_min']:.1%}, {rate_constraints['r_max']:.1%}], "
          f"q ∈ [{rate_constraints['q_min']:.1%}, {rate_constraints['q_max']:.1%}]") regression (WLS and constrained optimization)
- CSV export of results for production use
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import warnings

# Import configuration loader
from config.config_loader import load_config, ConfigurationError

# Import project modules from utils
from utils.market_data.deribit_md_manager import DeribitMDManager
from utils.market_data.orderbook_deribit_md_manager import OrderbookDeribitMDManager
from utils.base_offset_fitter.weight_least_square_regressor import WLSRegressor
from utils.base_offset_fitter.nonlinear_minimization import NonlinearMinimization
from utils.base_offset_fitter.fitter_result_manager import FitterResultManager
from utils.base_offset_fitter.maths import convert_rate_into_parameter


def load_market_data(config, data_file: str) -> pl.LazyFrame:
    """
    Load Deribit market data from CSV file.
    
    Args:
        date_str: Date string in YYYYMMDD format
        data_file: Path to the data file
        use_orderbook_data: If True, load as orderbook depth data, otherwise as BBO data
        
    Returns:
        LazyFrame with market data or sample data if file not found
    """
    # Check if file exists first
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print(f"💡 Please ensure the data file exists and is accessible")
        sys.exit(1)
        
    # Load data - both types use the same lazy loading approach
    if config.use_orderbook_data:
        print(f"📖 Loading order book depth data from {data_file}")
    else:
        print(f"📈 Loading BBO data from {data_file}")
        
    try:
        df_market_updates = pl.scan_csv(data_file)
        
        # Validate data structure
        _ = df_market_updates.select("symbol").unique().collect()
        print(f"✅ Successfully loaded data from {data_file}")
        return df_market_updates
        
    except Exception as e:
        print(f"❌ Error loading data from {data_file}: {e}")
        print(f"� Please check the file format and ensure it contains valid CSV data")
        sys.exit(1)


def run_comprehensive_analysis(symbol_manager: DeribitMDManager,
                               wls_regressor: WLSRegressor, 
                               nonlinear_minimizer: NonlinearMinimization,
                               df_conflated_md: pl.DataFrame,
                               use_constrained_optimization: bool,
                               time_interval_seconds: int) -> tuple[pl.DataFrame, dict]:
    """
    Run comprehensive time series analysis across all expiries using notebook logic.
    
    Args:
        symbol_manager: Market data manager
        wls_regressor: WLS regression fitter
        nonlinear_minimizer: Constrained optimization fitter
        df_conflated_md: Conflated market data
        use_constrained_optimization: Whether to use constrained optimization
        time_interval_seconds: Seconds between analysis points
        
    Returns:
        Tuple of (results DataFrame, success statistics)
    """
    print(f"🚀 Starting comprehensive analysis across all expiries...")
    print(f"🎯 Mode: {'Constrained Optimization' if use_constrained_optimization else 'WLS Only'}")
    print(f"⏱️  Sampling interval: {time_interval_seconds} seconds")
    
    # Configuration
    successful_fits = {}
    fitter = nonlinear_minimizer if use_constrained_optimization else wls_regressor
    
    # Disable verbose output for time series
    wls_regressor.set_printable(False)
    nonlinear_minimizer.reset_parameters()
    nonlinear_minimizer.clear_results()
    nonlinear_minimizer.set_printable(False)
    
    # Calculate time ranges for each expiry
    start_time_map = {
        each['expiry']: each for each in df_conflated_md.group_by('expiry').agg(
            pl.col('timestamp').first().alias('start_time'),
            pl.col('timestamp').last().alias('end_time')
        ).to_dicts()
    }
    
    print(f"📊 Processing {len(symbol_manager.opt_expiries)} expiries...")
    
    try:
        # Process each expiry
        for expiry in symbol_manager.opt_expiries:
            print(f"\n🔄 Processing expiry: {expiry}")
            successful_fits[expiry] = {'total': 0, 'successful': 0}
            
            # Calculate time range for this expiry
            start_time = start_time_map[expiry]['start_time'] + timedelta(seconds=time_interval_seconds)
            if symbol_manager.is_expiry_today(expiry):
                end_time = start_time.replace(hour=7, minute=0, second=0)
            else:
                end_time = start_time.replace(hour=23, minute=59, second=0)
            
            sampled_timestamps = pl.datetime_range(
                start=start_time, 
                end=end_time, 
                interval=f"{time_interval_seconds}s", 
                eager=True
            ).to_list()
            
            print(f"   � Time range: {start_time} to {end_time}")
            print(f"   🕒 Total timestamps: {len(sampled_timestamps)}")
            
            # Initialize per-expiry variables
            initial_guess = None
            
            # Process each timestamp for this expiry
            for ts in sampled_timestamps:
                try:                    
                    # Create option synthetic data for this timestamp
                    df_chain, df_synthetic = symbol_manager.create_option_synthetic(
                        df_conflated_md, expiry=expiry, timestamp=ts
                    )
                    
                    if not df_synthetic.is_empty():
                        result = None
                        tau, s0 = df_synthetic['tau'][0], df_synthetic['S'][0]
                        
                        # Check if we should use cutoff for 0DTE
                        is_cutoff, cutoff_result = fitter.check_if_cutoff_for_0DTE(
                            expiry, ts, symbol_manager.is_expiry_today(expiry), s0, tau
                        )
                        
                        if is_cutoff:
                            result = cutoff_result
                            continue
                        successful_fits[expiry]['total'] += 1
                        
                        # Perform fitting based on mode
                        if use_constrained_optimization:
                            # Initialize with WLS if needed
                            if initial_guess is None or (np.isnan(initial_guess[0]) and np.isnan(initial_guess[1])):
                                wls_temp_result = wls_regressor.fit(df_synthetic, expiry=expiry, timestamp=ts)
                                initial_guess = (wls_temp_result['r'], wls_temp_result['q'])
                            
                            # Convert rates to parameters
                            initial_guess_const, initial_guess_coef = convert_rate_into_parameter(initial_guess, s0, tau)
                            
                            # Constrained optimization
                            result = fitter.fit(df_synthetic, initial_guess_const, initial_guess_coef, expiry=expiry, timestamp=ts)
                            initial_guess = (result['r'], result['q'])
                        else:
                            # WLS only mode
                            result = fitter.fit(df_synthetic, expiry=expiry, timestamp=ts)
                        
                        if result and result['success_fitting']:
                            successful_fits[expiry]['successful'] += 1
                    else:
                        # Skip empty synthetic data
                        continue
                        
                except Exception as e:
                    print(f"   ⚠️  Error at {ts}: {e}")
                    continue
            
            success_rate = (successful_fits[expiry]['successful'] / successful_fits[expiry]['total']) * 100 if successful_fits[expiry]['total'] > 0 else 0
            print(f"   ✅ Completed: {successful_fits[expiry]['successful']}/{successful_fits[expiry]['total']} successful fits ({success_rate:.1f}%)")
    
    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")
        raise
    
    # Create results using FitterResultManager
    print(f"\n📊 Creating results DataFrame...")
    fit_result_manager = FitterResultManager(
        symbol_manager.opt_expiries, 
        symbol_manager.fut_expiries, 
        fitter.symbol_manager.df_symbol,
        fitter.fit_results, 
        successful_fits, 
        old_weight=0.95
    )
    
    df_results = fit_result_manager.create_results_df(fit_result_manager.fit_results).sort('timestamp')
    
    total_results = len(df_results)
    successful_results = len(df_results.filter(pl.col('success_fitting') == True))
    
    print(f"✅ Analysis complete!")
    print(f"   📈 Total results: {total_results:,}")
    print(f"   ✅ Successful fits: {successful_results:,}")
    print(f"   📊 Success rate: {(successful_results/total_results)*100:.1f}%")
    
    return df_results, successful_fits


def save_results_to_csv(df_results: pl.DataFrame, 
                        df_conflated_md: pl.DataFrame,
                        date_str: str) -> tuple[str|None, str|None]:
    """
    Save analysis results to CSV files if quality is acceptable.
    
    Args:
        df_results: Results DataFrame
        date_str: Date string for filename
        use_constrained: Whether constrained optimization was used
        
    Returns:
        Tuple of (results_path, summary_path) if saved, otherwise None
    """
    if df_results.is_empty():
        print("❌ No results to save - DataFrame is empty")
        return None, None

    # Check data quality
    total_results = len(df_results)
    successful_results = len(df_results.filter(pl.col('success_fitting') == True))
    success_rate = (successful_results / total_results) * 100 if total_results > 0 else 0
        
    print(f"\n� Data Quality Check:")
    print(f"   📊 Total results: {total_results:,}")
    print(f"   ✅ Successful fits: {successful_results:,}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    
    print("✅ Quality check passed - proceeding with CSV export")
    
    # Create exports directory
    exports_dir = "results/" + date_str
    os.makedirs(exports_dir, exist_ok=True)
        
    results_filename = f"baseoffset_results.csv"
    md_filename = "conflated_md.csv"
    
    results_path = os.path.join(exports_dir, results_filename)
    md_path = os.path.join(exports_dir, md_filename)
    
    try:
        # Save main results
        df_results.write_csv(results_path)        
        
        print(f"\n💾 Files saved successfully:")
        print(f"   📊 Results: {results_filename}")
        print(f"   📁 Location: {os.path.abspath(exports_dir)}")
        
        # Save conflated market data
        df_conflated_md.write_csv(md_path)
        print(f"   📊 Conflated Market Data: {md_filename}")
        print(f"   📁 Location: {os.path.abspath(exports_dir)}")

        # Display key metrics
        print(f"\n🎯 Key Results Summary:")
        avg_r = df_results.filter(pl.col('success_fitting') == True)['r'].mean() * 100
        avg_q = df_results.filter(pl.col('success_fitting') == True)['q'].mean() * 100
        avg_spread = df_results.filter(pl.col('success_fitting') == True)['r-q'].mean() * 100
        avg_r2 = df_results.filter(pl.col('success_fitting') == True)['r2'].mean()
        
        print(f"   💰 Average USD rate (r): {avg_r:.2f}%")
        print(f"   ₿  Average BTC rate (q): {avg_q:.2f}%")
        print(f"   📊 Average rate spread: {avg_spread:.2f}%")
        print(f"   📈 Average R²: {avg_r2:.4f}")
        
        return results_path, md_path
        
    except Exception as e:
        print(f"❌ Error saving CSV files: {e}")
        return None, None


def main():
    """
    Main analysis pipeline using notebook-style comprehensive fitting approach.
    """
    print("=" * 60)
    print("🚀 Base Offset Fitter - Cryptocurrency Options Analytics")
    print("=" * 60)
    print("📊 Analyzing Bitcoin options using put-call parity regression")
    print()
    
    # Load configuration
    try:
        config = load_config()
        print(f"✅ Configuration loaded successfully")
        print(f"📅 Analysis date: {config.date_str}")
        print(f"📊 Data source: {'Order book depth' if config.use_orderbook_data else 'Best bid/offer'}")
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    print(f"⚙️  Configuration Summary:")
    print(f"   🔧 Conflation: every {config.conflation_every}, period {config.conflation_period}")
    print(f"   🎯 Method: {'Constrained optimization' if config.use_constrained_optimization else 'WLS only'}")
    print(f"   ⏱️  Sampling interval: {config.time_interval_seconds} minutes")
    print(f"   🎚️  Smoothing weight: {config.old_weight}")
    
    # Load market data
    data_file = config.get_data_file_path()
    
    df_md = load_market_data(config, data_file)
    print(f"� Loaded market data with {df_md.collect().height:,} records")
    
    # Initialize managers
    if config.use_orderbook_data:
        symbol_manager = OrderbookDeribitMDManager(df_md, date_str=config.date_str, config_loader=config)
        print("🔧 Using OrderBook data manager")
    else:
        symbol_manager = DeribitMDManager(df_md, date_str=config.date_str)
        print("🔧 Using BBO data manager")
    
    wls_regressor = WLSRegressor(symbol_manager, config)
    nonlinear_minimizer = NonlinearMinimization(symbol_manager, config)
    
    # Conflate market data
    print(f"\n🔄 Conflating market data...")
    df_conflated_md = symbol_manager.get_conflated_md(
        freq=config.conflation_every, period=config.conflation_period
    ).sort(['expiry', 'timestamp'])
    
    print(f"✅ Conflated data: {len(df_conflated_md):,} records")
    print(f"📈 Available expiries: {len(symbol_manager.opt_expiries)} option, {len(symbol_manager.fut_expiries)} future")
    print(f"   Options: {symbol_manager.opt_expiries}")
    print(f"   Futures: {symbol_manager.fut_expiries}")
    
    # Run comprehensive analysis
    try:
        df_results, successful_fits = run_comprehensive_analysis(
            symbol_manager=symbol_manager,
            wls_regressor=wls_regressor,
            nonlinear_minimizer=nonlinear_minimizer, 
            df_conflated_md=df_conflated_md,
            use_constrained_optimization=config.use_constrained_optimization,
            time_interval_seconds=config.time_interval_seconds
        )
        
        if df_results.is_empty():
            print("❌ No results generated - analysis failed")
            return
        
        print(f"\n📊 Analysis Results Summary:")
        total_fits = len(df_results)
        successful_fits_count = len(df_results.filter(pl.col('success_fitting') == True))
        print(f"   📈 Total fits attempted: {total_fits:,}")
        print(f"   ✅ Successful fits: {successful_fits_count:,}")
        print(f"   📊 Overall success rate: {(successful_fits_count/total_fits)*100:.1f}%")

        # Save results to CSV if quality is acceptable
        results_path, md_path = save_results_to_csv(
            df_results=df_results,
            df_conflated_md=df_conflated_md,
            date_str=config.date_str
        )
        
        if results_path:
            print(f"\n🎉 Analysis pipeline completed successfully!")
            print(f"� Results exported to: {os.path.dirname(results_path)} and {os.path.dirname(md_path)}")
        else:
            print(f"\n⚠️  Analysis completed but results not exported (quality check failed)")
            
    except Exception as e:
        print(f"❌ Critical error in analysis pipeline: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n✨ Pipeline finished!")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    main()