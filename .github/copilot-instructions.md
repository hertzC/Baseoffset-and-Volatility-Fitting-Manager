# Base Offset Fitter - AI Coding Assistant Guide

## Project Overview
This is a **quantitative finance** project that analyzes Bitcoin options data from Deribit to extract forward pricing and basis calculations using **put-call parity regression**. The system implements sophisticated financial mathematics to derive USD interest rates and BTC funding rates from options market data.

## Core Architecture

### Data Pipeline Flow
1. **Raw Deribit Data** → `DeribitMDManager` → **Conflated Market Data**
2. **Conflated Data** → **Option Chain Construction** → **Synthetic Pricing**
3. **Synthetic Data** → `WLSRegressor` OR `NonlinearMinimization` → **Forward Rates**
4. **Results** → `PlotlyManager` → **Interactive Visualizations**

### Key Mathematical Relationships
- **Put-Call Parity**: `P - C = K*exp(-r*t) - S*exp(-q*t)`
- **Forward Price**: `F = S*exp((r-q)*t)` where r=USD rate, q=BTC funding rate
- **Linearized Regression**: `y = a*K + b` where `a = exp(-r*t)`, `b = -S*exp(-q*t)`

## Critical Components

### `/common/` Modules (Never modify signatures without understanding downstream impact)
- **`deribit_md_manager.py`**: Core data processing with symbol parsing (`BTC-{expiry}-{strike}-{C|P}`)
- **`weight_least_square_regressor.py`**: Implements WLS using statsmodels with weight=1/spread²
- **`nonlinear_minimization.py`**: Extends WLS with scipy.optimize constraints from futures bounds
- **`plotly_manager.py`**: Creates regression plots with mathematical formulas and error bars

### Symbol Naming Convention
- **Options**: `BTC-29FEB24-60000-C` (Call) / `BTC-29FEB24-60000-P` (Put)
- **Futures**: `BTC-29FEB24`
- **Index**: `INDEX` (spot price reference)
- **Perpetual**: `BTC-PERPETUAL`

## Development Patterns

### Error Handling Strategy
- **Graceful degradation**: Missing data files → automatic sample data generation
- **Parameter validation**: Empty DataFrames raise `ValueError` with descriptive messages
- **Warm starts**: Previous regression coefficients as initial guesses for optimization

### Data Processing Conventions
- **Lazy evaluation**: Use `pl.LazyFrame` for initial data loading, `.collect()` only when needed
- **Time conflation**: Regular intervals (`freq="1m"`) with lookback periods (`period="10m"`)
- **Spread filtering**: Filter by `spread < threshold` to remove illiquid strikes
- **Monotonicity constraints**: Enforce no-arbitrage bounds in option chain construction

### Fitting Workflow (Critical Sequence)
```python
# 1. Always check data availability
if df_option_synthetic.is_empty() or len(df_option_synthetic) < 3:
    # Handle insufficient data case

# 2. WLS first (unconstrained)
wls_result = wls_regressor.fit(df_option_synthetic)

# 3. Constrained optimization (uses WLS as initial guess)
constrained_result = nonlinear_minimizer.fit(
    df_option_synthetic, wls_result['const'], wls_result['coef']
)
```

## File Organization

### Data Expectations
- **Input**: `data_bbo/{YYYYMMDD}.market_updates-*.log` OR `data_orderbook/{YYYYMMDD}.market_updates-*.log` (CSV format with timestamp, symbol, bid_price, ask_price)
- **Configuration**: Set `use_orderbook_data = True/False` to choose between order book depth and best bid/offer data
- **Notebooks**: `notebooks/bitcoin_options_analysis.ipynb` for interactive analysis
- **Output**: Results are in-memory dictionaries with standardized keys: `r`, `q`, `F`, `r2`, `sse`

### Configuration Patterns
```python
date_str = "20240229"  # YYYYMMDD format
conflation_every = "1m"   # Time interval
conflation_period = "10m" # Lookback window
```

## Testing and Debugging

### Run Commands
```bash
python main.py                    # Full pipeline with sample data
jupyter notebook                  # Interactive analysis
pip install -r requirements.txt  # Dependencies
```

### Common Issues
- **Import errors**: Notebook cells modify `sys.path` to access `/common/` modules
- **Data format**: Ensure Deribit CSV has columns: symbol, timestamp, bid_price, ask_price
- **Time zones**: All timestamps assumed in same timezone (typically UTC)
- **Optimization failures**: Constrained optimization may fail → fallback to WLS results

## Key Dependencies & Versions
- **Polars** (not pandas): Primary data processing, lazy evaluation
- **Statsmodels**: WLS regression (`sm.WLS`)
- **Scipy**: Constrained optimization (`scipy.optimize.minimize`)
- **Plotly**: Interactive visualization with LaTeX math rendering

## Jupyter Notebook Workflow
1. **Setup**: Import libraries, configure Python path for `/common/` access
2. **Data Loading**: Load or generate sample data, initialize pipeline components
3. **Conflation**: Process raw ticks into regular intervals
4. **Single Analysis**: Demonstrate fitting for one expiry/timestamp
5. **Time Series**: Loop through multiple timestamps for comprehensive analysis
6. **Visualization**: Generate interactive plots with regression results

## Production Considerations
- **Memory**: Use lazy evaluation for large datasets
- **Performance**: WLS regression before constrained optimization (warm start pattern)
- **Validation**: Check `r2` values and compare WLS vs constrained results
- **Monitoring**: Log regression failures and data quality issues