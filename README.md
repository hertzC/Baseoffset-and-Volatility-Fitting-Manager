# Base Offset Fitter

A Python project for analyzing Bitcoin options data from Deribit to extract forward pricing and basis calculations using put-call parity regression.

## Overview

This project implements a sophisticated cryptocurrency options trading analytics system that:

- Processes Deribit Bitcoin options market data
- Constructs option chains with spread tightening and arbitrage constraints
- Performs put-call parity regression to extract USD and BTC interest rates
- Uses constrained optimization when futures data is available
- Provides interactive visualizations of results

## Project Structure

```
baseoffset-fitting-manager/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── automated_test_runner.py   # Automated test execution and reporting
├── btc_options/              # Bitcoin Options Analysis Library
│   ├── __init__.py
│   ├── data_managers/        # Market data processing
│   │   ├── __init__.py
│   │   ├── deribit_md_manager.py         # Core market data manager
│   │   └── orderbook_deribit_md_manager.py # Extended orderbook manager
│   ├── analytics/            # Regression and optimization
│   │   ├── __init__.py
│   │   ├── weight_least_square_regressor.py  # WLS regression analysis
│   │   └── nonlinear_minimization.py        # Constrained optimization
│   └── visualization/        # Interactive plotting and tables
│       ├── __init__.py
│       ├── plotly_manager.py             # Interactive visualizations
│       └── html_table_generator.py       # HTML table generation
├── tests/                     # Comprehensive test suite
│   ├── __init__.py
│   ├── test_bitcoin_options.py          # Core functionality tests
│   └── test_notebook.py                 # Notebook-specific tests
├── notebooks/                 # Jupyter notebooks for analysis
│   └── bitcoin_options_analysis.ipynb   # Main analysis notebook
├── data_bbo/                 # Best Bid/Offer market data
└── data_orderbook/           # Order Book Depth market data
```

## Configuration

### Data Source Selection
The system supports two types of market data:

```python
# In main.py or jupyter notebook
use_orderbook_data = False  # Set True/False to choose data type

# False: Best Bid/Offer data (data_bbo/)
# True:  Order Book Depth data (data_orderbook/)
```

### Directory Structure
```
├── data_bbo/                 # Best Bid/Offer market data
├── data_orderbook/           # Order Book Depth market data (optional)
```

### Put-Call Parity Analysis
- Implements the equation: `P - C = K*exp(-r*t) - S*exp(-q*t)`
- Extracts USD risk-free rate (r) and BTC funding rate (q)
- Calculates forward prices and basis

### Market Data Processing
- Handles Deribit market data format
- Applies monotonicity constraints to option spreads
- Implements no-arbitrage bounds
- Filters invalid data points

### Optimization Methods
- Weighted Least Squares (WLS) for unconstrained fitting
- Constrained optimization using futures market bounds
- Time-series fitting with warm start parameters

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
```bash
python main.py
```

This will run the complete analysis pipeline:
1. **Data Loading**: Automatically loads data or uses sample data if not available
2. **Market Processing**: Conflates data and creates option chains  
3. **Analysis**: Performs WLS regression and constrained optimization
4. **Results**: Displays extracted rates and forward prices

### Jupyter Notebook Analysis
```bash
jupyter notebook
# Open notebooks/bitcoin_options_analysis.ipynb
```

### Command Line Options
The main script automatically handles:
- ✅ Missing data files (uses sample data)
- ✅ Error handling and graceful degradation
- ✅ Detailed progress reporting
- ✅ Single timestamp analysis demonstration
- ✅ Optional time series analysis (commented out)

### Sample Output
```
🚀 Base Offset Fitter - Cryptocurrency Options Analytics
📊 Analyzing Bitcoin options using put-call parity regression

🔄 Step 1: Loading market data...
📊 Using sample data for demonstration

🔧 Step 2: Initializing analysis components...
📊 Available option expiries: ['29FEB24']
🔮 Available future expiries: ['29FEB24']

⚙️  Step 3: Conflating market data...
📈 Conflated data shape: (6, 13)

🎯 Step 4: Single timestamp analysis...
💰 USD Interest Rate (r): 0.0245
₿  BTC Funding Rate (q): 0.0123
📊 Forward Price (F): 62125.50
📈 R-squared: 0.9876

🎉 Analysis completed successfully!
```

## Data Format

Expected input data format from Deribit:
- Symbol naming: `BTC-{expiry}-{strike}-{C|P}` for options, `BTC-{expiry}` for futures
- Timestamps in `HH:MM:SS.fff` format
- Bid/ask prices in BTC terms

## Mathematical Background

The system implements put-call parity regression where:
- **Forward price**: `F = S*exp((r-q)*t)`
- **Rate extraction**: `r = -ln(coef)/t`, `q = -ln(-const/S)/t`
- **Constraints**: Forward price bounded by futures bid-ask when available

## Requirements

- Python 3.8+
- Polars for efficient data processing
- SciPy for optimization
- Plotly for interactive visualization
- Jupyter for notebook-based analysis