# 📈 Volatility Surface Fitting Methodology Guide

## 🎯 Executive Summary

This guide details the comprehensive methodology for fitting volatility surfaces to Bitcoin options market data using advanced wing model techniques. The process combines sophisticated mathematical modeling with robust optimization algorithms to extract reliable implied volatility surfaces that capture market smile dynamics while maintaining arbitrage-free properties.

**Key Features:**
- **Time-Adjusted Wing Model**: Advanced moneyness calculation with normalization terms
- **Multi-region Surface**: Central parabola with smoothing wings and flat extrapolation
- **Global Optimization**: Differential Evolution for robust parameter estimation
- **Arbitrage Validation**: Durrleman condition enforcement for market consistency
- **Real-time Calibration**: Efficient algorithms suitable for live trading systems

---

## 📋 Methodology Overview

### **Core Components**
1. **Market Data Collection**: Bitcoin options bid/ask data with volume and delta information
2. **Data Quality Enhancement**: Strike filtering, volume thresholding, and outlier removal
3. **Model Architecture**: Time-adjusted wing model with configurable parameters
4. **Parameter Optimization**: Global and local calibration algorithms
5. **Arbitrage Validation**: Mathematical consistency checks
6. **Surface Generation**: Complete volatility surface construction
7. **Quality Assessment**: Statistical validation and error analysis
8. **Results Output**: Calibrated parameters and volatility surfaces

## 📊 Step-by-Step Process

### **Step 1: Market Data Collection and Preparation**

#### 1.1 Data Source Selection
```python
# Configure data source
config = VolatilityConfig("config/volatility_config.yaml")
date_str = "20240229"
use_orderbook = config.data.use_orderbook_data  # True for orderbook, False for BBO

# Load market data
if use_orderbook:
    df_market_data = pl.read_csv(f"data_orderbook/{date_str}.output.csv.gz")
else:
    df_market_data = process_bbo_data(f"data_bbo/{date_str}.market_updates.log")
```

#### 1.2 Options Chain Filtering
```python
def filter_options_chain(df: pl.DataFrame, config: VolatilityConfig) -> pl.DataFrame:
    """Apply data quality filters to options chain"""
    return df.filter(
        # Delta range constraints
        (pl.col('call_delta') >= config.market_data.min_delta) &
        (pl.col('call_delta') <= config.market_data.max_delta) &
                
        # Minimum strikes requirement
        (pl.len().over('expiry') >= config.market_data.min_strikes)
    )
```

### **Step 2: Model Parameter Initialization**

#### 2.1 Time-Adjusted Wing Model Setup
```python
from utils.volatility_fitter.time_adjusted_wing_model import TimeAdjustedWingModel
from utils.volatility_fitter.wing_model.wing_model_parameters import WingModelParameters

def create_initial_parameters(forward_price: float, time_to_expiry: float, 
                            atm_volatility: float) -> WingModelParameters:
    """Create initial wing model parameters with market-based estimates"""
    
    return WingModelParameters(
        # Core parameters
        vr=atm_volatility,              # At-the-money volatility
        sr=0.0,                         # Slope (symmetric smile initially)
        pc=0.1,                         # Put curvature
        cc=0.1,                         # Call curvature
        
        # Cutoff points
        dc=-1.0,                        # Down cutoff (negative moneyness)
        uc=1.0,                         # Up cutoff (positive moneyness)
        
        # Smoothing parameters
        dsm=2.0,                        # Down smoothing multiplier
        usm=2.0,                        # Up smoothing multiplier
        
        # Model configuration
        forward_price=forward_price,
        ref_price=forward_price,
        time_to_expiry=time_to_expiry,
        model_name="time_adjusted_wing_model"
    )
```

#### 2.2 Parameter Bounds Configuration
```python
def get_optimization_bounds(config: VolatilityConfig, model_name: str) -> List[Tuple[float, float]]:
    """Retrieve parameter bounds from configuration"""
    bounds_config = config.get_parameter_bounds(model_name)
    
    parameter_bounds = [
        bounds_config['vr'],     # [0.05, 5.0] - volatility reference
        bounds_config['sr'],     # [-2.0, 2.0] - slope reference
        bounds_config['pc'],     # [-1.001, 5.0] - put curvature
        bounds_config['cc'],     # [0.001, 5.0] - call curvature
        bounds_config['dc'],     # [-5.0, 0.0] - down cutoff
        bounds_config['uc'],     # [0.0, 5.0] - up cutoff
        bounds_config['dsm'],    # [0.01, 10.0] - down smoothing
        bounds_config['usm']     # [0.01, 10.0] - up smoothing
    ]
    
    return parameter_bounds
```

### **Step 3: Optimization Strategy Selection**

#### 3.1 Global Optimization (Differential Evolution)
```python
from utils.volatility_fitter.calibrators import GlobalVolatilityCalibrator

def run_global_calibration(initial_params: WingModelParameters, 
                          market_data: pl.DataFrame, 
                          parameter_bounds: List[Tuple[float, float]], 
                          config: VolatilityConfig) -> CalibrationResult:
    """Run global optimization using Differential Evolution"""
    
    # Create global calibrator
    global_calibrator = GlobalVolatilityCalibrator(
        model_class=TimeAdjustedWingModel,
        enable_bounds=True,
        tolerance=config.calibration_tolerance,
        arbitrage_penalty=1e5,
        max_iterations=config.max_calibration_iterations,
        workers=5  # Parallel processing
    )
    
    # Extract market data arrays
    strikes = market_data['strike'].to_list()
    market_vols = market_data['implied_volatility'].to_list()
    market_vegas = market_data['vega'].to_list()
    weights = market_data['weight'].to_list()
    
    # Run Differential Evolution
    result = global_calibrator.calibrate_with_differential_evolution(
        initial_params=initial_params,
        strikes=strikes,
        market_volatilities=market_vols,
        market_vegas=market_vegas,
        parameter_bounds=parameter_bounds,
        enforce_arbitrage_free=config.enforce_arbitrage_free,
        popsize=15,          # Population size multiplier
        maxiter=1000,        # Maximum generations
        seed=42,             # Reproducibility
        weights=weights
    )
    
    return result
```

#### 3.2 Local Optimization (Gradient-Based)
```python
from utils.volatility_fitter.calibrators import LocalVolatilityCalibrator

def run_local_calibration(initial_params: WingModelParameters, 
                         market_data: pl.DataFrame, 
                         parameter_bounds: List[Tuple[float, float]], 
                         config: VolatilityConfig) -> CalibrationResult:
    """Run local optimization using gradient-based methods"""
    
    # Create local calibrator
    local_calibrator = LocalVolatilityCalibrator(
        model_class=TimeAdjustedWingModel,
        method="SLSQP",      # Sequential Least Squares Programming
        enable_bounds=True,
        tolerance=config.calibration_tolerance,
        arbitrage_penalty=1e5,
        max_iterations=config.max_calibration_iterations
    )
    
    # Extract market data
    strikes = market_data['strike'].to_list()
    market_vols = market_data['implied_volatility'].to_list()
    market_vegas = market_data['vega'].to_list()
    weights = market_data['weight'].to_list()
    
    # Run SLSQP optimization
    result = local_calibrator.calibrate(
        initial_params=initial_params,
        strikes=strikes,
        market_volatilities=market_vols,
        market_vegas=market_vegas,
        parameter_bounds=parameter_bounds,
        enforce_arbitrage_free=config.enforce_arbitrage_free,
        weights=weights
    )
    
    return result
```

### **Step 4: Objective Function and Weighting**

#### 4.1 Weighted Least Squares Objective
```python
def calculate_objective_function(parameters: WingModelParameters, 
                               strikes: List[float], 
                               market_vols: List[float], 
                               market_vegas: List[float], 
                               weights: List[float]) -> float:
    """Calculate weighted sum of squared errors between model and market"""
    
    # Create model instance
    model = TimeAdjustedWingModel(parameters)
    
    # Calculate model volatilities
    model_vols = [model.calculate_volatility_from_strike(strike) for strike in strikes]
    
    # Calculate weighted errors
    squared_errors = []
    for i, (market_vol, model_vol, vega, weight) in enumerate(
        zip(market_vols, model_vols, market_vegas, weights)
    ):
        error = (model_vol - market_vol) ** 2
        weighted_error = error * vega * weight  # Vega and custom weighting
        squared_errors.append(weighted_error)
    
    return sum(squared_errors)
```

#### 4.2 Arbitrage Penalty Integration
```python
def calculate_arbitrage_penalty(model: TimeAdjustedWingModel, penalty_factor: float = 1e5) -> float:
    """Calculate penalty for arbitrage violations using Durrleman condition"""
    
    # Get Durrleman condition values
    log_moneyness, g_values = model.calculate_durrleman_condition(num_points=501)
    
    # Count violations (g_k should be positive for no arbitrage)
    violations = g_values[g_values < 0]
    
    if len(violations) > 0:
        # Penalty proportional to severity and number of violations
        penalty = penalty_factor * (abs(violations).sum() + len(violations))
        return penalty
    
    return 0.0
```

### **Step 5: Model Validation and Quality Control**

#### 5.1 Durrleman Condition Validation
```python
def validate_arbitrage_free_conditions(model: TimeAdjustedWingModel, 
                                     config: VolatilityConfig) -> Dict[str, Any]:
    """Comprehensive arbitrage validation using Durrleman condition"""
    
    # Calculate Durrleman condition
    log_moneyness, g_values = model.calculate_durrleman_condition(
        num_points=config.validation.arbitrage.durrleman.num_points
    )
    
    # Analysis metrics
    min_g_value = g_values.min()
    violation_count = (g_values < config.validation.arbitrage.durrleman.min_g_value).sum()
    violation_percentage = violation_count / len(g_values) * 100
    
    # Strike range analysis
    params = model.parameters
    min_strike = params.forward_price * config.validation.arbitrage.arbitrage_bounds.lower_bound
    max_strike = params.forward_price * config.validation.arbitrage.arbitrage_bounds.upper_bound
    
    return {
        'is_arbitrage_free': violation_count == 0,
        'min_g_value': float(min_g_value),
        'violation_count': int(violation_count),
        'violation_percentage': float(violation_percentage),
        'total_points': len(g_values),
        'strike_range': (min_strike, max_strike),
        'assessment': 'PASS' if violation_count == 0 else 'FAIL'
    }
```

#### 5.2 Calibration Quality Metrics
```python
def assess_calibration_quality(calibration_result: CalibrationResult, 
                             market_data: pl.DataFrame, 
                             config: VolatilityConfig) -> Dict[str, Any]:
    """Evaluate calibration quality using statistical metrics"""
    
    model = TimeAdjustedWingModel(calibration_result.parameters)
    
    # Calculate model volatilities
    strikes = market_data['strike'].to_list()
    market_vols = market_data['implied_volatility'].to_list()
    model_vols = [model.calculate_volatility_from_strike(strike) for strike in strikes]
    
    # Statistical metrics
    residuals = np.array(model_vols) - np.array(market_vols)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))
    
    # R-squared calculation
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((market_vols - np.mean(market_vols)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Quality assessment
    quality_pass = (
        rmse <= config.validation.quality.max_rmse and
        r_squared >= config.validation.quality.min_r_squared
    )
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'max_error': float(max_error),
        'r_squared': float(r_squared),
        'num_points': len(strikes),
        'quality_pass': quality_pass,
        'optimization_error': float(calibration_result.error),
        'optimization_time': getattr(calibration_result, 'time_elapsed', 0.0)
    }
```

### **Step 6: Volatility Surface Generation**

#### 6.1 Complete Surface Construction
```python
def generate_volatility_surface(model: TimeAdjustedWingModel, 
                              config: VolatilityConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Generate complete volatility surface across strike range"""
    
    params = model.parameters
    
    # Define strike range
    min_strike = params.forward_price * config.output.surface.strike_range[0]
    max_strike = params.forward_price * config.output.surface.strike_range[1]
    
    # Generate strike array
    strikes = np.linspace(min_strike, max_strike, config.output.surface.num_strikes)
    
    # Calculate volatilities
    volatilities = np.array([
        model.calculate_volatility_from_strike(strike) for strike in strikes
    ])
    
    return strikes, volatilities
```

#### 6.2 Regional Analysis and Boundaries
```python
def analyze_volatility_regions(model: TimeAdjustedWingModel) -> Dict[str, Any]:
    """Analyze different regions of the volatility surface"""
    
    params = model.parameters
    
    # Calculate region boundaries in strike space
    def moneyness_to_strike(moneyness: float) -> float:
        """Convert model moneyness back to strike price"""
        # This requires solving the inverse of the moneyness calculation
        # Approximate solution for practical purposes
        return params.forward_price * np.exp(moneyness / model.get_normalization_term(params.time_to_expiry))
    
    region_strikes = {
        'far_otm_put': moneyness_to_strike(params.dc * (1 + params.dsm)),
        'down_smoothing_start': moneyness_to_strike(params.dc * (1 + params.dsm)),
        'down_cutoff': moneyness_to_strike(params.dc),
        'atm': params.forward_price,
        'up_cutoff': moneyness_to_strike(params.uc),
        'up_smoothing_end': moneyness_to_strike(params.uc * (1 + params.usm)),
        'far_otm_call': moneyness_to_strike(params.uc * (1 + params.usm))
    }
    
    return region_strikes
```

### **Step 7: Results Processing and Output**

#### 7.1 Calibration Results Summary
```python
def create_calibration_summary(calibration_result: CalibrationResult, 
                             quality_metrics: Dict[str, Any], 
                             arbitrage_validation: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive summary of calibration results"""
    
    params = calibration_result.parameters
    
    return {
        'calibration_status': {
            'success': calibration_result.success,
            'optimization_method': calibration_result.optimization_method,
            'optimization_error': calibration_result.error,
            'optimization_time': getattr(calibration_result, 'time_elapsed', 0.0),
            'message': calibration_result.message
        },
        
        'model_parameters': {
            'atm_volatility': params.vr,
            'slope': params.sr,
            'put_curvature': params.pc,
            'call_curvature': params.cc,
            'down_cutoff': params.dc,
            'up_cutoff': params.uc,
            'down_smoothing': params.dsm,
            'up_smoothing': params.usm,
            'forward_price': params.forward_price,
            'time_to_expiry': params.time_to_expiry
        },
        
        'quality_metrics': quality_metrics,
        'arbitrage_validation': arbitrage_validation,
        
        'timestamp': datetime.now().isoformat(),
        'model_type': 'TimeAdjustedWingModel'
    }
```
## 🎛️ Optimization Strategies

### **Global vs Local Optimization**

#### **Differential Evolution (Global)**
- **Best for**: Complex parameter spaces, unknown starting points
- **Population size**: 15 × number of parameters
- **Generations**: 1000+ for thorough search
- **Parallel processing**: Multi-core optimization support
- **Robust but computationally expensive**

#### **SLSQP (Local)**
- **Best for**: Good initial estimates, fast convergence needed
- **Gradient-based**: Efficient for smooth objective functions
- **Bounded constraints**: Supports parameter bounds and constraints
- **Fast but may get trapped in local minima**

# Run Both optimization in parallel
 - the global optimizer would take long time to run than the local optimizer
 - local optimizer can run every 1s and the global optimzer can run every 5s
 - if the error between both result does not deviate more than the threshold, stick with local optimizer or else use the global optimizer's output