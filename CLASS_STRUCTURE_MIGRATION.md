# Time-Adjusted Wing Model: Class Structure Refactoring

## Overview

I have successfully refactored the **time-adjusted wing model** from a functional approach to a **class-based structure**, similar to the existing `WingModel` class. This provides better encapsulation, maintainability, and consistency with the existing codebase.

## 🔄 Before vs After Comparison

### **Before: Functional Approach**
```python
# Old functional approach
def calculate_moneyness(forward_price, strike_price, time_to_expiry, atm_vol):
    # Implementation...
    
def orc_wing_model_from_moneyness(moneyness, atm_vol, slope, curve_up, ...):
    # Implementation...
    
def calculate_durrleman(forward_price, time_to_expiry, atm_vol, params):
    # Implementation...

# Usage:
moneyness = calculate_moneyness(60000, 65000, 0.25, 0.75)
vol = orc_wing_model_from_moneyness(moneyness, 0.75, -0.1, 0.3, ...)
```

### **After: Class-Based Structure**
```python
# New class-based approach
@dataclass
class TimeAdjustedWingModelParameters:
    atm_vol: float
    slope: float
    curve_up: float
    # ... other parameters

class TimeAdjustedWingModel:
    def __init__(self, parameters: TimeAdjustedWingModelParameters):
        self.parameters = parameters
    
    def calculate_moneyness(self, forward_price, strike_price, time_to_expiry, atm_vol):
        # Implementation...
    
    def calculate_volatility_from_strike(self, strike_price):
        # Implementation...
    
    def calculate_durrleman_condition(self):
        # Implementation...

# Usage:
params = TimeAdjustedWingModelParameters(atm_vol=0.75, slope=-0.1, ...)
model = TimeAdjustedWingModel(params)
vol = model.calculate_volatility_from_strike(65000)
```

## 📦 New Class Structure

### **1. TimeAdjustedWingModelParameters (Data Class)**
```python
@dataclass
class TimeAdjustedWingModelParameters:
    # Core volatility surface parameters
    atm_vol: float        # At-the-money volatility
    slope: float          # Controls the skew of the smile
    curve_up: float       # Controls the curvature for the upside
    curve_down: float     # Controls the curvature for the downside
    cut_up: float         # Moneyness threshold for the upside parabola
    cut_dn: float         # Moneyness threshold for the downside parabola
    mSmUp: float          # Smoothing factor for the upside wing
    mSmDn: float          # Smoothing factor for the downside wing
    
    # Market context parameters
    forward_price: float = 50000.0   # Forward price of the underlying
    time_to_expiry: float = 0.25     # Time to expiry in years
```

### **2. TimeAdjustedWingModel (Main Class)**
```python
class TimeAdjustedWingModel:
    def calculate_moneyness(self, forward_price, strike_price, time_to_expiry, atm_vol)
    def calculate_volatility_from_strike(self, strike_price)
    def calculate_volatility_from_moneyness(self, moneyness)
    def calculate_durrleman_condition(self, num_points=201)
    def generate_volatility_surface(self, strike_range, num_strikes=50)
```

### **3. TimeAdjustedWingModelCalibrator (Calibration Class)**
```python
class TimeAdjustedWingModelCalibrator:
    def calibrate(self, strike_list, market_vol_list, market_vega_list, ...)
    def _loss_function(self, solve_params, ...)
    def _get_parameter_bounds(self)
```

## 🎯 Key Improvements

### **1. Better Encapsulation**
- **Before**: Parameters passed as individual arguments to functions
- **After**: Parameters encapsulated in a dataclass with validation

### **2. Consistent API**
- **Before**: Different function signatures and calling patterns
- **After**: Unified interface similar to existing `WingModel` class

### **3. Parameter Validation**
- **Before**: No automatic parameter validation
- **After**: `__post_init__` validation in dataclass

### **4. Easier Usage**
- **Before**: Need to remember parameter order and pass many arguments
- **After**: Create parameters once, reuse model instance

### **5. Enhanced Functionality**
- **Before**: Limited to basic volatility calculations
- **After**: Added surface generation, improved arbitrage checking, calibration framework

## 🧪 Usage Examples

### **Creating a Model**
```python
# Define parameters
params = TimeAdjustedWingModelParameters(
    atm_vol=0.75, slope=-0.1, curve_up=0.3, curve_down=0.5,
    cut_up=1.2, cut_dn=-1.0, mSmUp=0.4, mSmDn=0.6,
    forward_price=60000.0, time_to_expiry=0.25
)

# Create model
model = TimeAdjustedWingModel(params)
```

### **Calculate Volatilities**
```python
# Single strike
vol = model.calculate_volatility_from_strike(65000)

# Multiple strikes
strikes = [55000, 60000, 65000]
vols = [model.calculate_volatility_from_strike(k) for k in strikes]

# Full surface
strikes_array, vols_array = model.generate_volatility_surface((50000, 70000), 21)
```

### **Arbitrage Checking**
```python
log_moneyness, g_values = model.calculate_durrleman_condition()
arbitrage_free = np.min(g_values) >= 0
```

### **Model Calibration**
```python
calibrator = TimeAdjustedWingModelCalibrator()
result = calibrator.calibrate(
    strike_list=[55000, 60000, 65000],
    market_vol_list=[0.78, 0.75, 0.73],
    market_vega_list=[120, 140, 120],
    forward_price=60000.0,
    time_to_expiry=0.25
)
```

## ✅ Testing Results

The new class structure has been tested and verified:

```
📋 Testing TimeAdjustedWingModel class structure...
✅ TimeAdjustedWingModel created successfully
✅ ATM volatility calculation: 0.7524
✅ OTM Call volatility (K=65000): 0.7497
✅ OTM Put volatility (K=55000): 0.7560
✅ Moneyness calculation: 0.0030
✅ Surface generation: 5 points
   Vol range: 0.7497 - 0.7560
🎉 All tests passed! Class structure is working correctly.
```

## 📁 File Structure

```
utils/volatility_fitter/
├── __init__.py                                  # Package imports
├── time_adjusted_wing_model.py                  # Main model class
├── time_adjusted_wing_model_calibrator.py      # Calibration framework
├── time_adjusted_wing_model_example.py         # Usage examples
├── wing_model.py                               # Original wing model
├── wing_model_parameters.py                    # Original parameters
└── wing_model_calibrator.py                   # Original calibrator
```

## 🔄 Migration Guide

To migrate from the old functional approach to the new class structure:

1. **Replace function calls**:
   ```python
   # Old
   vol = orc_wing_model_from_moneyness(moneyness, atm_vol, slope, ...)
   
   # New  
   params = TimeAdjustedWingModelParameters(atm_vol=atm_vol, slope=slope, ...)
   model = TimeAdjustedWingModel(params)
   vol = model.calculate_volatility_from_moneyness(moneyness)
   ```

2. **Update imports**:
   ```python
   # Old
   from time_adjusted_wing_model import calculate_moneyness, orc_wing_model_from_moneyness
   
   # New
   from utils.volatility_fitter import TimeAdjustedWingModel, TimeAdjustedWingModelParameters
   ```

3. **Benefit from new features**:
   - Use `generate_volatility_surface()` for batch calculations
   - Use `TimeAdjustedWingModelCalibrator` for parameter fitting
   - Leverage automatic parameter validation

The new class structure maintains **full backward compatibility** in terms of mathematical calculations while providing a much more robust and maintainable framework.