# Volatility Model Architecture Documentation

## Overview

This document describes the new unified architecture for volatility models in the Baseoffset-Fitting-Manager project. The architecture introduces an abstract base class pattern that allows for consistent implementation of multiple volatility models with a single, unified calibrator.

## Architecture Components

### 1. BaseVolatilityModel (Abstract Base Class)

Located in: `utils/volatility_fitter/base_volatility_model.py`

**Purpose**: Provides a common interface for all volatility models, ensuring consistency and extensibility.

**Key Abstract Methods**:
- `calculate_volatility_from_strike(strike_price: float) -> float`
- `calculate_volatility_from_moneyness(moneyness: float) -> float`
- `calculate_durrleman_condition(num_points: int) -> Tuple[np.ndarray, np.ndarray]`
- `get_strike_ranges() -> Dict[str, Any]`

**Common Methods**:
- `generate_volatility_surface(strike_range, num_strikes) -> Tuple[np.ndarray, np.ndarray]`

### 2. UnifiedVolatilityCalibrator

Located in: `utils/volatility_fitter/base_volatility_model.py`

**Purpose**: A single calibrator that can work with any volatility model implementing `BaseVolatilityModel`.

**Key Features**:
- **Model-agnostic**: Works with any model inheriting from `BaseVolatilityModel`
- **Unified interface**: Consistent calibration API across all models
- **Advanced optimization**: Multiple random starts, arbitrage enforcement, flexible bounds
- **Error handling**: Robust exception handling and meaningful error messages

**Usage Example**:
```python
from utils.volatility_fitter import UnifiedVolatilityCalibrator, TimeAdjustedWingModel

# Create calibrator for any model
calibrator = UnifiedVolatilityCalibrator(
    model_class=TimeAdjustedWingModel,
    enable_bounds=True,
    tolerance=1e-8,
    arbitrage_penalty=1e6
)

# Calibrate with consistent interface
result = calibrator.calibrate(
    initial_params=initial_params,
    strikes=strikes,
    market_volatilities=market_vols,
    market_vegas=market_vegas,
    parameter_bounds=bounds,
    enforce_arbitrage_free=True
)
```

### 3. Model Implementations

#### TimeAdjustedWingModel
Located in: `utils/volatility_fitter/time_adjusted_wing_model/time_adjusted_wing_model.py`

**Inheritance**: `TimeAdjustedWingModel(BaseVolatilityModel)`

**Key Features**:
- Time-adjusted moneyness calculation with normalization term
- 6-region volatility smile modeling (central parabolas + smoothing wings + flat extrapolation)
- Durrleman arbitrage condition checking

#### WingModel  
Located in: `utils/volatility_fitter/wing_model/wing_model.py`

**Inheritance**: `WingModel(BaseVolatilityModel)`

**Key Features**:
- Traditional log forward moneyness
- 6-region volatility smile modeling with reference price adjustments
- Butterfly arbitrage condition checking

### 4. Parameter Classes

Both models use dataclasses for parameter management with required methods:

**Required Methods**:
- `get_parameter_names() -> list[str]`: Returns names of fitted parameters
- `get_fitted_vol_parameter() -> list[float]`: Returns values of fitted parameters

**TimeAdjustedWingModelParameters**:
```python
@dataclass
class TimeAdjustedWingModelParameters:
    atm_vol: float
    slope: float
    call_curve: float
    put_curve: float
    up_cutoff: float
    down_cutoff: float
    up_smoothing: float
    down_smoothing: float
    forward_price: float
    time_to_expiry: float
```

**WingModelParameters**:
```python
@dataclass  
class WingModelParameters:
    vr: float  # volatility reference
    sr: float  # slope reference
    pc: float  # put curvature
    cc: float  # call curvature
    dc: float  # down cutoff
    uc: float  # up cutoff
    dsm: float # down smoothing
    usm: float # up smoothing
    # ... additional market context parameters
```

## Backward Compatibility

The existing model-specific calibrators (`TimeAdjustedWingModelCalibrator`, `WingModelCalibrator`) are maintained for backward compatibility. They now use the `UnifiedVolatilityCalibrator` internally while preserving their original interfaces.

**Example**: `TimeAdjustedWingModelCalibrator` maintains its original `calibrate()` method signature but delegates to `UnifiedVolatilityCalibrator` internally.

## Adding New Models

To add a new volatility model:

1. **Create Model Class**: Inherit from `BaseVolatilityModel`
```python
class NewVolatilityModel(BaseVolatilityModel):
    def __init__(self, parameters: NewModelParameters):
        super().__init__(parameters)
        
    def calculate_volatility_from_strike(self, strike_price: float) -> float:
        # Implementation
        pass
        
    def calculate_volatility_from_moneyness(self, moneyness: float) -> float:
        # Implementation  
        pass
        
    def calculate_durrleman_condition(self, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        # Implementation
        pass
        
    def get_strike_ranges(self) -> Dict[str, Any]:
        # Implementation
        pass
```

2. **Create Parameter Class**: Implement required methods
```python
@dataclass
class NewModelParameters:
    param1: float
    param2: float
    # ... other parameters
    
    def get_parameter_names(self) -> list[str]:
        return ['param1', 'param2']  # fitted parameters only
        
    def get_fitted_vol_parameter(self) -> list[float]:
        return [self.param1, self.param2]
```

3. **Use UnifiedVolatilityCalibrator**: No additional calibrator needed
```python
calibrator = UnifiedVolatilityCalibrator(model_class=NewVolatilityModel)
result = calibrator.calibrate(...)
```

## Benefits of New Architecture

1. **Consistency**: All models follow the same interface pattern
2. **Extensibility**: Easy to add new models without duplicating calibration logic
3. **Maintainability**: Single calibrator reduces code duplication and bugs
4. **Flexibility**: Unified calibrator supports advanced features for all models
5. **Performance**: Optimized calibration logic shared across all models
6. **Testing**: Common test patterns can be applied to all models

## Migration Guide

### For Existing Code
- **No changes required**: Existing model-specific calibrators continue to work
- **Gradual migration**: Can migrate to `UnifiedVolatilityCalibrator` when convenient

### For New Development
- **Use UnifiedVolatilityCalibrator**: Preferred for new calibration code
- **Follow BaseVolatilityModel pattern**: For new model implementations
- **Leverage common functionality**: Use shared methods from base class

## Performance Considerations

- **Unified calibrator** uses optimized objective functions and constraint handling
- **Multiple random starts** available for avoiding local minima
- **Efficient arbitrage checking** with configurable penalties
- **Robust parameter bounds** with automatic validation

## Future Enhancements

The architecture is designed to support:
- **Additional volatility models** (SABR, Heston, SVI, etc.)
- **Model ensemble methods** for combining multiple models
- **Advanced calibration strategies** (Bayesian optimization, machine learning)
- **Real-time calibration** with streaming market data
- **Model comparison frameworks** for systematic evaluation

## Example Usage

See `unified_calibrator_example.py` for comprehensive examples demonstrating:
- Basic calibration with both models
- Performance comparison between models
- Advanced calibration options
- Error handling and validation