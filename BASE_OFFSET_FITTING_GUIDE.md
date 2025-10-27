# Base Offset Fitting: Comprehensive Step-by-Step Guide

## Overview

Base Offset Fitting is a quantitative finance methodology for extracting implied forward pricing and interest rate information from Bitcoin options data using put-call parity regression analysis. This approach enables the discovery of market-implied USD interest rates and BTC funding rates from options market data.

---

## 📚 Theoretical Foundation

### Put-Call Parity Relationship
The fundamental equation governing options pricing relationships:

```
C - P = F × e^(-r×τ) - K × e^(-r×τ)
```

Where:
- `C` = Call option price
- `P` = Put option price  
- `F` = Forward price of underlying asset
- `K` = Strike price
- `r` = Risk-free interest rate
- `τ` = Time to expiration

### Linearization for Regression
Rearranging for put-call difference:

```
P - C = K × e^(-r×τ) - S × e^(-q×τ)
```

This becomes a linear regression problem:
```
y = a×K + b
```

Where:
- `y = P - C` (put-call price difference)
- `a = e^(-r×τ)` (coefficient related to USD interest rate)
- `b = -S × e^(-q×τ)` (constant related to BTC funding rate)

---

## 🔧 Step-by-Step Implementation Process

### **Step 1: Option Chain Construction**

#### 1.1  Spread Calculation and Conflation
```python
# Calculate bid-ask spreads and mid prices
for timestamp in timestamps:
    option_chain = symbol_manager.get_option_chain_at_timestamp(
        expiry=expiry, 
        timestamp=timestamp
    )
```

**Key calculations:**
- Mid prices: `(bid + ask) / 2`
- Absolute spreads: `ask - bid`
- Relative spreads: `spread / mid_price`
- Data quality filtering based on spread thresholds

### **Step 2: Spread Tightening and Option Constraint Application**

#### 2.1 Purpose and Economic Rationale
Before creating put-call parity synthetics, raw market data must be "tightened" to ensure:
- **No-arbitrage consistency**: Eliminates impossible price relationships
- **Monotonicity preservation**: Maintains economic ordering across strikes
- **Data quality enhancement**: Removes pricing inconsistencies that would distort rate extraction

#### 2.2 Constraint Categories

**A. Monotonicity Constraints**
```python
# Call options: Values decrease with strike (decreasing intrinsic value)
for i in range(len(strikes) - 2, -1, -1):
    if volume[i+1] >= volume_threshold:
        call_bid[i] = max(call_bid[i], call_bid[i+1])

# Put options: Values increase with strike (increasing intrinsic value)  
for i in range(1, len(strikes)):
    if volume[i-1] >= volume_threshold:
        put_bid[i] = max(put_bid[i], put_bid[i-1])
```

**B. No-Arbitrage Bounds**
```python
# Adjacent strike constraints
for i in range(1, len(strikes)):
    strike_diff = (strike[i] - strike[i-1]) / spot_price
    strike_diff *= exp(-interest_rate * time_to_expiry)  # Present value
    
    # Call spread bounds: |C(K1) - C(K2)| <= PV(K2 - K1)
    if volume[i-1] >= volume_threshold:
        call_bid[i] = max(call_bid[i], call_bid[i-1] - strike_diff)
    
    # Put spread bounds: Similar constraints for puts
    if volume[i-1] >= volume_threshold:
        put_ask[i] = min(put_ask[i], put_ask[i-1] + strike_diff)
```

#### 2.3 Volume-Based Conditional Application
```python
def apply_tightening_with_volume_filter(prices, volumes, volume_threshold):
    """Apply constraints only where sufficient liquidity exists"""
    for i, volume in enumerate(volumes):
        if volume >= volume_threshold:
            # Apply constraint at this strike
            prices[i] = apply_constraint(prices[i])
        else:
            # Preserve original price for low-volume strikes
            prices[i] = original_prices[i]
```

**Benefits:**
- **Liquidity focus**: Only modifies well-traded strikes with real market information
- **Preserve sparse data**: Maintains original quotes where volume is thin
- **Quality control**: Prevents distortion from outlier small trades


### **Step 3: Put-Call Parity Synthetic Construction**

#### 3.1 Synthetic Data Generation
```python
# Create put-call parity synthetic for each strike
synthetic_data = []
for strike in strikes:
    call_mid = option_chain.get_call_mid(strike)
    put_mid = option_chain.get_put_mid(strike)
    
    # Put-call difference (dependent variable)
    mid = put_mid - call_mid
    
    # Combined spread (for weighting)
    spread = call_spread + put_spread
    
    synthetic_data.append({
        'strike': strike,
        'mid': mid,           # y-variable: P - C
        'spread': spread,     # for inverse variance weighting
        'S': spot_price,      # current spot price
        'tau': time_to_expiry # time to expiration in years
    })
```

**Critical aspects:**
- Each synthetic point represents put-call difference for one strike
- Spread combination accounts for uncertainty propagation
- Only strikes with both call and put quotes are included

### **Step 4: Constrained Optimization Enhancement**

#### 4.1 Nonlinear Optimization Setup
```python
# Enhanced fitting with constraints and regularization
def objective(params, x, y, weight, past_params, lambda_reg, spot, tau):
    const, coef = params
    
    # Primary objective: weighted sum of squared errors
    residuals = y - (const + coef * x)
    sse = np.sum(weight * residuals**2)
    
    # Regularization penalty (temporal smoothing)
    penalty = lambda_reg * np.sum((current_rates - past_rates)**2)
    
    return sse + penalty
```

#### 4.2 Constraint Implementation
```python
def build_constraints(spot, tau, future_bid, future_ask, has_futures):
    """Build optimization constraints"""
    constraints = [
        # Rate bounds: reasonable interest rate ranges
        {'type': 'ineq', 'fun': lambda p: -log(p[1]) - r_min * tau},
        {'type': 'ineq', 'fun': lambda p: log(p[1]) + r_max * tau},
        
        # Funding rate bounds
        {'type': 'ineq', 'fun': lambda p: -log(-p[0] / spot) - q_min * tau},
        {'type': 'ineq', 'fun': lambda p: log(-p[0] / spot) + q_max * tau},
    ]
    
    # Add futures constraints if available
    if has_futures:
        forward_bounds = get_future_bounds(future_bid, future_ask, multiplier)
        constraints.extend(forward_bound_constraints)
    
    return constraints
```

**Financial interpretation:**
- **Coefficient (a)**: `e^(-r×τ)` → USD interest rate extraction
- **Constant (b)**: `-S × e^(-q×τ)` → BTC funding rate extraction

**Constraint benefits:**
- **Economic realism**: Prevents unrealistic interest rate estimates
- **Temporal stability**: Regularization ensures smooth rate evolution
- **Futures consistency**: When available, aligns with futures market pricing

### **Step 5: Quality Assessment and Validation**

#### 5.1 Statistical Metrics
```python
# Model quality assessment
r_squared = model.rsquared_adj              # Goodness of fit
sse = model.ssr                             # Sum of squared errors
parameter_significance = model.pvalues      # Statistical significance
```

#### 5.2 Economic Validation
```python
# Check for economic reasonableness
def validate_results(result):
    checks = {
        'rate_bounds': r_min < result.r < r_max,
        'funding_bounds': q_min < result.q < q_max,
        'fit_quality': result.r2 > min_r_squared,
        'sufficient_data': len(strikes) >= min_strikes
    }
    return all(checks.values())
```

#### 6 Exponential Moving Average Implementation

### **Exponential Moving Average Smoothing of the final interest rate spread (r-q) **
```python
def average_basis_smoothening(self, old_basis: float, new_basis: float) -> float:
    """Apply exponential moving average to smoothen basis spread."""
    # Default weight favors historical stability (95% old, 5% new)
    old_weight = 0.95  # Configurable smoothing parameter
    return (1 - old_weight) * new_basis + old_weight * old_basis

class FitterResultManager:
    def organize_results_by_expiry(self) -> dict[str, list[Result]]:
        """Organize results and apply temporal smoothing by expiry."""
        fit_results = {expiry: [] for expiry in self.opt_expiries}
        
        for result in self.fit_results:
            expiry = result['expiry']
            current_basis = result['r'] - result['q']  # Raw basis spread
            
            # Get last smoothed value for this expiry
            last_smoothed = (fit_results[expiry][-1]['smoothened_r-q'] 
                           if len(fit_results[expiry]) > 0 else current_basis)
            
            # Apply smoothing only for successful fits
            smoothed_basis = (self.average_basis_smoothening(last_smoothed, current_basis)
                            if result['success_fitting'] else last_smoothed)
            
            # Store both raw and smoothed values
            result.update({
                'r-q': current_basis,           # Raw basis spread
                'smoothened_r-q': smoothed_basis  # Smoothed basis spread
            })
            
            fit_results[expiry].append(result)
        return fit_results
```

#### 6.1 Smoothing Parameters and Tuning
```python
# Smoothing weight selection
old_weight_options = {
    'aggressive': 0.80,    # More responsive to new data (20% new information)
    'moderate': 0.90,      # Balanced approach (10% new information)  
    'conservative': 0.95,  # Stable, slow adaptation (5% new information)
    'very_stable': 0.98    # Maximum stability (2% new information)
}