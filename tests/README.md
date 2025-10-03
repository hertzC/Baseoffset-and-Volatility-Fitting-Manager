# Bitcoin Options Analysis - Unit Tests

This directory contains comprehensive unit tests for the Bitcoin Options Analysis pipeline, ensuring that synthetic option creation and fitting functionality produces consistent results across code changes.

## Test Structure

### 📂 Test Files

- **`test_bitcoin_options_analysis.py`** - Main test suite with multiple test classes
- **`run_tests.py`** - Convenient test runner script

### 🧪 Test Categories

#### 1. **TestSyntheticCreation**
Tests the creation of synthetic option data from put-call parity:
- ✅ Validates synthetic data structure and consistency
- ✅ Ensures correct strike prices and time-to-expiry values
- ✅ Verifies positive spreads and realistic pricing

#### 2. **TestWLSRegression** 
Tests Weighted Least Squares regression functionality:
- ✅ Validates parameter recovery with known input data
- ✅ Tests edge cases (empty data, insufficient strikes)
- ✅ Ensures R² values indicate good fits
- ✅ Verifies USD and BTC rate extraction accuracy

#### 3. **TestNonlinearMinimization**
Tests constrained optimization functionality:
- ✅ Validates constrained optimization convergence  
- ✅ Tests parameter management (reset, update, inspection)
- ✅ Ensures fallback handling when optimization fails
- ✅ Verifies results consistency with WLS initial guesses

#### 4. **TestRegressionValues**
Regression tests with fixed baseline values:
- ✅ Tests exact parameter recovery with synthetic data
- ✅ Catches unexpected changes in algorithm behavior
- ✅ Validates mathematical consistency across updates

## 📊 Test Data Generation

### Realistic Synthetic Data
The tests use mathematically exact synthetic data based on put-call parity:

```python
# Linear relationship: P - C = const + coef * K
const = -S * exp(-q * tau)  # ~-54,500 for test parameters
coef = exp(-r * tau)        # ~0.9986 for test parameters
```

**Test Parameters:**
- USD Rate (r): 5.0% annually
- BTC Rate (q): 1.0% annually  
- Spot Price (S): $55,000
- Time to Expiry (τ): 0.0274 years (~10 days)

This ensures the fitting algorithms should recover the exact input parameters, making the tests deterministic and reliable.

## 🚀 Running Tests

### Option 1: Use the Test Runner (Recommended)
```bash
python run_tests.py
```

### Option 2: Direct pytest
```bash
# Run all tests
python -m pytest tests/test_bitcoin_options_analysis.py -v

# Run specific test class
python -m pytest tests/test_bitcoin_options_analysis.py::TestWLSRegression -v

# Run specific test method
python -m pytest tests/test_bitcoin_options_analysis.py::TestWLSRegression::test_wls_fitting_consistency -v
```

### Option 3: With Coverage (requires pytest-cov)
```bash
python -m pytest tests/test_bitcoin_options_analysis.py --cov=btc_options --cov-report=html
```

## ✅ Expected Results

When all tests pass, you should see:
```
======================================= test session starts ========================================
...
tests/test_bitcoin_options_analysis.py::TestSyntheticCreation::test_synthetic_creation_consistency PASSED
tests/test_bitcoin_options_analysis.py::TestWLSRegression::test_wls_fitting_consistency PASSED  
tests/test_bitcoin_options_analysis.py::TestWLSRegression::test_wls_parameter_validation PASSED
tests/test_bitcoin_options_analysis.py::TestNonlinearMinimization::test_nonlinear_fitting_consistency PASSED
tests/test_bitcoin_options_analysis.py::TestNonlinearMinimization::test_parameter_management PASSED
tests/test_bitcoin_options_analysis.py::TestNonlinearMinimization::test_results_management PASSED
tests/test_bitcoin_options_analysis.py::TestRegressionValues::test_wls_baseline_values PASSED

================================== 7 passed, X warnings in Y.XXs ==================================
```

## 🛡️ Regression Protection

These tests protect against:

- **Algorithm Changes**: If the fitting logic changes unexpectedly, tests will catch differences in results
- **Parameter Bugs**: Tests verify exact recovery of known input parameters
- **Edge Cases**: Tests ensure graceful handling of insufficient data and optimization failures
- **Interface Changes**: Tests validate that the API contracts remain stable

## 🔧 Maintaining Tests

### When to Update Test Values

**✅ Update tests when:**
- You intentionally change the fitting algorithm
- You modify the mathematical formulas (e.g., Black-Scholes to Black-76)
- You change parameter ranges or constraints

**❌ Don't update tests for:**
- Minor code refactoring that shouldn't change results
- Documentation updates
- UI/visualization changes

### Updating Expected Values

If you make intentional changes to the algorithms, update the expected ranges in the test assertions:

```python
# In TestRegressionValues.test_wls_baseline_values()
expected_values = {
    'r': (0.049, 0.051),  # Update these ranges if algorithm changes
    'q': (0.009, 0.011),  # Should be tight for regression testing
    'r2': (0.99, 1.0),    # R-squared should be nearly perfect
    'sse': (0.0, 0.001)   # SSE should be very small
}
```

## 📈 Test Coverage

The test suite covers:

- **Data Generation**: Synthetic option chain creation
- **Core Algorithms**: WLS regression and nonlinear optimization  
- **Parameter Management**: Reset, update, and inspection functionality
- **Error Handling**: Edge cases and optimization failures
- **Mathematical Consistency**: Put-call parity relationships
- **API Contracts**: Expected input/output formats

## 🐛 Debugging Failed Tests

### Common Issues:

1. **Parameter Out of Range**: Check if algorithm changes affected rate calculations
2. **Optimization Failures**: Verify constraint setup and initial guesses
3. **Data Format Changes**: Ensure DataFrame schemas haven't changed
4. **Import Errors**: Check if module paths or dependencies changed

### Debug Commands:
```bash
# Run with more verbose output
python -m pytest tests/test_bitcoin_options_analysis.py -vvv --tb=long

# Run specific failing test with print statements
python -m pytest tests/test_bitcoin_options_analysis.py::TestWLSRegression::test_wls_fitting_consistency -v -s
```

---

**Remember**: These tests are your safety net. If they fail after code changes, investigate carefully before updating the expected values. The goal is to catch unintended changes while allowing for intentional improvements.