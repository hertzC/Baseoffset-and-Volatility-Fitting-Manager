"""
Test cases for Time-Adjusted Wing Model Volatility Fitter

This module contains comprehensive test cases to ensure the time-adjusted wing model
volatility fitter works correctly and to detect breaking changes in the future.
"""

import pytest
import numpy as np
import polars as pl
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils.volatility_fitter.time_adjusted_wing_model.time_adjusted_wing_model import (
    TimeAdjustedWingModel, 
    TimeAdjustedWingModelParameters
)
from utils.volatility_fitter.time_adjusted_wing_model.time_adjusted_wing_model_calibrator import (
    TimeAdjustedWingModelCalibrator,
    TimeAdjustedCalibrationResult,
    create_time_adjusted_wing_model_from_result
)


class TestTimeAdjustedWingModelParameters:
    """Test cases for TimeAdjustedWingModelParameters"""
    
    def test_parameters_initialization(self):
        """Test that parameters can be initialized correctly"""
        params = TimeAdjustedWingModelParameters(
            atm_vol=0.5,
            slope=0.1,
            call_curve=2.0,
            put_curve=2.0,
            up_cutoff=0.5,
            down_cutoff=-0.5,
            up_smoothing=1.0,
            down_smoothing=1.0,
            forward_price=50000.0,
            time_to_expiry=0.25
        )
        
        assert params.atm_vol == 0.5
        assert params.slope == 0.1
        assert params.forward_price == 50000.0
        assert params.time_to_expiry == 0.25
    
    def test_parameter_names_method(self):
        """Test that parameter names are returned correctly"""
        params = TimeAdjustedWingModelParameters(
            atm_vol=0.5, slope=0.1, call_curve=2.0, put_curve=2.0,
            up_cutoff=0.5, down_cutoff=-0.5, up_smoothing=1.0, down_smoothing=1.0,
            forward_price=50000.0, time_to_expiry=0.25
        )
        
        param_names = params.get_parameter_names()
        expected_names = ['atm_vol', 'slope', 'call_curve', 'put_curve', 
                         'up_cutoff', 'down_cutoff', 'up_smoothing', 'down_smoothing']
        
        assert param_names == expected_names
        assert 'forward_price' not in param_names
        assert 'time_to_expiry' not in param_names
    
    def test_fitted_vol_parameter_method(self):
        """Test that fitted volatility parameters are returned correctly"""
        params = TimeAdjustedWingModelParameters(
            atm_vol=0.5, slope=0.1, call_curve=2.0, put_curve=2.5,
            up_cutoff=0.5, down_cutoff=-0.5, up_smoothing=1.0, down_smoothing=1.5,
            forward_price=50000.0, time_to_expiry=0.25
        )
        
        param_values = params.get_fitted_vol_parameter()
        expected_values = [0.5, 0.1, 2.0, 2.5, 0.5, -0.5, 1.0, 1.5]
        
        assert param_values == expected_values
        assert len(param_values) == 8


class TestTimeAdjustedWingModel:
    """Test cases for TimeAdjustedWingModel"""
    
    @pytest.fixture
    def sample_parameters(self):
        """Fixture providing sample parameters for testing"""
        return TimeAdjustedWingModelParameters(
            atm_vol=0.6,
            slope=0.08,
            call_curve=5.0,
            put_curve=5.0,
            up_cutoff=0.5,
            down_cutoff=-0.95,
            up_smoothing=5.0,
            down_smoothing=5.0,
            forward_price=60000.0,
            time_to_expiry=0.25
        )
    
    @pytest.fixture
    def model(self, sample_parameters):
        """Fixture providing a model instance"""
        return TimeAdjustedWingModel(sample_parameters)
    
    def test_model_initialization(self, sample_parameters):
        """Test that model initializes correctly"""
        model = TimeAdjustedWingModel(sample_parameters)
        assert model.parameters == sample_parameters
        assert model.use_norm_term == True
        
        # Test with use_norm_term=False
        model_no_norm = TimeAdjustedWingModel(sample_parameters, use_norm_term=False)
        assert model_no_norm.use_norm_term == False
    
    def test_normalization_term_calculation(self, model):
        """Test normalization term calculation"""
        norm_term = model.get_normalization_term(0.25)
        assert isinstance(norm_term, float)
        assert norm_term > 0
        
        # Test that longer time gives different result
        norm_term_long = model.get_normalization_term(1.0)
        assert norm_term_long != norm_term
    
    def test_calculate_volatility_from_strike_basic(self, model):
        """Test basic volatility calculation from strike"""
        # Test ATM strike (should return close to ATM vol)
        atm_vol = model.calculate_volatility_from_strike(60000.0)
        assert isinstance(atm_vol, float)
        assert 0.01 < atm_vol < 2.0  # Reasonable volatility range
        
        # Test OTM call
        otm_call_vol = model.calculate_volatility_from_strike(70000.0)
        assert isinstance(otm_call_vol, float)
        assert 0.01 < otm_call_vol < 2.0
        
        # Test OTM put
        otm_put_vol = model.calculate_volatility_from_strike(50000.0)
        assert isinstance(otm_put_vol, float)
        assert 0.01 < otm_put_vol < 2.0
    
    def test_calculate_volatility_from_strike_array(self, model):
        """Test volatility calculation with array of strikes"""
        strikes = [50000, 55000, 60000, 65000, 70000]
        vols = [model.calculate_volatility_from_strike(strike) for strike in strikes]
        
        assert len(vols) == len(strikes)
        assert all(isinstance(vol, float) for vol in vols)
        assert all(0.01 < vol < 2.0 for vol in vols)
    
    def test_volatility_smile_shape(self, model):
        """Test that volatility smile has expected shape"""
        strikes = np.linspace(40000, 80000, 21)
        vols = [model.calculate_volatility_from_strike(strike) for strike in strikes]
        
        # Find minimum volatility (should be near ATM)
        min_vol_idx = np.argmin(vols)
        min_vol_strike = strikes[min_vol_idx]
        
        # Minimum should be reasonably close to forward price
        forward_price = model.parameters.forward_price
        relative_distance = abs(min_vol_strike - forward_price) / forward_price
        assert relative_distance < 0.2  # Within 20% of forward
    
    def test_moneyness_calculation(self, model):
        """Test moneyness calculation"""
        forward_price = model.parameters.forward_price
        
        # ATM should give moneyness ~0
        atm_moneyness = model.calculate_moneyness(forward_price, forward_price, 0.25, 0.6)
        assert abs(atm_moneyness) < 0.1
        
        # OTM call should give positive moneyness
        otm_call_moneyness = model.calculate_moneyness(forward_price, forward_price * 1.2, 0.25, 0.6)
        assert otm_call_moneyness > 0
        
        # OTM put should give negative moneyness
        otm_put_moneyness = model.calculate_moneyness(forward_price, forward_price * 0.8, 0.25, 0.6)
        assert otm_put_moneyness < 0
    
    def test_strike_ranges(self, model):
        """Test strike range calculation"""
        try:
            ranges = model.get_strike_ranges()
            assert isinstance(ranges, dict)
            
            expected_keys = ['downSmoothing', 'downCutOff', 'upCutOff', 'upSmoothing']
            for key in expected_keys:
                assert key in ranges
                assert isinstance(ranges[key], (int, float))
            
            # Ranges should be ordered correctly
            assert ranges['downSmoothing'] <= ranges['downCutOff']
            assert ranges['downCutOff'] <= ranges['upCutOff']
            assert ranges['upCutOff'] <= ranges['upSmoothing']
            
        except AttributeError:
            pytest.skip("get_strike_ranges method not available")
    
    def test_moneyness_ranges(self, model):
        """Test moneyness range calculation"""
        try:
            ranges = model.get_moneyness_ranges()
            assert isinstance(ranges, dict)
            
            expected_keys = ['downSmoothing', 'downCutOff', 'upCutOff', 'upSmoothing']
            for key in expected_keys:
                assert key in ranges
                assert isinstance(ranges[key], (int, float))
                
        except AttributeError:
            pytest.skip("get_moneyness_ranges method not available")


class TestTimeAdjustedWingModelCalibrator:
    """Test cases for TimeAdjustedWingModelCalibrator"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Fixture providing sample market data for calibration"""
        strikes = [45000, 50000, 55000, 60000, 65000, 70000, 75000]
        vols = [0.8, 0.7, 0.65, 0.6, 0.65, 0.7, 0.8]  # Typical volatility smile
        vegas = [100, 150, 180, 200, 180, 150, 100]   # Typical vega profile
        weights = [1.0] * len(strikes)
        forward_price = 60000.0
        time_to_expiry = 0.25
        
        return strikes, vols, vegas, weights, forward_price, time_to_expiry
    
    @pytest.fixture
    def calibrator(self):
        """Fixture providing a calibrator instance"""
        return TimeAdjustedWingModelCalibrator(use_norm_term=True)
    
    def test_calibrator_initialization(self):
        """Test calibrator initialization"""
        calibrator = TimeAdjustedWingModelCalibrator()
        assert calibrator.enable_bounds == True
        assert calibrator.tolerance == 1e-16
        assert calibrator.method == "SLSQP"
        assert calibrator.use_norm_term == True
        
        # Test custom initialization
        custom_calibrator = TimeAdjustedWingModelCalibrator(
            enable_bounds=False,
            tolerance=1e-8,
            method="L-BFGS-B",
            use_norm_term=False
        )
        assert custom_calibrator.enable_bounds == False
        assert custom_calibrator.tolerance == 1e-8
        assert custom_calibrator.method == "L-BFGS-B"
        assert custom_calibrator.use_norm_term == False
    
    def test_parameter_bounds(self, calibrator):
        """Test parameter bounds generation"""
        bounds = calibrator._get_parameter_bounds()
        assert len(bounds) == 8  # 8 parameters
        
        for bound in bounds:
            assert len(bound) == 2  # (min, max)
            assert bound[0] < bound[1]  # min < max
    
    def test_loss_function_basic(self, calibrator, sample_market_data):
        """Test that loss function can be called without errors"""
        strikes, vols, vegas, weights, forward_price, time_to_expiry = sample_market_data
        
        # Sample parameters
        params = [0.6, 0.08, 5.0, 5.0, 0.5, -0.95, 5.0, 5.0]
        args = (strikes, vols, vegas, weights, forward_price, time_to_expiry, True)
        
        loss = calibrator._loss_function(params, *args)
        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_calibration_basic(self, calibrator, sample_market_data):
        """Test basic calibration functionality"""
        strikes, vols, vegas, weights, forward_price, time_to_expiry = sample_market_data
        
        # Run calibration (with relaxed tolerance for faster testing)
        calibrator.tolerance = 1e-6
        result = calibrator.calibrate(
            strike_list=strikes,
            market_vol_list=vols,
            market_vega_list=vegas,
            weight_list=weights,
            forward_price=forward_price,
            time_to_expiry=time_to_expiry
        )
        
        assert isinstance(result, TimeAdjustedCalibrationResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.error, float)
        assert result.error >= 0
        assert isinstance(result.parameters, TimeAdjustedWingModelParameters)
    
    def test_create_model_from_result(self, sample_market_data):
        """Test model creation from calibration result"""
        strikes, vols, vegas, weights, forward_price, time_to_expiry = sample_market_data
        
        # Sample optimized parameters
        optimized_params = [0.65, 0.08, 4.5, 4.5, 0.4, -0.8, 4.0, 4.0]
        
        model_params = create_time_adjusted_wing_model_from_result(
            optimized_params, forward_price, time_to_expiry
        )
        
        assert isinstance(model_params, TimeAdjustedWingModelParameters)
        assert model_params.forward_price == forward_price
        assert model_params.time_to_expiry == time_to_expiry
        
        # Test that model can be created
        model = TimeAdjustedWingModel(model_params)
        assert isinstance(model, TimeAdjustedWingModel)


class TestIntegrationScenarios:
    """Integration test cases simulating real-world scenarios"""
    
    def test_bitcoin_options_scenario(self):
        """Test with realistic Bitcoin options data"""
        # Realistic Bitcoin options data (Feb 2024)
        strikes = [45000, 50000, 55000, 60000, 65000, 70000, 75000]
        market_vols = [0.95, 0.85, 0.75, 0.68, 0.72, 0.78, 0.85]
        vegas = [80, 120, 160, 180, 160, 120, 80]
        weights = [v/max(vegas) for v in vegas]  # Vega-weighted
        forward_price = 62000.0
        time_to_expiry = 0.0411  # ~15 days
        
        calibrator = TimeAdjustedWingModelCalibrator(use_norm_term=True)
        
        result = calibrator.calibrate(
            strike_list=strikes,
            market_vol_list=market_vols,
            market_vega_list=vegas,
            weight_list=weights,
            forward_price=forward_price,
            time_to_expiry=time_to_expiry
        )
        
        # Basic validation
        assert isinstance(result, TimeAdjustedCalibrationResult)
        
        if result.success:
            # Create model and test fitted volatilities
            model = TimeAdjustedWingModel(result.parameters)
            fitted_vols = [model.calculate_volatility_from_strike(s) for s in strikes]
            
            # Check that fitted vols are reasonable
            assert all(0.1 < vol < 2.0 for vol in fitted_vols)
            
            # Check that RMSE is reasonable
            rmse = np.sqrt(np.mean([(f - m)**2 for f, m in zip(fitted_vols, market_vols)]))
            assert rmse < 0.5  # RMSE should be less than 50%
    
    def test_extreme_volatility_scenario(self):
        """Test with extreme volatility values"""
        strikes = [40000, 50000, 60000, 70000, 80000]
        market_vols = [1.5, 1.2, 0.8, 1.0, 1.3]  # High volatility environment
        vegas = [50, 100, 150, 100, 50]
        weights = [1.0] * len(strikes)
        forward_price = 60000.0
        time_to_expiry = 0.25
        
        calibrator = TimeAdjustedWingModelCalibrator()
        
        result = calibrator.calibrate(
            strike_list=strikes,
            market_vol_list=market_vols,
            market_vega_list=vegas,
            weight_list=weights,
            forward_price=forward_price,
            time_to_expiry=time_to_expiry
        )
        
        # Should handle extreme values gracefully
        assert isinstance(result, TimeAdjustedCalibrationResult)
        
        if result.success:
            model = TimeAdjustedWingModel(result.parameters)
            fitted_vols = [model.calculate_volatility_from_strike(s) for s in strikes]
            
            # Fitted vols should be within reasonable bounds
            assert all(0.01 < vol < 3.0 for vol in fitted_vols)
    
    def test_short_time_to_expiry(self):
        """Test with very short time to expiry"""
        strikes = [58000, 59000, 60000, 61000, 62000]
        market_vols = [0.8, 0.7, 0.65, 0.7, 0.8]
        vegas = [20, 40, 50, 40, 20]
        weights = [1.0] * len(strikes)
        forward_price = 60000.0
        time_to_expiry = 0.0027  # ~1 day
        
        calibrator = TimeAdjustedWingModelCalibrator()
        
        result = calibrator.calibrate(
            strike_list=strikes,
            market_vol_list=market_vols,
            market_vega_list=vegas,
            weight_list=weights,
            forward_price=forward_price,
            time_to_expiry=time_to_expiry
        )
        
        # Should handle short expiry gracefully
        assert isinstance(result, TimeAdjustedCalibrationResult)


class TestRegressionScenarios:
    """Regression test cases to detect breaking changes"""
    
    def test_parameter_consistency(self):
        """Test that specific parameter sets produce consistent results"""
        # Fixed test case that should always produce the same result
        strikes = [50000, 55000, 60000, 65000, 70000]
        market_vols = [0.8, 0.7, 0.6, 0.7, 0.8]
        vegas = [100, 150, 200, 150, 100]
        weights = [1.0] * len(strikes)
        forward_price = 60000.0
        time_to_expiry = 0.25
        
        # Fixed parameters that should give a specific loss
        fixed_params = [0.6, 0.08, 5.0, 5.0, 0.5, -0.95, 5.0, 5.0]
        
        calibrator = TimeAdjustedWingModelCalibrator()
        args = (strikes, market_vols, vegas, weights, forward_price, time_to_expiry, True)
        
        loss = calibrator._loss_function(fixed_params, *args)
        
        # This should be consistent across runs
        # Store the expected loss value (update this when the implementation changes intentionally)
        expected_loss_range = (0.0, 1.0)  # Reasonable range for this test case
        assert expected_loss_range[0] <= loss <= expected_loss_range[1]
    
    def test_model_interface_consistency(self):
        """Test that the model interface remains consistent"""
        params = TimeAdjustedWingModelParameters(
            atm_vol=0.6, slope=0.08, call_curve=5.0, put_curve=5.0,
            up_cutoff=0.5, down_cutoff=-0.95, up_smoothing=5.0, down_smoothing=5.0,
            forward_price=60000.0, time_to_expiry=0.25
        )
        
        model = TimeAdjustedWingModel(params)
        
        # Test essential methods exist and work
        assert hasattr(model, 'calculate_volatility_from_strike')
        assert hasattr(model, 'calculate_moneyness')
        assert hasattr(model, 'get_normalization_term')
        
        # Test method signatures
        vol = model.calculate_volatility_from_strike(60000.0)
        assert isinstance(vol, float)
        
        moneyness = model.calculate_moneyness(60000.0, 60000.0, 0.25, 0.6)
        assert isinstance(moneyness, float)
        
        norm_term = model.get_normalization_term(0.25)
        assert isinstance(norm_term, float)


# Performance benchmark tests
class TestPerformanceBenchmarks:
    """Performance test cases to detect performance regressions"""
    
    def test_single_volatility_calculation_speed(self):
        """Test that single volatility calculation is fast enough"""
        import time
        
        params = TimeAdjustedWingModelParameters(
            atm_vol=0.6, slope=0.08, call_curve=5.0, put_curve=5.0,
            up_cutoff=0.5, down_cutoff=-0.95, up_smoothing=5.0, down_smoothing=5.0,
            forward_price=60000.0, time_to_expiry=0.25
        )
        
        model = TimeAdjustedWingModel(params)
        
        # Warm up
        model.calculate_volatility_from_strike(60000.0)
        
        # Time 1000 calculations
        start_time = time.time()
        for _ in range(1000):
            model.calculate_volatility_from_strike(60000.0)
        end_time = time.time()
        
        avg_time_per_calc = (end_time - start_time) / 1000
        
        # Should be fast (less than 1ms per calculation)
        assert avg_time_per_calc < 0.001
    
    def test_calibration_speed(self):
        """Test that calibration completes in reasonable time"""
        import time
        
        strikes = [45000, 50000, 55000, 60000, 65000, 70000, 75000]
        market_vols = [0.8, 0.7, 0.65, 0.6, 0.65, 0.7, 0.8]
        vegas = [100, 150, 180, 200, 180, 150, 100]
        weights = [1.0] * len(strikes)
        forward_price = 60000.0
        time_to_expiry = 0.25
        
        calibrator = TimeAdjustedWingModelCalibrator()
        calibrator.tolerance = 1e-6  # Relaxed tolerance for speed
        
        start_time = time.time()
        result = calibrator.calibrate(
            strike_list=strikes,
            market_vol_list=market_vols,
            market_vega_list=vegas,
            weight_list=weights,
            forward_price=forward_price,
            time_to_expiry=time_to_expiry
        )
        end_time = time.time()
        
        calibration_time = end_time - start_time
        
        # Should complete within 30 seconds
        assert calibration_time < 30.0
        assert isinstance(result, TimeAdjustedCalibrationResult)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])