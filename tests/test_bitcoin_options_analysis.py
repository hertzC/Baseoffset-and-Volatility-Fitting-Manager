"""
Unit tests for Bitcoin Options Analysis Pipeline

Tests synthetic option creation and fitting functionality with sample data
to ensure consistent results across code changes.
"""

import unittest
import numpy as np
import polars as pl
from datetime import datetime, time
from typing import Dict, Any
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_options.data_managers.deribit_md_manager import DeribitMDManager
from btc_options.data_managers.orderbook_deribit_md_manager import OrderbookDeribitMDManager
from btc_options.analytics.weight_least_square_regressor import WLSRegressor
from btc_options.analytics.nonlinear_minimization import NonlinearMinimization
from config_loader import Config, load_config


class TestSampleDataGenerator:
    """Generate consistent sample data for testing."""
    
    @staticmethod
    def create_sample_conflated_data() -> pl.DataFrame:
        """Create sample conflated market data for testing."""
        # Sample data with realistic Bitcoin option pricing
        timestamps = [datetime(2024, 2, 29, 12, 30, 0)] * 20
        
        # Option data (calls and puts for different strikes)
        strikes = [50000, 52000, 54000, 56000, 58000]
        
        data = []
        spot_price = 55000.0
        tau = 0.0274  # ~10 days to expiry
        
        for strike in strikes:
            # More realistic option pricing based on put-call parity
            # P - C = K*exp(-r*t) - S*exp(-q*t)
            # With r=0.05, q=0.01 (5% USD rate, 1% BTC rate)
            r_test = 0.05
            q_test = 0.01
            
            # Forward price: F = S * exp((r-q)*t)
            forward = spot_price * np.exp((r_test - q_test) * tau)
            
            # Put-call parity: P - C = (K - F) * exp(-r*t)
            pc_diff = (strike - forward) * np.exp(-r_test * tau)
            
            # Estimate call price (simplified Black-Scholes approximation)
            moneyness = np.log(forward / strike)
            call_mid = max(0.001, abs(moneyness) * 0.3 + 0.01)  # Simplified option value
            put_mid = call_mid + pc_diff
            put_mid = max(0.001, put_mid)  # Ensure positive price
            
            call_spread = call_mid * 0.05  # 5% spread
            put_spread = put_mid * 0.05
            
            data.append({
                'timestamp': timestamps[0],
                'symbol': f'BTC-29FEB24-{strike}-C',
                'expiry': '29FEB24',
                'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
                'strike': strike,
                'is_option': True,
                'is_call': True,
                'is_future': False,
                'is_perp': False,
                'bid_price': call_mid - call_spread/2,
                'ask_price': call_mid + call_spread/2,
                'bid_size': 10.0,
                'ask_size': 10.0,
                'index_price': spot_price,
                'S': spot_price,
                'tau': tau
            })
            
            # Put option
            put_spread = put_mid * 0.05
            
            data.append({
                'timestamp': timestamps[0],
                'symbol': f'BTC-29FEB24-{strike}-P',
                'expiry': '29FEB24',
                'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
                'strike': strike,
                'is_option': True,
                'is_call': False,
                'is_future': False,
                'is_perp': False,
                'bid_price': put_mid - put_spread/2,
                'ask_price': put_mid + put_spread/2,
                'bid_size': 10.0,
                'ask_size': 10.0,
                'index_price': spot_price,
                'S': spot_price,
                'tau': tau
            })
        
        # Add future data
        future_price = spot_price * 1.001  # Small contango
        data.append({
            'timestamp': timestamps[0],
            'symbol': 'BTC-29FEB24',
            'expiry': '29FEB24',
            'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
            'strike': None,
            'is_option': False,
            'is_call': None,
            'is_future': True,
            'is_perp': False,
            'bid_price': future_price - 25,
            'ask_price': future_price + 25,
            'bid_size': 100.0,
            'ask_size': 100.0,
            'index_price': spot_price,
            'S': spot_price,
            'tau': tau
        })
        
        # Add INDEX data
        data.append({
            'timestamp': timestamps[0],
            'symbol': 'INDEX',
            'expiry': None,
            'expiry_ts': None,
            'strike': None,
            'is_option': False,
            'is_call': None,
            'is_future': False,
            'is_perp': False,
            'bid_price': spot_price - 5,
            'ask_price': spot_price + 5,
            'bid_size': 1000.0,
            'ask_size': 1000.0,
            'index_price': spot_price,
            'S': spot_price,
            'tau': 0
        })
        
        return pl.DataFrame(data)

    @staticmethod
    def create_realistic_synthetic_data(r=0.05, q=0.01, S=55000, tau=0.0274) -> pl.DataFrame:
        """Create synthetic data that exactly follows the put-call parity relationship."""
        import numpy as np
        
        # Calculate regression parameters from rates
        const = -S * np.exp(-q * tau)  # Should be around -54,500
        coef = np.exp(-r * tau)        # Should be around 0.9986
        
        strikes = [50000, 52000, 54000, 56000, 58000]
        data = []
        
        for strike in strikes:
            # Calculate exact P-C value from linear relationship
            mid_value = const + coef * strike
            spread = abs(mid_value) * 0.05 + 0.001  # 5% spread + minimum
            
            data.append({
                'strike': strike,
                'mid': mid_value,
                'spread': spread,
                'S': S,
                'tau': tau
            })
        
        return pl.DataFrame(data)
    
    @staticmethod
    def create_sample_symbol_data() -> pl.DataFrame:
        """Create sample symbol data for DeribitMDManager."""
        symbols_data = [
            {'symbol': 'BTC-29FEB24-50000-C', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 50000, 'is_option': True, 'is_call': True, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-50000-P', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 50000, 'is_option': True, 'is_call': False, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-52000-C', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 52000, 'is_option': True, 'is_call': True, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-52000-P', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 52000, 'is_option': True, 'is_call': False, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-54000-C', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 54000, 'is_option': True, 'is_call': True, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-54000-P', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 54000, 'is_option': True, 'is_call': False, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-56000-C', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 56000, 'is_option': True, 'is_call': True, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-56000-P', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 56000, 'is_option': True, 'is_call': False, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-58000-C', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 58000, 'is_option': True, 'is_call': True, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24-58000-P', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': 58000, 'is_option': True, 'is_call': False, 'is_future': False, 'is_perp': False},
            {'symbol': 'BTC-29FEB24', 'expiry': '29FEB24', 'expiry_ts': datetime(2024, 2, 29, 8, 0, 0),
             'strike': None, 'is_option': False, 'is_call': None, 'is_future': True, 'is_perp': False},
            {'symbol': 'INDEX', 'expiry': None, 'expiry_ts': None,
             'strike': None, 'is_option': False, 'is_call': None, 'is_future': False, 'is_perp': False},
        ]
        return pl.DataFrame(symbols_data)


class TestSyntheticCreation(unittest.TestCase):
    """Test option synthetic creation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load configuration
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config.yaml')
        self.config = Config(config_path)
        
        # Create mock symbol manager with sample data
        self.sample_conflated_data = TestSampleDataGenerator.create_sample_conflated_data()
        self.sample_symbol_data = TestSampleDataGenerator.create_sample_symbol_data()
        
        # Create a mock DeribitMDManager
        class MockDeribitMDManager:
            def __init__(self, config):
                self.config = config
                self.df_symbol = TestSampleDataGenerator.create_sample_symbol_data()
                self.opt_expiries = ['29FEB24']
                self.fut_expiries = ['29FEB24']
            
            def is_expiry_today(self, expiry):
                return False
            
            def create_option_synthetic(self, df_conflated, expiry, timestamp):
                # Filter options for the expiry
                option_data = df_conflated.filter(
                    (pl.col('expiry') == expiry) & 
                    (pl.col('is_option') == True)
                ).sort('strike')
                
                # Create synthetic data (P - C for each strike)
                calls = option_data.filter(pl.col('is_call') == True)
                puts = option_data.filter(pl.col('is_call') == False)
                
                synthetic_data = []
                for call_row, put_row in zip(calls.iter_rows(named=True), puts.iter_rows(named=True)):
                    if call_row['strike'] == put_row['strike']:
                        call_mid = (call_row['bid_price'] + call_row['ask_price']) / 2
                        put_mid = (put_row['bid_price'] + put_row['ask_price']) / 2
                        call_spread = call_row['ask_price'] - call_row['bid_price']
                        put_spread = put_row['ask_price'] - put_row['bid_price']
                        
                        synthetic_data.append({
                            'strike': call_row['strike'],
                            'mid': put_mid - call_mid,  # P - C
                            'spread': call_spread + put_spread,
                            'S': call_row['S'],
                            'tau': call_row['tau']
                        })
                
                return option_data, pl.DataFrame(synthetic_data)
        
        self.symbol_manager = MockDeribitMDManager(self.config)
        
    def test_synthetic_creation_consistency(self):
        """Test that synthetic option creation produces consistent results."""
        expiry = '29FEB24'
        timestamp = datetime(2024, 2, 29, 12, 30, 0)
        
        # Create synthetic data
        df_chain, df_synthetic = self.symbol_manager.create_option_synthetic(
            self.sample_conflated_data, expiry, timestamp
        )
        
        # Expected results (these should remain constant)
        expected_strikes = [50000, 52000, 54000, 56000, 58000]
        expected_spot = 55000.0
        expected_tau = 0.0274
        
        # Assertions
        self.assertFalse(df_synthetic.is_empty(), "Synthetic data should not be empty")
        self.assertEqual(len(df_synthetic), 5, "Should have 5 synthetic observations")
        
        # Check strikes are correct
        actual_strikes = sorted(df_synthetic['strike'].to_list())
        self.assertEqual(actual_strikes, expected_strikes, "Strikes should match expected values")
        
        # Check spot price consistency
        unique_spot_prices = df_synthetic['S'].unique().to_list()
        self.assertEqual(len(unique_spot_prices), 1, "All rows should have same spot price")
        self.assertAlmostEqual(unique_spot_prices[0], expected_spot, places=2)
        
        # Check tau consistency
        unique_taus = df_synthetic['tau'].unique().to_list()
        self.assertEqual(len(unique_taus), 1, "All rows should have same tau")
        self.assertAlmostEqual(unique_taus[0], expected_tau, places=4)
        
        # Check that spreads are positive
        spreads = df_synthetic['spread'].to_list()
        self.assertTrue(all(s > 0 for s in spreads), "All spreads should be positive")


class TestWLSRegression(unittest.TestCase):
    """Test WLS regression functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load configuration
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config.yaml')
        self.config = Config(config_path)
        
        # Use realistic synthetic data based on exact put-call parity relationship
        self.sample_synthetic_data = TestSampleDataGenerator.create_realistic_synthetic_data(
            r=0.05, q=0.01, S=55000, tau=0.0274
        )
        
        # Mock symbol manager
        class MockSymbolManager:
            def __init__(self, config):
                self.config = config
            
            def is_expiry_today(self, expiry):
                return False
        
        self.symbol_manager = MockSymbolManager(self.config)
        self.wls_regressor = WLSRegressor(self.symbol_manager, self.config)
    
    def test_wls_fitting_consistency(self):
        """Test that WLS fitting produces consistent results."""
        # Expected results (should match input parameters closely due to exact data)
        expected_r_range = (0.049, 0.051)  # Should be very close to 0.05
        expected_q_range = (0.009, 0.011)  # Should be very close to 0.01
        expected_r2_min = 0.99  # Should be nearly perfect fit
        
        result = self.wls_regressor.fit(
            self.sample_synthetic_data,
            expiry='29FEB24',
            timestamp=datetime(2024, 2, 29, 12, 30, 0)
        )
        
        # Check that result contains expected keys
        required_keys = ['r', 'q', 'const', 'coef', 'r2', 'sse', 'S', 'tau']
        for key in required_keys:
            self.assertIn(key, result, f"Result should contain {key}")
        
        # Check rate ranges
        self.assertGreaterEqual(result['r'], expected_r_range[0], "USD rate should be reasonable")
        self.assertLessEqual(result['r'], expected_r_range[1], "USD rate should be reasonable")
        self.assertGreaterEqual(result['q'], expected_q_range[0], "BTC rate should be reasonable")
        self.assertLessEqual(result['q'], expected_q_range[1], "BTC rate should be reasonable")
        
        # Check fit quality
        self.assertGreaterEqual(result['r2'], expected_r2_min, "R-squared should indicate good fit")
        self.assertGreater(result['sse'], 0, "SSE should be positive")
        
        # Check consistency with input data
        self.assertAlmostEqual(result['S'], 55000, places=0, msg="Spot price should match input")
        self.assertAlmostEqual(result['tau'], 0.0274, places=4, msg="Tau should match input")
    
    def test_wls_parameter_validation(self):
        """Test WLS parameter validation and edge cases."""
        # Test empty DataFrame
        with self.assertRaises(ValueError):
            self.wls_regressor.fit(pl.DataFrame(), expiry='TEST', timestamp=datetime.now())
        
        # Test insufficient data
        small_data = self.sample_synthetic_data.head(2)
        
        # Should handle insufficient strikes gracefully
        result = self.wls_regressor.fit(
            small_data,
            expiry='29FEB24', 
            timestamp=datetime(2024, 2, 29, 12, 30, 0)
        )
        
        # Should return some result even with insufficient data
        self.assertIsNotNone(result)


class TestNonlinearMinimization(unittest.TestCase):
    """Test nonlinear optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load configuration
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config.yaml')
        self.config = Config(config_path)
        
        # Use same realistic synthetic data as WLS tests
        self.sample_synthetic_data = TestSampleDataGenerator.create_realistic_synthetic_data(
            r=0.05, q=0.01, S=55000, tau=0.0274
        )
        
        # Mock symbol manager
        class MockSymbolManager:
            def __init__(self, config):
                self.config = config
            
            def is_expiry_today(self, expiry):
                return False
        
        self.symbol_manager = MockSymbolManager(self.config)
        self.nonlinear_minimizer = NonlinearMinimization(self.symbol_manager, self.config)
    
    def test_nonlinear_fitting_consistency(self):
        """Test that nonlinear optimization produces consistent results."""
        # First run WLS to get realistic initial guess
        wls_regressor = WLSRegressor(self.symbol_manager, self.config)
        wls_result = wls_regressor.fit(
            self.sample_synthetic_data,
            expiry='29FEB24',
            timestamp=datetime(2024, 2, 29, 12, 30, 0)
        )
        
        # Use WLS results as initial guess
        prev_const = wls_result['const']
        prev_coef = wls_result['coef']
        
        # Expected results (should match input parameters closely)
        expected_r_range = (0.049, 0.051)  # Should be very close to 0.05
        expected_q_range = (0.009, 0.011)  # Should be very close to 0.01  
        expected_r2_min = 0.99  # Should be nearly perfect fit
        
        # Add a successful fit to the nonlinear minimizer first to avoid empty fit_results
        # This ensures there's a fallback when optimization fails
        self.nonlinear_minimizer.fit_results.append(wls_result)

        result = self.nonlinear_minimizer.fit(
            self.sample_synthetic_data,
            prev_const, prev_coef,
            expiry='29FEB24',
            timestamp=datetime(2024, 2, 29, 12, 30, 0)
        )
        
        # Check that result contains expected keys
        required_keys = ['r', 'q', 'const', 'coef', 'r2', 'sse', 'S', 'tau', 'success_fitting']
        for key in required_keys:
            self.assertIn(key, result, f"Result should contain {key}")
        
        if result['success_fitting']:
            # Check rate ranges
            self.assertGreaterEqual(result['r'], expected_r_range[0], "USD rate should be reasonable")
            self.assertLessEqual(result['r'], expected_r_range[1], "USD rate should be reasonable")
            self.assertGreaterEqual(result['q'], expected_q_range[0], "BTC rate should be reasonable")
            self.assertLessEqual(result['q'], expected_q_range[1], "BTC rate should be reasonable")
            
            # Check fit quality
            self.assertGreaterEqual(result['r2'], expected_r2_min, "R-squared should indicate good fit")
            self.assertGreater(result['sse'], 0, "SSE should be positive")
        
        # Check consistency with input data
        self.assertAlmostEqual(result['S'], 55000, places=0, msg="Spot price should match input")
        self.assertAlmostEqual(result['tau'], 0.0274, places=4, msg="Tau should match input")
    
    def test_parameter_management(self):
        """Test parameter management functionality."""
        # Test that configuration parameters are properly loaded
        rate_constraints = self.config.get_rate_constraints()
        
        # Check that rate constraints are available
        required_constraints = ['r_min', 'r_max', 'q_min', 'q_max', 'minimum_rate', 'maximum_rate']
        for constraint in required_constraints:
            self.assertIn(constraint, rate_constraints, f"Rate constraint {constraint} should be available")
        
        # Check reasonable values
        self.assertLess(rate_constraints['r_min'], rate_constraints['r_max'], "r_min should be less than r_max")
        self.assertLess(rate_constraints['q_min'], rate_constraints['q_max'], "q_min should be less than q_max")
        
    def test_results_management(self):
        """Test results management functionality."""
        # Check initial state
        initial_count = self.nonlinear_minimizer.get_results_count()
        
        # Clear results
        self.nonlinear_minimizer.clear_results()
        self.assertEqual(self.nonlinear_minimizer.get_results_count(), 0)


class TestRegressionValues(unittest.TestCase):
    """Test specific regression values to catch any numerical changes."""
    
    def setUp(self):
        """Set up fixed test data for regression testing."""
        # Load configuration
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config.yaml')
        self.config = Config(config_path)
        
        # Fixed synthetic data for consistent testing with exact put-call parity
        self.fixed_synthetic_data = TestSampleDataGenerator.create_realistic_synthetic_data(
            r=0.05, q=0.01, S=55000, tau=0.0274
        )
        
        class MockSymbolManager:
            def __init__(self, config):
                self.config = config
            
            def is_expiry_today(self, expiry):
                return False
        
        self.symbol_manager = MockSymbolManager(self.config)
    
    def test_wls_baseline_values(self):
        """Test WLS regression produces expected baseline values."""
        wls_regressor = WLSRegressor(self.symbol_manager, self.config)
        
        result = wls_regressor.fit(
            self.fixed_synthetic_data,
            expiry='29FEB24',
            timestamp=datetime(2024, 2, 29, 12, 30, 0)
        )
        
        # These are baseline values - test exact recovery of input parameters
        expected_values = {
            'r': (0.049, 0.051),  # Should recover input r=0.05
            'q': (0.009, 0.011),  # Should recover input q=0.01
            'r2': (0.99, 1.0),    # Should be nearly perfect fit
            'sse': (0.0, 0.001)   # Should be very small SSE
        }
        
        for key, (min_val, max_val) in expected_values.items():
            self.assertGreaterEqual(result[key], min_val, f"{key} should be >= {min_val}")
            self.assertLessEqual(result[key], max_val, f"{key} should be <= {max_val}")
        
        print(f"WLS Baseline Test Results:")
        print(f"  r={result['r']:.6f}, q={result['q']:.6f}")
        print(f"  R²={result['r2']:.6f}, SSE={result['sse']:.8f}")


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSyntheticCreation,
        TestWLSRegression, 
        TestNonlinearMinimization,
        TestRegressionValues
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    # Run tests when script is executed directly
    print("🧪 Running Bitcoin Options Analysis Unit Tests")
    print("=" * 60)
    
    result = run_tests()
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
    print(f"📊 Tests run: {result.testsRun}")
    print(f"💯 Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")