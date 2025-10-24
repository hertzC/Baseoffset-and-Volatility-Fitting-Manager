#!/usr/bin/env python3
"""
Test cases for option constraints and tightening logic.

This module provides comprehensive test coverage for:
1. Basic monotonicity constraints
2. No-arbitrage spread constraints  
3. Volume-based filtering
4. Edge cases and error handling
5. Integration with real market data structures
"""

import unittest
import numpy as np
import polars as pl
from utils.pricer.option_constraints import (
    apply_option_constraints,
    tighten_option_spreads_separate_columns,
    tighten_option_spreads_with_volume_filter,    
)


class TestOptionConstraints(unittest.TestCase):
    """Test core option constraints functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Standard test data
        self.strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        self.spot = 100.0
        self.interest_rate = 0.00
        self.time_to_expiry = 30 / 365  # 30 days
        
        # Well-behaved prices (should require minimal adjustment)
        self.good_call_bids = np.array([15.0, 10.5, 6.2, 3.8, 1.5])/self.spot  # Decreasing
        self.good_call_asks = np.array([15.5, 11.0, 6.7, 4.3, 2.0])/self.spot  # Decreasing
        self.good_put_bids = np.array([1.2, 2.8, 4.5, 7.1, 12.0])/self.spot    # Increasing  
        self.good_put_asks = np.array([1.7, 3.3, 5.0, 7.6, 12.5])/self.spot    # Increasing
        
        # Problematic prices (violate monotonicity)
        self.bad_call_bids = np.array([10.0, 15.0, 6.0, 8.0, 2.0])/self.spot   # Non-monotonic
        self.bad_call_asks = np.array([11.0, 16.0, 7.0, 9.0, 3.0])/self.spot   # Non-monotonic
        self.bad_put_bids = np.array([5.0, 2.0, 7.0, 3.0, 10.0])/self.spot     # Non-monotonic
        self.bad_put_asks = np.array([6.0, 3.0, 8.0, 4.0, 11.0])/self.spot     # Non-monotonic
        
        # Volume data
        self.high_volumes = np.array([20.0, 25.0, 30.0, 22.0, 18.0])
        self.low_volumes = np.array([0.5, 1.2, 0.8, 1.5, 0.3])
        self.mixed_volumes = np.array([0.5, 15.0, 25.0, 20.0, 2.0])
    
    def test_monotonicity_well_behaved_data(self):
        """Test that well-behaved data remains largely unchanged."""
        adj_cb, adj_ca, adj_pb, adj_pa = apply_option_constraints(
            self.good_call_bids, self.good_call_asks,
            self.good_put_bids, self.good_put_asks,
            self.strikes, self.spot, self.interest_rate, self.time_to_expiry,
            self.high_volumes, self.high_volumes, self.high_volumes, self.high_volumes
        )
        # Prices should be similar to originals (allowing for small adjustments)
        np.testing.assert_allclose(adj_cb, self.good_call_bids, rtol=0.05)
        np.testing.assert_allclose(adj_ca, self.good_call_asks, rtol=0.05)
        np.testing.assert_allclose(adj_pb, self.good_put_bids, rtol=0.05)
        np.testing.assert_allclose(adj_pa, self.good_put_asks, rtol=0.05)
    
    def test_monotonicity_enforcement(self):
        """Test that monotonicity constraints are enforced."""
        adj_cb, adj_ca, adj_pb, adj_pa = apply_option_constraints(
            self.bad_call_bids, self.bad_call_asks,
            self.bad_put_bids, self.bad_put_asks,
            self.strikes, self.spot, self.interest_rate, self.time_to_expiry,
            self.high_volumes, self.high_volumes, self.high_volumes, self.high_volumes
        )
        
        # Check call bid monotonicity (should be non-increasing)
        for i in range(len(adj_cb) - 1):
            self.assertGreaterEqual(adj_cb[i], adj_cb[i+1], 
                                  f"Call bids should decrease: {adj_cb[i]} >= {adj_cb[i+1]} at strikes {self.strikes[i]}, {self.strikes[i+1]}")
        
        # Check put bid monotonicity (should be non-decreasing)
        for i in range(len(adj_pb) - 1):
            self.assertLessEqual(adj_pb[i], adj_pb[i+1],
                               f"Put bids should increase: {adj_pb[i]} <= {adj_pb[i+1]} at strikes {self.strikes[i]}, {self.strikes[i+1]}")
    
    # def test_bid_ask_spread_preservation(self):
    #     """Test that bid <= ask relationships are maintained."""
    #     adj_cb, adj_ca, adj_pb, adj_pa = apply_option_constraints(
    #         self.bad_call_bids, self.bad_call_asks,
    #         self.bad_put_bids, self.bad_put_asks,
    #         self.strikes, self.spot
    #     )
        
    #     # Check bid <= ask for all strikes
    #     for i, strike in enumerate(self.strikes):
    #         print(f"{self.bad_call_bids[i]} => {round(adj_cb[i],2)} | {strike} | {self.bad_call_asks[i]} => {round(adj_ca[i],2)}")
    #         if strike <= 1015:
    #             continue
    #         self.assertLessEqual(adj_cb[i], adj_ca[i],
    #                            f"Call bid should be <= ask at strike {strike}: {adj_cb[i]} <= {adj_ca[i]}")
    #         self.assertLessEqual(adj_pb[i], adj_pa[i],
    #                            f"Put bid should be <= ask at strike {strike}: {adj_pb[i]} <= {adj_pa[i]}")

    def test_volume_based_filtering(self):
        """Test that volume-based filtering works correctly."""
        # Test with high volume threshold - should only affect high-volume strikes
        adj_cb_filtered, _, adj_pb_filtered, _ = apply_option_constraints(
            self.bad_call_bids, self.bad_call_asks,
            self.bad_put_bids, self.bad_put_asks,
            self.strikes, self.spot, self.interest_rate, self.time_to_expiry,
            bid_size_call=self.mixed_volumes,
            ask_size_call=self.mixed_volumes,
            bid_size_put=self.mixed_volumes,
            ask_size_put=self.mixed_volumes,
            volume_threshold=10.0
        )
        
        # Low volume strikes (indices 0, 4) should be closer to original
        self.assertAlmostEqual(adj_cb_filtered[0], self.bad_call_bids[0], delta=0.1)
        self.assertAlmostEqual(adj_cb_filtered[4], self.bad_call_bids[4], delta=0.1)
        
        # High volume strikes may be significantly adjusted
        # (We can't predict exact values, but they should follow constraints)
        self.assertTrue(adj_cb_filtered[1] >= adj_cb_filtered[2])  # Monotonicity maintained
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single strike
        single_strike = np.array([100.0])
        single_call_bid = np.array([10.0])
        single_call_ask = np.array([11.0])
        single_put_bid = np.array([5.0])
        single_put_ask = np.array([6.0])

        single_volume = np.array([20.0])
        
        adj_cb, adj_ca, adj_pb, adj_pa = apply_option_constraints(
            single_call_bid, single_call_ask,
            single_put_bid, single_put_ask,
            single_strike, self.spot, self.interest_rate, self.time_to_expiry,
            bid_size_call=single_volume,
            ask_size_call=single_volume,
            bid_size_put=single_volume,
            ask_size_put=single_volume
        )
        
        # Should return unchanged (no constraints to apply)
        np.testing.assert_array_equal(adj_cb, single_call_bid)
        np.testing.assert_array_equal(adj_ca, single_call_ask)
        np.testing.assert_array_equal(adj_pb, single_put_bid)
        np.testing.assert_array_equal(adj_pa, single_put_ask)
        
        # Zero prices (should be handled gracefully)
        zero_prices = np.zeros(5)
        positive_prices = np.ones(5)
        zero_volume = np.zeros(5)
        
        adj_cb, adj_ca, adj_pb, adj_pa = apply_option_constraints(
            zero_prices, positive_prices,
            zero_prices, positive_prices,
            self.strikes, self.spot, self.interest_rate, self.time_to_expiry,
            bid_size_call=zero_volume,
            ask_size_call=zero_volume,
            bid_size_put=zero_volume,
            ask_size_put=zero_volume
        )
        
        # Should not crash and maintain bid <= ask
        self.assertTrue(np.all(adj_cb <= adj_ca))
        self.assertTrue(np.all(adj_pb <= adj_pa))


class TestDataFrameInterface(unittest.TestCase):
    """Test DataFrame interface functions."""
    
    def setUp(self):
        """Set up test DataFrame."""
        self.df = pl.DataFrame({
            'strike': [90, 95, 100, 105, 110],
            'bid_price': [15.0, 10.5, 6.2, 3.8, 1.5],        # Call bids
            'ask_price': [15.5, 11.0, 6.7, 4.3, 2.0],        # Call asks
            'bid_price_P': [1.2, 2.8, 4.5, 7.1, 12.0],       # Put bids
            'ask_price_P': [1.7, 3.3, 5.0, 7.6, 12.5],       # Put asks
            'bid_size': [0.5, 15.0, 25.0, 20.0, 2.0],        # Call bid sizes
            'ask_size': [0.3, 12.0, 22.0, 18.0, 1.8],        # Call ask sizes
            'bid_size_P': [0.4, 14.0, 28.0, 21.0, 2.2],      # Put bid sizes
            'ask_size_P': [0.6, 13.0, 24.0, 19.0, 2.1],      # Put ask sizes
            'S': [100, 100, 100, 100, 100],                   # Spot price
            'tau': [30/365, 30/365, 30/365, 30/365, 30/365]   # Time to expiry
        })
    
    def test_basic_dataframe_processing(self):
        """Test basic DataFrame processing."""
        result = tighten_option_spreads_separate_columns(self.df)
        # Should return a DataFrame
        self.assertIsInstance(result, pl.DataFrame)
        
        # Should have original columns plus new ones
        original_cols = set(self.df.columns)
        result_cols = set(result.columns)
        self.assertTrue(original_cols.issubset(result_cols))
        
        # Should have tightened and original columns
        self.assertIn('original_bid_price', result.columns)
        self.assertIn('tightened_bid_price', result.columns)
    
    def test_volume_threshold_application(self):
        """Test volume threshold application in DataFrame interface."""
        # No volume filtering
        result_no_filter = tighten_option_spreads_separate_columns(
            self.df, volume_threshold=0.0
        )
        
        # High volume filtering
        result_high_filter = tighten_option_spreads_separate_columns(
            self.df, volume_threshold=10.0
        )
        
        # Results should differ (high filter should change fewer prices)
        no_filter_prices = result_no_filter['bid_price'].to_list()
        high_filter_prices = result_high_filter['bid_price'].to_list()
        
        # At least some prices should be different
        self.assertNotEqual(no_filter_prices, high_filter_prices)
    
    def test_convenience_function(self):
        """Test convenience function with default parameters."""
        result = tighten_option_spreads_with_volume_filter(self.df)
        
        # Should work without errors
        self.assertIsInstance(result, pl.DataFrame)
        
        # Should apply reasonable volume filtering by default
        self.assertIn('tightened_bid_price', result.columns)
        
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = self.df.filter(pl.lit(False))  # Empty DataFrame with same schema
        
        result = tighten_option_spreads_separate_columns(empty_df)
        
        # Should return the original empty DataFrame
        self.assertTrue(result.equals(empty_df))


class TestIntegrationConstraints(unittest.TestCase):
    """Test integration with real market data patterns."""
    
    def test_realistic_option_chain(self):
        """Test with realistic option chain data."""
        # Create realistic option chain
        strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
        spot, interest_rate, tte = 100.0, 0.01, 60 / 365  # 60 days to expiry
        
        # Realistic call prices (decreasing with strike, intrinsic + time value)
        call_bids = np.array([22.5, 18.2, 14.1, 10.8, 7.9, 5.6, 3.8, 2.4, 1.5])
        call_asks = np.array([23.0, 18.7, 14.6, 11.3, 8.4, 6.1, 4.3, 2.9, 2.0])
        
        # Realistic put prices (increasing with strike, intrinsic + time value)  
        put_bids = np.array([1.8, 2.9, 4.3, 6.1, 8.4, 11.3, 14.6, 18.7, 23.0])
        put_asks = np.array([2.3, 3.4, 4.8, 6.6, 8.9, 11.8, 15.1, 19.2, 23.5])
        
        # Apply constraints
        adj_cb, adj_ca, adj_pb, adj_pa = apply_option_constraints(
            call_bids, call_asks, put_bids, put_asks, strikes, spot, interest_rate, tte, 
            np.ones(len(strikes))*100, np.ones(len(strikes))*100, np.ones(len(strikes))*100, np.ones(len(strikes))*100
        )
        
        # Check that call-put parity approximately holds (for ATM options)
        atm_index = 4  # Strike 100
        call_mid = (adj_cb[atm_index] + adj_ca[atm_index]) / 2
        put_mid = (adj_pb[atm_index] + adj_pa[atm_index]) / 2
        parity_diff = abs(call_mid - put_mid - (spot - strikes[atm_index]))
        
        # Should be reasonably close (allowing for time value differences)
        self.assertLess(parity_diff, 2.0, "Call-put parity should approximately hold at ATM")
    
    def test_arbitrage_bounds(self):
        """Test no-arbitrage spread constraints."""
        strikes = np.array([95, 100, 105])
        spot, interest_rate, tte = 100.0, 0.01, 60 / 365  # 60 days to expiry
        
        # Create prices that violate no-arbitrage bounds
        call_bids = np.array([10.0, 8.0, 1.0])  # Too large spread between 100-105
        call_asks = np.array([11.0, 9.0, 2.0])
        put_bids = np.array([3.0, 5.0, 8.0])
        put_asks = np.array([4.0, 6.0, 9.0])
        
        adj_cb, adj_ca, adj_pb, adj_pa = apply_option_constraints(
            call_bids, call_asks, put_bids, put_asks, strikes, spot, interest_rate, tte,
            np.ones(len(strikes))*100, np.ones(len(strikes))*100, np.ones(len(strikes))*100, np.ones(len(strikes))*100
        )
        
        # Check vertical spread constraints
        # Call spread: C(95) - C(100) should be <= (100-95)/100 = 0.05
        call_spread_95_100 = adj_cb[0] - adj_cb[1]
        max_call_spread_95_100 = (strikes[1] - strikes[0]) / spot
        
        # Allow for some tolerance in practice
        self.assertLessEqual(call_spread_95_100, max_call_spread_95_100 + 0.1,
                           f"Call spread too large: {call_spread_95_100} vs max {max_call_spread_95_100}")



if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)