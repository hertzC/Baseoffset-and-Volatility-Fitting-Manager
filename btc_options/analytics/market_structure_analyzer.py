"""
Market Structure Analyzer

Module for analyzing option market data to detect arbitrage violations and structural issues.
Includes monotonicity checks, spread quality analysis, and no-arbitrage bounds verification.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MarketStructureResults:
    """Container for market structure analysis results."""
    spot_price: float
    total_strikes: int
    
    # Spread quality
    negative_call_spreads: int
    negative_put_spreads: int
    avg_call_spread: float
    avg_put_spread: float
    
    # Monotonicity violations
    call_bid_violations: int
    call_ask_violations: int
    put_bid_violations: int
    put_ask_violations: int
    total_mono_violations: int
    
    # Arbitrage violations
    call_arbitrage_violations: int
    put_arbitrage_violations: int
    total_arbitrage_violations: int
    
    # Overall assessment
    total_issues: int
    quality_assessment: str
    
    def print_report(self) -> None:
        """Print a comprehensive market structure report."""
        print(f"🔍 MARKET STRUCTURE ANALYSIS (Spot: ${self.spot_price:.2f})")
        print("=" * 60)
        
        print(f"📊 Data Overview:")
        print(f"   Total strikes analyzed: {self.total_strikes}")
        
        print(f"\n📈 Spread Quality:")
        print(f"   Negative call spreads: {self.negative_call_spreads}")
        print(f"   Negative put spreads: {self.negative_put_spreads}")
        print(f"   Average call spread: {self.avg_call_spread:.5f}")
        print(f"   Average put spread: {self.avg_put_spread:.5f}")
        
        print(f"\n📉 Monotonicity Violations:")
        print(f"   Call bid violations: {self.call_bid_violations}")
        print(f"   Call ask violations: {self.call_ask_violations}")
        print(f"   Put bid violations: {self.put_bid_violations}")
        print(f"   Put ask violations: {self.put_ask_violations}")
        print(f"   Total monotonicity issues: {self.total_mono_violations}")
        
        print(f"\n⚖️ Arbitrage Violations:")
        print(f"   Call arbitrage violations: {self.call_arbitrage_violations}")
        print(f"   Put arbitrage violations: {self.put_arbitrage_violations}")
        print(f"   Total arbitrage violations: {self.total_arbitrage_violations}")
        
        print(f"\n📋 OVERALL ASSESSMENT:")
        print(f"   Total issues detected: {self.total_issues}")
        print(f"   Quality rating: {self.quality_assessment}")
        print("=" * 60)


class MarketStructureAnalyzer:
    """
    Analyzer for option market structure and arbitrage violations.
    
    Performs comprehensive checks including:
    - Spread quality (negative spreads)
    - Monotonicity (calls decreasing, puts increasing with strike)
    - No-arbitrage bounds (vertical spread constraints)
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize the analyzer.
        
        Args:
            tolerance: Floating point precision tolerance for arbitrage checks
        """
        self.tolerance = tolerance
    
    def analyze(self, price_data: pl.DataFrame) -> MarketStructureResults:
        """
        Perform comprehensive market structure analysis.
        
        Args:
            price_data: DataFrame with columns ['strike', 'S', 'bid_price', 'ask_price', 
                       'bid_price_P', 'ask_price_P'] sorted by strike
        
        Returns:
            MarketStructureResults with detailed analysis
        """
        if price_data.is_empty():
            raise ValueError("Price data cannot be empty")
        
        required_cols = ['strike', 'S', 'bid_price', 'ask_price', 'bid_price_P', 'ask_price_P']
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get basic info
        spot_price = float(price_data['S'][0])
        total_strikes = len(price_data)
        
        # Add spread columns and sort by strike
        analysis_df = price_data.with_columns([
            (pl.col('ask_price') - pl.col('bid_price')).alias('call_spread'),
            (pl.col('ask_price_P') - pl.col('bid_price_P')).alias('put_spread'),
        ]).sort('strike')
        
        # Spread quality analysis
        spread_results = self._analyze_spread_quality(analysis_df)
        
        # Monotonicity and arbitrage analysis
        if total_strikes > 1:
            mono_results = self._analyze_monotonicity(analysis_df, spot_price)
            arbitrage_results = self._analyze_arbitrage_violations(analysis_df, spot_price)
        else:
            mono_results = {'call_bid_violations': 0, 'call_ask_violations': 0, 
                          'put_bid_violations': 0, 'put_ask_violations': 0}
            arbitrage_results = {'call_arbitrage_violations': 0, 'put_arbitrage_violations': 0}
        
        # Calculate totals and assessment
        total_mono_violations = sum(mono_results.values())
        total_arbitrage_violations = sum(arbitrage_results.values())
        total_issues = (total_mono_violations + spread_results['negative_call_spreads'] + 
                       spread_results['negative_put_spreads'] + total_arbitrage_violations)
        
        quality_assessment = "✅ Good" if total_issues <= 2 else "⚠️ Needs review"
        
        return MarketStructureResults(
            spot_price=spot_price,
            total_strikes=total_strikes,
            negative_call_spreads=spread_results['negative_call_spreads'],
            negative_put_spreads=spread_results['negative_put_spreads'],
            avg_call_spread=spread_results['avg_call_spread'],
            avg_put_spread=spread_results['avg_put_spread'],
            call_bid_violations=mono_results['call_bid_violations'],
            call_ask_violations=mono_results['call_ask_violations'],
            put_bid_violations=mono_results['put_bid_violations'],
            put_ask_violations=mono_results['put_ask_violations'],
            total_mono_violations=total_mono_violations,
            call_arbitrage_violations=arbitrage_results['call_arbitrage_violations'],
            put_arbitrage_violations=arbitrage_results['put_arbitrage_violations'],
            total_arbitrage_violations=total_arbitrage_violations,
            total_issues=total_issues,
            quality_assessment=quality_assessment
        )
    
    def _analyze_spread_quality(self, analysis_df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze bid-ask spread quality."""
        negative_call_spreads = len(analysis_df.filter(pl.col('call_spread') <= 0))
        negative_put_spreads = len(analysis_df.filter(pl.col('put_spread') <= 0))
        avg_call_spread = float(analysis_df['call_spread'].mean() or 0)
        avg_put_spread = float(analysis_df['put_spread'].mean() or 0)
        
        return {
            'negative_call_spreads': negative_call_spreads,
            'negative_put_spreads': negative_put_spreads,
            'avg_call_spread': avg_call_spread,
            'avg_put_spread': avg_put_spread
        }
    
    def _analyze_monotonicity(self, analysis_df: pl.DataFrame, spot_price: float) -> Dict[str, int]:
        """
        Analyze monotonicity violations.
        
        Calls should decrease with strike, puts should increase with strike.
        """
        prices = analysis_df.select(['bid_price', 'ask_price', 'bid_price_P', 'ask_price_P']).to_numpy()
        
        call_bids, call_asks = prices[:, 0], prices[:, 1]
        put_bids, put_asks = prices[:, 2], prices[:, 3]
        
        # Vectorized monotonicity check
        call_bid_violations = int(np.sum(call_bids[:-1] < call_bids[1:]))
        call_ask_violations = int(np.sum(call_asks[:-1] < call_asks[1:]))
        put_bid_violations = int(np.sum(put_bids[:-1] > put_bids[1:]))
        put_ask_violations = int(np.sum(put_asks[:-1] > put_asks[1:]))
        
        return {
            'call_bid_violations': call_bid_violations,
            'call_ask_violations': call_ask_violations,
            'put_bid_violations': put_bid_violations,
            'put_ask_violations': put_ask_violations
        }
    
    def _analyze_arbitrage_violations(self, analysis_df: pl.DataFrame, spot_price: float) -> Dict[str, int]:
        """
        Analyze no-arbitrage bound violations.
        
        Vertical spreads cannot exceed the strike difference (in risk-neutral measure).
        """
        prices = analysis_df.select(['strike', 'bid_price', 'ask_price', 'bid_price_P', 'ask_price_P']).to_numpy()
        
        strikes = prices[:, 0]
        call_bids, call_asks = prices[:, 1], prices[:, 2]
        put_bids, put_asks = prices[:, 3], prices[:, 4]
        
        # Calculate strike differences in BTC terms
        strike_diffs_btc = np.diff(strikes) / spot_price
        
        # Calculate price differences
        call_bid_diffs = np.diff(call_bids)
        call_ask_diffs = np.diff(call_asks)
        put_bid_diffs = np.diff(put_bids)
        put_ask_diffs = np.diff(put_asks)
        
        # No-arbitrage violations: spreads can't exceed strike difference
        call_violations = (
            int(np.sum(-call_bid_diffs > strike_diffs_btc + self.tolerance)) +
            int(np.sum(-call_ask_diffs > strike_diffs_btc + self.tolerance))
        )
        put_violations = (
            int(np.sum(put_bid_diffs > strike_diffs_btc + self.tolerance)) +
            int(np.sum(put_ask_diffs > strike_diffs_btc + self.tolerance))
        )
        
        return {
            'call_arbitrage_violations': call_violations,
            'put_arbitrage_violations': put_violations
        }
    
    def quick_check(self, price_data: pl.DataFrame) -> str:
        """
        Perform a quick quality check and return a summary string.
        
        Args:
            price_data: DataFrame with option pricing data
            
        Returns:
            Quick summary string
        """
        try:
            results = self.analyze(price_data)
            return (f"📊 Quality Check: {results.total_mono_violations} mono, "
                   f"{results.negative_call_spreads + results.negative_put_spreads} negative spreads, "
                   f"{results.total_arbitrage_violations} no-arbitrage violations - {results.quality_assessment}")
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"


def analyze_market_structure(price_data: pl.DataFrame, tolerance: float = 1e-6, 
                           print_report: bool = True) -> MarketStructureResults:
    """
    Convenience function for market structure analysis.
    
    Args:
        price_data: DataFrame with option pricing data
        tolerance: Floating point tolerance for arbitrage checks
        print_report: Whether to print the detailed report
        
    Returns:
        MarketStructureResults object
    """
    analyzer = MarketStructureAnalyzer(tolerance=tolerance)
    results = analyzer.analyze(price_data)
    
    if print_report:
        results.print_report()
    
    return results