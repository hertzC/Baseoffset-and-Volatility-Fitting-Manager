"""
Arbitrage Analysis Module

This module provides functions to check for various types of arbitrage violations
in option pricing data.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def check_butterfly_arbitrage(
    strikes: List[float], 
    call_prices: List[float], 
    put_prices: List[float], 
    tolerance: float = 1e-6
) -> List[Dict[str, Any]]:
    """
    Check for butterfly arbitrage violations in option prices.
    
    Butterfly arbitrage condition: For strikes K1 < K2 < K3 with equal spacing,
    the butterfly spread C(K1) + C(K3) - 2*C(K2) should be non-negative.
    
    Args:
        strikes: List of strike prices (should be sorted)
        call_prices: List of call option prices corresponding to strikes
        put_prices: List of put option prices corresponding to strikes  
        tolerance: Tolerance for numerical errors (default 1e-6)
        
    Returns:
        List of dictionaries containing violation details. Empty list if no violations.
        Each violation dict contains:
        - type: 'Call Butterfly' or 'Put Butterfly'
        - strikes: (K1, K2, K3) tuple
        - prices: (P1, P2, P3) tuple  
        - violation: The negative butterfly value
        - severity: Absolute value of violation
    """
    violations = []
    
    if len(strikes) != len(call_prices) or len(strikes) != len(put_prices):
        raise ValueError("Strikes, call_prices, and put_prices must have the same length")
    
    if len(strikes) < 3:
        return violations  # Need at least 3 strikes for butterfly check
    
    # Check each triplet of consecutive strikes
    for i in range(1, len(strikes) - 1):
        k1, k2, k3 = strikes[i-1], strikes[i], strikes[i+1]
        
        # Only check if strikes are reasonably evenly spaced (within 20% difference)
        spacing1 = k2 - k1
        spacing2 = k3 - k2
        if spacing1 > 0 and spacing2 > 0:
            spacing_ratio = abs(spacing1 - spacing2) / min(spacing1, spacing2)
            if spacing_ratio < 0.2:  # Strikes are reasonably evenly spaced
                
                # Check call butterfly condition
                c1, c2, c3 = call_prices[i-1], call_prices[i], call_prices[i+1]
                call_butterfly_value = c1 + c3 - 2 * c2
                
                if call_butterfly_value < -tolerance:
                    violations.append({
                        'type': 'Call Butterfly',
                        'strikes': (k1, k2, k3),
                        'prices': (c1, c2, c3),
                        'violation': call_butterfly_value,
                        'severity': abs(call_butterfly_value)
                    })
                
                # Check put butterfly condition  
                p1, p2, p3 = put_prices[i-1], put_prices[i], put_prices[i+1]
                put_butterfly_value = p1 + p3 - 2 * p2
                
                if put_butterfly_value < -tolerance:
                    violations.append({
                        'type': 'Put Butterfly',
                        'strikes': (k1, k2, k3),
                        'prices': (p1, p2, p3),
                        'violation': put_butterfly_value,
                        'severity': abs(put_butterfly_value)
                    })
    
    return violations


def check_price_monotonicity(
    strikes: List[float], 
    call_prices: List[float], 
    put_prices: List[float]
) -> Dict[str, Any]:
    """
    Check monotonicity conditions for option prices.
    
    Monotonicity conditions:
    - Call prices should decrease with increasing strike (non-increasing)
    - Put prices should increase with increasing strike (non-decreasing)
    
    Args:
        strikes: List of strike prices (should be sorted)
        call_prices: List of call option prices
        put_prices: List of put option prices
        
    Returns:
        Dictionary with monotonicity check results:
        - call_monotonic: Boolean indicating if call prices are monotonic
        - put_monotonic: Boolean indicating if put prices are monotonic  
        - call_violations: List of (strike1, strike2, price1, price2) tuples for violations
        - put_violations: List of (strike1, strike2, price1, price2) tuples for violations
    """
    if len(strikes) != len(call_prices) or len(strikes) != len(put_prices):
        raise ValueError("Strikes, call_prices, and put_prices must have the same length")
    
    call_violations = []
    put_violations = []
    
    # Check call price monotonicity (should decrease with strike)
    for i in range(len(call_prices) - 1):
        if call_prices[i] < call_prices[i + 1]:
            call_violations.append((
                strikes[i], strikes[i + 1], 
                call_prices[i], call_prices[i + 1]
            ))
    
    # Check put price monotonicity (should increase with strike)  
    for i in range(len(put_prices) - 1):
        if put_prices[i] > put_prices[i + 1]:
            put_violations.append((
                strikes[i], strikes[i + 1],
                put_prices[i], put_prices[i + 1]
            ))
    
    return {
        'call_monotonic': len(call_violations) == 0,
        'put_monotonic': len(put_violations) == 0,
        'call_violations': call_violations,
        'put_violations': put_violations
    }


def check_call_put_parity(
    strikes: List[float],
    call_prices: List[float], 
    put_prices: List[float],
    forward_price: float,
    discount_factor: float = 1.0,
    tolerance: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Check call-put parity violations.
    
    Call-put parity: C - P = DF * (F - K)
    where DF is the discount factor, F is forward price, K is strike
    
    Args:
        strikes: List of strike prices
        call_prices: List of call option prices
        put_prices: List of put option prices  
        forward_price: Forward price of underlying
        discount_factor: Discount factor (default 1.0 for futures)
        tolerance: Tolerance for parity violations
        
    Returns:
        List of parity violation dictionaries
    """
    violations = []
    
    for i, strike in enumerate(strikes):
        call_price = call_prices[i]
        put_price = put_prices[i]
        
        # Calculate theoretical parity value
        theoretical_diff = discount_factor * (forward_price - strike)
        actual_diff = call_price - put_price
        
        parity_violation = abs(actual_diff - theoretical_diff)
        
        if parity_violation > tolerance:
            violations.append({
                'strike': strike,
                'call_price': call_price,
                'put_price': put_price,
                'actual_diff': actual_diff,
                'theoretical_diff': theoretical_diff,
                'violation': parity_violation
            })
    
    return violations


def analyze_arbitrage_comprehensive(
    strikes: List[float],
    call_prices: List[float],
    put_prices: List[float], 
    forward_price: float,
    discount_factor: float = 1.0,
    butterfly_tolerance: float = 1e-6,
    parity_tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Perform comprehensive arbitrage analysis on option prices.
    
    Args:
        strikes: List of strike prices
        call_prices: List of call prices
        put_prices: List of put prices
        forward_price: Forward price of underlying
        discount_factor: Discount factor for present value calculations
        butterfly_tolerance: Tolerance for butterfly arbitrage
        parity_tolerance: Tolerance for call-put parity
        
    Returns:
        Comprehensive arbitrage analysis results
    """
    # Validate inputs
    if not (len(strikes) == len(call_prices) == len(put_prices)):
        raise ValueError("All input lists must have the same length")
    
    # Check if strikes are sorted
    if not all(strikes[i] <= strikes[i+1] for i in range(len(strikes)-1)):
        # Sort all arrays by strike price
        sorted_indices = np.argsort(strikes)
        strikes = [strikes[i] for i in sorted_indices]
        call_prices = [call_prices[i] for i in sorted_indices]
        put_prices = [put_prices[i] for i in sorted_indices]
    
    # Perform all arbitrage checks
    butterfly_violations = check_butterfly_arbitrage(
        strikes, call_prices, put_prices, butterfly_tolerance
    )
    
    monotonicity_results = check_price_monotonicity(
        strikes, call_prices, put_prices
    )
    
    parity_violations = check_call_put_parity(
        strikes, call_prices, put_prices, forward_price, discount_factor, parity_tolerance
    )
    
    # Count total violations
    total_violations = (
        len(butterfly_violations) + 
        len(monotonicity_results['call_violations']) +
        len(monotonicity_results['put_violations']) +
        len(parity_violations)
    )
    
    # Summary
    is_arbitrage_free = total_violations == 0
    
    return {
        'is_arbitrage_free': is_arbitrage_free,
        'total_violations': total_violations,
        'butterfly_arbitrage': {
            'violations': butterfly_violations,
            'count': len(butterfly_violations)
        },
        'monotonicity': monotonicity_results,
        'call_put_parity': {
            'violations': parity_violations,
            'count': len(parity_violations)
        },
        'summary': {
            'num_strikes': len(strikes),
            'strike_range': (min(strikes), max(strikes)),
            'call_price_range': (min(call_prices), max(call_prices)),
            'put_price_range': (min(put_prices), max(put_prices))
        }
    }


def format_arbitrage_report(analysis_results: Dict[str, Any]) -> str:
    """
    Format arbitrage analysis results into a readable report.
    
    Args:
        analysis_results: Results from analyze_arbitrage_comprehensive
        
    Returns:
        Formatted string report
    """
    report = []
    
    # Header
    status = "✅ ARBITRAGE-FREE" if analysis_results['is_arbitrage_free'] else "⚠️ ARBITRAGE VIOLATIONS DETECTED"
    report.append(f"🔍 ARBITRAGE ANALYSIS REPORT: {status}")
    report.append("=" * 60)
    
    # Summary
    summary = analysis_results['summary']
    report.append(f"📊 Summary: {summary['num_strikes']} strikes analyzed")
    report.append(f"   Strike range: {summary['strike_range'][0]:.0f} - {summary['strike_range'][1]:.0f}")
    report.append(f"   Total violations: {analysis_results['total_violations']}")
    report.append("")
    
    # Butterfly arbitrage
    butterfly = analysis_results['butterfly_arbitrage']
    report.append(f"🦋 Butterfly Arbitrage: {butterfly['count']} violations")
    if butterfly['violations']:
        for i, violation in enumerate(butterfly['violations'][:3]):  # Show top 3
            k1, k2, k3 = violation['strikes']
            report.append(f"   {i+1}. {violation['type']} at {k1:.0f}-{k2:.0f}-{k3:.0f}: {violation['violation']:.6f}")
    
    # Monotonicity
    mono = analysis_results['monotonicity']
    call_status = "✅" if mono['call_monotonic'] else f"❌ ({len(mono['call_violations'])} violations)"
    put_status = "✅" if mono['put_monotonic'] else f"❌ ({len(mono['put_violations'])} violations)"
    report.append(f"📈 Price Monotonicity:")
    report.append(f"   Call prices (decreasing): {call_status}")
    report.append(f"   Put prices (increasing): {put_status}")
    
    # Call-put parity
    parity = analysis_results['call_put_parity']
    report.append(f"⚖️ Call-Put Parity: {parity['count']} violations")
    if parity['violations']:
        avg_violation = np.mean([v['violation'] for v in parity['violations']])
        max_violation = max([v['violation'] for v in parity['violations']])
        report.append(f"   Average violation: {avg_violation:.6f}")
        report.append(f"   Maximum violation: {max_violation:.6f}")
    
    return "\n".join(report)