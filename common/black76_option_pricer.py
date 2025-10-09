"""
Black-76 Option Pricing Model for Forward Contracts

This module provides a comprehensive implementation of the Black-76 option pricing model,
specifically designed for European options on forward contracts. The Black-76 model is
widely used in commodities and interest rate derivatives markets.

Key Features:
- Full option pricing for calls and puts
- Complete Greeks calculation suite (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility calculation using Newton-Raphson method
- Vectorized operations for efficient computation of option chains
- Put-call parity verification utilities

Mathematical Foundation:
The Black-76 model extends Black-Scholes for forward contracts:
- Call Price: e^(-rT) * [F*N(d1) - K*N(d2)]
- Put Price: e^(-rT) * [K*N(-d2) - F*N(-d1)]
- d1 = [ln(F/K) + 0.5*σ²*T] / (σ*√T)
- d2 = d1 - σ*√T

Author: GitHub Copilot
Date: October 2025
"""

import numpy as np
from scipy.stats import norm
from scipy.special import ndtr


class Black76OptionPricer:
    """
    Black-76 option pricing model for options on forward contracts.
    
    This class provides pricing and Greeks calculation for European options
    on forward contracts using the Black-76 model.
    
    Key Features:
    - Call and Put option pricing
    - All Greeks: Delta, Gamma, Theta, Vega, Rho
    - Forward Delta and Gamma
    - Vectorized operations for efficient computation
    - Implied volatility calculation using Newton-Raphson
    
    Examples:
    --------
    >>> # Single option pricing
    >>> pricer = Black76OptionPricer(F=100, K=105, T=0.25, r=0.05, sigma=0.25)
    >>> call_price = pricer.call_price()
    >>> greeks = pricer.get_all_greeks('call')
    
    >>> # Vectorized option chain
    >>> strikes = np.array([90, 95, 100, 105, 110])
    >>> chain_pricer = Black76OptionPricer(F=100, K=strikes, T=0.25, r=0.05, sigma=0.25)
    >>> call_prices = chain_pricer.call_price()
    """
    
    def __init__(self, F, K, T, r, sigma):
        """
        Initialize the Black-76 option pricer.
        
        Parameters:
        -----------
        F : float or array-like
            Forward price
        K : float or array-like
            Strike price
        T : float or array-like
            Time to expiration (in years)
        r : float or array-like
            Risk-free interest rate (for discounting)
        sigma : float or array-like
            Volatility (annualized)
        """
        self.F = np.asarray(F, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        
        # Calculate d1 and d2 once for efficiency
        self._calculate_d_params()
    
    def _calculate_d_params(self):
        """Calculate d1 and d2 parameters used in Black-76 formulas."""
        sqrt_T = np.sqrt(self.T)
        sigma_sqrt_T = self.sigma * sqrt_T
        
        self.d1 = (np.log(self.F / self.K) + 0.5 * self.sigma**2 * self.T) / sigma_sqrt_T
        self.d2 = self.d1 - sigma_sqrt_T
        
        # Cache normal distribution values for efficiency
        self.N_d1 = ndtr(self.d1)
        self.N_d2 = ndtr(self.d2)
        self.N_neg_d1 = ndtr(-self.d1)
        self.N_neg_d2 = ndtr(-self.d2)
        
        # Cache normal density values
        self.n_d1 = norm.pdf(self.d1)
        self.n_d2 = norm.pdf(self.d2)
    
    def call_price(self):
        """
        Calculate Black-76 call option price.
        
        Formula: e^(-rT) * [F*N(d1) - K*N(d2)]
        
        Returns:
        --------
        float or numpy.ndarray
            Call option price(s)
        """
        discount_factor = np.exp(-self.r * self.T)
        return discount_factor * (self.F * self.N_d1 - self.K * self.N_d2)
    
    def put_price(self):
        """
        Calculate Black-76 put option price.
        
        Formula: e^(-rT) * [K*N(-d2) - F*N(-d1)]
        
        Returns:
        --------
        float or numpy.ndarray
            Put option price(s)
        """
        discount_factor = np.exp(-self.r * self.T)
        return discount_factor * (self.K * self.N_neg_d2 - self.F * self.N_neg_d1)
    
    def call_delta(self):
        """
        Calculate call option delta (∂C/∂S).
        Note: For forwards, this is the sensitivity to the underlying spot price.
        
        Returns:
        --------
        float or numpy.ndarray
            Call delta
        """
        return np.exp(-self.r * self.T) * self.N_d1
    
    def put_delta(self):
        """
        Calculate put option delta (∂P/∂S).
        
        Returns:
        --------
        float or numpy.ndarray
            Put delta
        """
        return -np.exp(-self.r * self.T) * self.N_neg_d1
    
    def gamma(self):
        """
        Calculate option gamma (∂²V/∂S²).
        Same for calls and puts.
        
        Returns:
        --------
        float or numpy.ndarray
            Option gamma
        """
        discount_factor = np.exp(-self.r * self.T)
        return discount_factor * self.n_d1 / (self.F * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """
        Calculate option vega (∂V/∂σ).
        Same for calls and puts.
        
        Returns:
        --------
        float or numpy.ndarray
            Option vega
        """
        discount_factor = np.exp(-self.r * self.T)
        return discount_factor * self.F * self.n_d1 * np.sqrt(self.T)
    
    def call_theta(self):
        """
        Calculate call option theta (∂C/∂T).
        
        Returns:
        --------
        float or numpy.ndarray
            Call theta (negative for time decay)
        """
        discount_factor = np.exp(-self.r * self.T)
        sqrt_T = np.sqrt(self.T)
        
        term1 = -discount_factor * self.F * self.n_d1 * self.sigma / (2 * sqrt_T)
        term2 = self.r * discount_factor * (self.F * self.N_d1 - self.K * self.N_d2)
        
        return term1 + term2
    
    def put_theta(self):
        """
        Calculate put option theta (∂P/∂T).
        
        Returns:
        --------
        float or numpy.ndarray
            Put theta (negative for time decay)
        """
        discount_factor = np.exp(-self.r * self.T)
        sqrt_T = np.sqrt(self.T)
        
        term1 = -discount_factor * self.F * self.n_d1 * self.sigma / (2 * sqrt_T)
        term2 = self.r * discount_factor * (self.K * self.N_neg_d2 - self.F * self.N_neg_d1)
        
        return term1 + term2
    
    def call_rho(self):
        """
        Calculate call option rho (∂C/∂r).
        
        Returns:
        --------
        float or numpy.ndarray
            Call rho
        """
        call_price = self.call_price()
        return -self.T * call_price
    
    def put_rho(self):
        """
        Calculate put option rho (∂P/∂r).
        
        Returns:
        --------
        float or numpy.ndarray
            Put rho
        """
        put_price = self.put_price()
        return -self.T * put_price
    
    def get_all_greeks(self, option_type='call'):
        """
        Get all Greeks for the specified option type.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        dict
            Dictionary containing all Greeks and option price
            Keys: 'price', 'delta', 'gamma', 'theta', 'vega', 'rho'
        """
        if option_type.lower() == 'call':
            return {
                'price': self.call_price(),
                'delta': self.call_delta(),
                'gamma': self.gamma(),
                'theta': self.call_theta(),
                'vega': self.vega(),
                'rho': self.call_rho()
            }
        elif option_type.lower() == 'put':
            return {
                'price': self.put_price(),
                'delta': self.put_delta(),
                'gamma': self.gamma(),
                'theta': self.put_theta(),
                'vega': self.vega(),
                'rho': self.put_rho()
            }
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def implied_volatility(self, target_price, option_type='call', max_iterations=200, precision=1e-5):
        """
        Calculate implied volatility using Newton-Raphson method.
        
        This method uses an iterative approach to find the volatility that produces
        the target market price when input into the Black-76 pricing formula.
        
        Parameters:
        -----------
        target_price : float or array-like
            Market price of the option
        option_type : str
            'call' or 'put'
        max_iterations : int
            Maximum number of iterations
        precision : float
            Convergence precision
        
        Returns:
        --------
        float or numpy.ndarray
            Implied volatility
            
        Raises:
        -------
        ValueError
            If option_type is not 'call' or 'put'
        """
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
            
        target_price = np.asarray(target_price, dtype=float)
        
        # Determine if we have scalar or array inputs
        is_scalar = target_price.ndim == 0
        
        # Ensure we work with arrays internally
        if is_scalar:
            target_price = np.array([target_price])
        
        # Initial guess for volatility
        sigma_guess = np.full_like(target_price, 0.5, dtype=float)
        
        for i in range(max_iterations):
            # Create new pricer with current volatility guess
            temp_pricer = Black76OptionPricer(self.F, self.K, self.T, self.r, sigma_guess)
            
            # Calculate price and vega
            if option_type.lower() == 'call':
                price = temp_pricer.call_price()
            else:
                price = temp_pricer.put_price()
            
            vega = temp_pricer.vega()
            
            # Calculate price difference
            diff = target_price - price
            
            # Check convergence
            if np.all(np.abs(diff) < precision):
                break
            
            # Newton-Raphson update (only where vega is not zero)
            mask = (vega != 0) & np.isfinite(vega) & np.isfinite(diff)
            sigma_guess = np.where(mask, sigma_guess + diff / vega, sigma_guess)
            
            # Ensure volatility stays positive
            sigma_guess = np.maximum(sigma_guess, 1e-6)
        
        # Return scalar if input was scalar, otherwise return array
        return float(sigma_guess[0]) if is_scalar else sigma_guess
    
    def update_parameters(self, **kwargs):
        """
        Update one or more parameters and recalculate d1, d2.
        
        This method allows dynamic updating of pricing parameters without
        creating a new instance of the class.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameters to update (F, K, T, r, sigma)
            
        Raises:
        -------
        ValueError
            If an invalid parameter name is provided
        """
        valid_params = {'F', 'K', 'T', 'r', 'sigma'}
        
        for param, value in kwargs.items():
            if param not in valid_params:
                raise ValueError(f"Invalid parameter: {param}. Valid parameters are: {valid_params}")
            setattr(self, param, np.asarray(value, dtype=float))
        
        # Recalculate d parameters
        self._calculate_d_params()
    
    def verify_put_call_parity(self):
        """
        Verify put-call parity relationship.
        
        For the Black-76 model, put-call parity is:
        C - P = e^(-rT) * (F - K)
        
        Returns:
        --------
        dict
            Dictionary with parity check results
        """
        call_price = self.call_price()
        put_price = self.put_price()
        
        actual_parity = call_price - put_price
        theoretical_parity = np.exp(-self.r * self.T) * (self.F - self.K)
        difference = np.abs(actual_parity - theoretical_parity)
        
        return {
            'call_minus_put': actual_parity,
            'theoretical_parity': theoretical_parity,
            'difference': difference,
            'max_difference': np.max(difference) if difference.ndim > 0 else difference,
            'parity_holds': np.allclose(actual_parity, theoretical_parity, atol=1e-10)
        }
    
    def __repr__(self):
        """String representation of the pricer."""
        return (f"Black76OptionPricer(F={self.F}, K={self.K}, T={self.T}, "
                f"r={self.r}, sigma={self.sigma})")


# Convenience functions for backward compatibility and simple calculations
def black76_call(F, K, T, r, vol):
    """
    Convenience function for Black-76 call pricing.
    
    Parameters:
    -----------
    F : float or array-like
        Forward price
    K : float or array-like
        Strike price
    T : float or array-like
        Time to expiration (years)
    r : float or array-like
        Risk-free rate
    vol : float or array-like
        Volatility
        
    Returns:
    --------
    float or numpy.ndarray
        Call option price(s)
    """
    pricer = Black76OptionPricer(F, K, T, r, vol)
    return pricer.call_price()


def black76_put(F, K, T, r, vol):
    """
    Convenience function for Black-76 put pricing.
    
    Parameters:
    -----------
    F : float or array-like
        Forward price
    K : float or array-like
        Strike price
    T : float or array-like
        Time to expiration (years)
    r : float or array-like
        Risk-free rate
    vol : float or array-like
        Volatility
        
    Returns:
    --------
    float or numpy.ndarray
        Put option price(s)
    """
    pricer = Black76OptionPricer(F, K, T, r, vol)
    return pricer.put_price()


def black76_vega(F, K, T, r, sigma):
    """
    Convenience function for Black-76 vega calculation.
    
    Parameters:
    -----------
    F : float or array-like
        Forward price
    K : float or array-like
        Strike price
    T : float or array-like
        Time to expiration (years)
    r : float or array-like
        Risk-free rate
    sigma : float or array-like
        Volatility
        
    Returns:
    --------
    float or numpy.ndarray
        Option vega
    """
    pricer = Black76OptionPricer(F, K, T, r, sigma)
    return pricer.vega()


def find_implied_volatility(target_value, F, K, T, r, option_type='C', **kwargs):
    """
    Vectorized implied volatility solver using Newton-Raphson method with Black-76 model.
    
    All arguments can be numpy arrays, scalars, or other array-like objects.
    Returns array of implied volatilities or scalar if all inputs are scalar.
    
    Parameters:
    -----------
    target_value : array-like or scalar
        Option market prices
    F : array-like or scalar  
        Forward prices
    K : array-like or scalar
        Strike prices
    T : array-like or scalar
        Time to expiration (in years)
    r : array-like or scalar
        Risk-free interest rate (for discounting)
    option_type : str, optional
        'C' for call (default), 'P' for put
    max_iterations : int, optional
        Maximum iterations for Newton-Raphson (default: 200)
    precision : float, optional
        Convergence precision (default: 1e-5)
    
    Returns:
    --------
    numpy.ndarray or float
        Implied volatilities
        
    Examples:
    --------
    >>> # Single option
    >>> iv = find_implied_volatility(5.0, 100, 105, 0.25, 0.05, 'C')
    
    >>> # Option chain
    >>> strikes = np.array([90, 95, 100, 105, 110])
    >>> prices = np.array([12.5, 8.3, 5.1, 2.8, 1.2])
    >>> ivs = find_implied_volatility(prices, 100, strikes, 0.25, 0.05, 'C')
    """
    MAX_ITERATIONS = kwargs.get('max_iterations', 200)
    PRECISION = kwargs.get('precision', 1e-5)
    
    # Convert all inputs to numpy arrays for vectorization
    target_value = np.asarray(target_value, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    
    # Get the broadcasted shape
    try:
        shape = np.broadcast(target_value, F, K, T, r).shape
    except ValueError as e:
        raise ValueError(f"Input arrays could not be broadcast together: {e}")
    
    # Initialize volatility array
    sigmas = np.full(shape, 0.5, dtype=float)
    done = np.zeros(shape, dtype=bool)
    
    # Newton-Raphson iterations
    for iteration in range(MAX_ITERATIONS):
        # Calculate option prices and vegas using Black-76
        if option_type.upper() == 'P':
            prices = black76_put(F, K, T, r, sigmas)
        else:
            prices = black76_call(F, K, T, r, sigmas)
        
        vegas = black76_vega(F, K, T, r, sigmas)
        
        # Calculate price differences
        diff = target_value - prices
        
        # Check convergence
        converged = np.abs(diff) < PRECISION
        
        # Only update where not converged and vega is not zero
        update_mask = ~done & (vegas != 0) & np.isfinite(vegas) & np.isfinite(diff)
        
        # Newton-Raphson update - use np.where for safe assignment
        sigmas = np.where(update_mask, sigmas + diff / vegas, sigmas)
        
        # Ensure volatility stays positive
        sigmas = np.maximum(sigmas, 1e-6)
        
        # Mark converged elements as done
        done = done | converged
        
        # Early exit if all converged
        if np.all(done):
            break
    
    # Return scalar if input was scalar, otherwise return array
    if sigmas.shape == ():
        return float(sigmas)
    else:
        return sigmas


# Alias for backward compatibility with notebook code
find_vol = find_implied_volatility