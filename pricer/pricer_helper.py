# Convenience functions for backward compatibility and simple calculations
import numpy as np

from pricer.black76_option_pricer import Black76OptionPricer


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
    option_type : str or array-like, optional
        'C' for call, 'P' for put. Can be a single string (default: 'C') 
        or an array of strings for vectorized operation
    max_iterations : int, optional
        Maximum iterations for Newton-Raphson (default: 200)
    precision : float, optional
        Convergence precision (default: 1e-5)
    initial_vola : float, optional
        Initial volatility guess (default: 0.5)
    min_time_hours : float, optional
        Minimum time to expiration in hours for stability (default: 2.0)
    vol_bounds : tuple, optional
        Volatility bounds as (min, max) (default: (0.01, 5.0))
    
    Damping Parameters:
    ------------------
    base_damping : float, optional
        Base damping factor (default: 0.7)
    extreme_moneyness_damping : float, optional
        Damping for extreme moneyness options (default: 0.3)
    short_expiry_damping : float, optional
        Damping for very short expiry options (default: 0.2)
    large_update_damping : float, optional
        Damping for large volatility updates (default: 0.4)
    moneyness_bounds : tuple, optional
        Moneyness bounds for extreme damping as (min, max) (default: (0.5, 2.0))
    short_expiry_days : float, optional
        Days threshold for short expiry damping (default: 7.0)
    max_update_per_iter : float, optional
        Maximum volatility change per iteration (default: 0.3)
    large_update_threshold : float, optional
        Threshold for large update damping (default: 0.5)
    min_iteration_factor : float, optional
        Minimum progressive damping factor (default: 0.3)
    
    Returns:
    --------
    numpy.ndarray or float
        Implied volatilities
        
    Examples:
    --------
    >>> # Single option
    >>> iv = find_implied_volatility(5.0, 100, 105, 0.25, 0.05, 'C')
    
    >>> # Option chain with same option type
    >>> strikes = np.array([90, 95, 100, 105, 110])
    >>> prices = np.array([12.5, 8.3, 5.1, 2.8, 1.2])
    >>> ivs = find_implied_volatility(prices, 100, strikes, 0.25, 0.05, 'C')
    
    >>> # Mixed option types
    >>> option_types = np.array(['C', 'C', 'P', 'P', 'C'])
    >>> ivs = find_implied_volatility(prices, 100, strikes, 0.25, 0.05, option_types)
    
    >>> # Custom damping parameters
    >>> ivs = find_implied_volatility(prices, 100, strikes, 0.25, 0.05, 'C',
    ...                              base_damping=0.8, 
    ...                              extreme_moneyness_damping=0.2,
    ...                              max_update_per_iter=0.2)
    """
    # Main algorithm parameters
    MAX_ITERATIONS = kwargs.get('max_iterations', 200)
    PRECISION = kwargs.get('precision', 1e-5)
    INITIAL_VOLA = kwargs.get('initial_vola', 0.5)
    MIN_TIME_HOURS = kwargs.get('min_time_hours', 2.0)
    VOL_BOUNDS = kwargs.get('vol_bounds', (0.01, 5.0))
    MIN_VEGA = kwargs.get('min_vega', 1e-4)  # floor to avoid division by 0 vega
    
    # Damping parameters
    BASE_DAMPING = kwargs.get('base_damping', 0.7)
    EXTREME_MONEYNESS_DAMPING = kwargs.get('extreme_moneyness_damping', 0.3)
    SHORT_EXPIRY_DAMPING = kwargs.get('short_expiry_damping', 0.2)
    LARGE_UPDATE_DAMPING = kwargs.get('large_update_damping', 0.4)
    MONEYNESS_BOUNDS = kwargs.get('moneyness_bounds', (0.5, 2.0))
    SHORT_EXPIRY_DAYS = kwargs.get('short_expiry_days', 7.0)
    MAX_UPDATE_PER_ITER = kwargs.get('max_update_per_iter', 0.3)
    LARGE_UPDATE_THRESHOLD = kwargs.get('large_update_threshold', 0.5)
    MIN_ITERATION_FACTOR = kwargs.get('min_iteration_factor', 0.3)
    
    # Convert all inputs to numpy arrays for vectorization
    target_value = np.asarray(target_value, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    
    # Check for null/NaN target prices and handle early return
    is_nan_input = ~np.isfinite(target_value) | (target_value <= 0)
    
    # Additional checks for numerical stability
    # Check for very short time to expiration using parameterized threshold
    is_too_short = T < (MIN_TIME_HOURS/(365*24))  # Convert hours to years
    
    # Combine all invalid conditions
    is_invalid = is_nan_input | is_too_short
    
    # If all inputs are invalid, return NaN array of same shape
    if np.all(is_invalid):
        return np.full_like(target_value, np.nan)
    
    # Handle option_type - can be string or array of strings
    if isinstance(option_type, str):
        # Single option type - broadcast to match other arrays
        option_type_array = None
        is_single_option_type = True
    else:
        # Array of option types
        option_type_array = np.asarray(option_type, dtype=str)
        is_single_option_type = False
    
    # Get the broadcasted shape
    try:
        if is_single_option_type:
            shape = np.broadcast(target_value, F, K, T, r).shape
        else:
            shape = np.broadcast(target_value, F, K, T, r, option_type_array).shape
    except ValueError as e:
        raise ValueError(f"Input arrays could not be broadcast together: {e}")
    
    # Initialize volatility array with reasonable starting guess
    sigmas = np.full(shape, INITIAL_VOLA, dtype=float)
    done = np.zeros(shape, dtype=bool)
    
    # Mark invalid inputs as done (skip processing)
    done = done | is_invalid
    
    # Newton-Raphson iterations
    for iteration in range(MAX_ITERATIONS):
        # Calculate option prices and vegas using Black-76
        if is_single_option_type:
            # Single option type for all options
            if option_type.upper() == 'P':
                prices = black76_put(F, K, T, r, sigmas)
            else:
                prices = black76_call(F, K, T, r, sigmas)
        else:
            # Mixed option types - calculate calls and puts separately
            # Broadcast option_type_array to match sigmas shape
            option_type_broadcast = np.broadcast_to(option_type_array, shape)
            
            # Create masks for calls and puts
            is_put = (option_type_broadcast == 'P')
            is_call = ~is_put
            
            # Initialize prices array
            prices = np.zeros_like(sigmas)
            
            # Calculate put prices where needed
            if np.any(is_put):
                prices = np.where(is_put, black76_put(F, K, T, r, sigmas), prices)
            
            # Calculate call prices where needed
            if np.any(is_call):
                prices = np.where(is_call, black76_call(F, K, T, r, sigmas), prices)
        
        vegas = black76_vega(F, K, T, r, sigmas)
        
        # Calculate price differences
        diff = target_value - prices
        
        # Check convergence
        converged = np.abs(diff) < PRECISION
        
        # Only update where not converged, vega is not zero, and input is valid
        update_mask = ~done & (vegas != 0) & np.isfinite(vegas) & np.isfinite(diff) & ~is_invalid
        
        # Newton-Raphson update - use np.where for safe assignment
        delta_sigma = diff / np.maximum(vegas, MIN_VEGA)
        
        # Enhanced adaptive damping logic
        # Calculate moneyness for adaptive damping
        moneyness = F / K
        
        # Base damping factors based on option characteristics
        base_damping = np.full_like(sigmas, BASE_DAMPING)  # Use parameterized base damping
        
        # More conservative damping for extreme cases using parameterized bounds
        extreme_moneyness = (moneyness < MONEYNESS_BOUNDS[0]) | (moneyness > MONEYNESS_BOUNDS[1])
        very_short_expiry = T < (SHORT_EXPIRY_DAYS/(365*24))  # Convert days to years
        large_update = np.abs(delta_sigma) > LARGE_UPDATE_THRESHOLD
        
        # Apply different damping factors using parameterized values
        base_damping = np.where(extreme_moneyness, EXTREME_MONEYNESS_DAMPING, base_damping)
        base_damping = np.where(very_short_expiry, SHORT_EXPIRY_DAMPING, base_damping)
        base_damping = np.where(large_update, LARGE_UPDATE_DAMPING, base_damping)
        
        # Progressive damping - reduce damping as iterations progress
        iteration_factor = max(MIN_ITERATION_FACTOR, 1.0 - iteration / MAX_ITERATIONS)
        adaptive_damping = base_damping * iteration_factor
        
        # Limit maximum update size per iteration using parameterized value
        clamped_delta = np.clip(delta_sigma, -MAX_UPDATE_PER_ITER, MAX_UPDATE_PER_ITER)
        
        # Apply adaptive damping
        damped_delta = adaptive_damping * clamped_delta
        
        sigmas = np.where(update_mask, sigmas + damped_delta, sigmas)
        
        # Ensure volatility stays within reasonable bounds using parameterized bounds
        sigmas = np.clip(sigmas, VOL_BOUNDS[0], VOL_BOUNDS[1])
        
        # Mark converged elements as done
        done = done | converged
        
        # Early exit if all converged
        if np.all(done):
            break
    
    # Set NaN for invalid input prices
    sigmas = np.where(is_invalid, np.nan, sigmas)
    
    # Return scalar if input was scalar, otherwise return array
    if sigmas.shape == ():
        return float(sigmas)
    else:
        return sigmas


# Alias for backward compatibility with notebook code
find_vol = find_implied_volatility