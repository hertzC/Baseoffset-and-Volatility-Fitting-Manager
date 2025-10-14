import numpy as np


def convert_paramter_into_rate(parameter: np.ndarray, spot: float, tau: float) -> np.ndarray:
    # Calculate rates from regression parameters
    const, coef = parameter
    r = float(np.log(coef) / -tau)  # USD interest rate
    q = float(np.log(-const / spot) / -tau)  # BTC funding rate
    return np.array([r, q])


def convert_rate_into_parameter(rate: np.ndarray, spot: float, tau: float) -> np.ndarray:
    r, q = rate
    coef = np.exp(-r * tau)
    const = -spot * np.exp(-q * tau)
    return np.array([const, coef])