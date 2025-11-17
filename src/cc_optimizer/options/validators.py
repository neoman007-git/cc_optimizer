"""
Input validation utilities for option pricing.

Validates option pricing parameters to ensure they are within acceptable ranges
and of correct types.
"""

import numpy as np
from typing import Union


def validate_positive(value: float, name: str) -> None:
    """
    Validate that a value is positive.

    Args:
        value: The value to validate
        name: The parameter name (for error messages)

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """
    Validate that a value is non-negative.

    Args:
        value: The value to validate
        name: The parameter name (for error messages)

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_option_type(option_type: str) -> str:
    """
    Validate and normalize option type.

    Args:
        option_type: Option type ('call', 'put', 'c', 'p')

    Returns:
        Normalized option type ('call' or 'put')

    Raises:
        ValueError: If option_type is invalid
    """
    option_type_lower = option_type.lower().strip()

    if option_type_lower in ('call', 'c'):
        return 'call'
    elif option_type_lower in ('put', 'p'):
        return 'put'
    else:
        raise ValueError(
            f"option_type must be 'call', 'put', 'c', or 'p', got '{option_type}'"
        )


def validate_bs_parameters(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str
) -> dict:
    """
    Validate all Black-Scholes parameters.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Dictionary with validated and normalized parameters

    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameters are not numeric
    """
    # Type checking
    for param, name in [(S, 'S'), (K, 'K'), (T, 'T'), (r, 'r'), (sigma, 'sigma')]:
        if not isinstance(param, (int, float, np.number)):
            raise TypeError(f"{name} must be numeric, got {type(param).__name__}")

    # Check for NaN or Inf
    for param, name in [(S, 'S'), (K, 'K'), (T, 'T'), (r, 'r'), (sigma, 'sigma')]:
        if np.isnan(param) or np.isinf(param):
            raise ValueError(f"{name} must be finite, got {param}")

    # Validate ranges
    validate_positive(S, 'S (spot price)')
    validate_positive(K, 'K (strike price)')
    validate_non_negative(T, 'T (time to expiration)')
    validate_positive(sigma, 'sigma (volatility)')

    # Normalize option type
    normalized_type = validate_option_type(option_type)

    return {
        'S': float(S),
        'K': float(K),
        'T': float(T),
        'r': float(r),
        'sigma': float(sigma),
        'option_type': normalized_type
    }
