"""
Options pricing module for cc_optimizer.

This module provides vanilla European option pricing using the Black-Scholes model,
along with Greeks calculations for risk management.

Classes:
    BlackScholesModel: Core Black-Scholes pricing model
    Greeks: Greeks calculator (delta, gamma, theta, vega)

Functions:
    bs_pricer: Simple functional interface for pricing
    calculate_greeks: Calculate all Greeks at once
"""

from .black_scholes import BlackScholesModel, bs_pricer
from .greeks import Greeks, calculate_greeks

__all__ = [
    'BlackScholesModel',
    'Greeks',
    'bs_pricer',
    'calculate_greeks',
]
