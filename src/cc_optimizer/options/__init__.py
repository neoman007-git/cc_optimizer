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
    is_hist_vol_model_available: Check if hist_vol_model is installed
    list_available_tokens: List tokens available in hist_vol_model
    get_hist_vol_model_info: Get hist_vol_model installation information
"""

from .black_scholes import BlackScholesModel, bs_pricer
from .greeks import Greeks, calculate_greeks
from .iv_helpers import (
    is_hist_vol_model_available,
    list_available_tokens,
    get_hist_vol_model_info,
    compute_iv_with_cache,
    clear_iv_cache,
    get_iv_cache_info
)

__all__ = [
    'BlackScholesModel',
    'Greeks',
    'bs_pricer',
    'calculate_greeks',
    'is_hist_vol_model_available',
    'list_available_tokens',
    'get_hist_vol_model_info',
    'compute_iv_with_cache',
    'clear_iv_cache',
    'get_iv_cache_info',
]
