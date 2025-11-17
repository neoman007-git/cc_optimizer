"""
Backtesting Example: Using cc_optimizer with hist_vol_model for Strategy Backtesting

This example demonstrates how to use the integrated option pricer for backtesting
trading strategies. It shows:
1. Pricing a sequence of options (simulating a backtest)
2. Using caching for performance
3. Computing P&L and Greeks
4. Best practices for backtesting workflows

Author: Claude Code
Date: 2025-11-17
"""

import time
from typing import List, Dict
import numpy as np

from cc_optimizer.options import (
    BlackScholesModel,
    Greeks,
    clear_iv_cache,
    get_iv_cache_info,
    list_available_tokens
)


def price_option_sequence(
    token: str,
    spot_prices: List[float],
    strike_pct: float = 100.0,
    tenor_days: float = 30.0,
    option_type: str = 'call',
    quote_type: str = 'mid'
) -> List[Dict]:
    """
    Price a sequence of options (simulating a backtest).

    Args:
        token: Token symbol (e.g., 'HBAR')
        spot_prices: List of spot prices at different time points
        strike_pct: Strike as percentage of first spot price
        tenor_days: Option tenor in days
        option_type: 'call' or 'put'
        quote_type: 'bid', 'mid', or 'ask'

    Returns:
        List of dictionaries with pricing results
    """
    # Set strike based on initial spot
    K = spot_prices[0] * (strike_pct / 100.0)

    results = []
    for i, S in enumerate(spot_prices):
        # Price option at this spot level
        model = BlackScholesModel.from_iv_model(
            S=S,
            K=K,
            T=tenor_days/365,
            r=0.0,  # Zero rate for crypto
            token=token,
            tenor_days=tenor_days,
            strike_percent=strike_pct,
            option_type=option_type,
            quote_type=quote_type
        )

        # Calculate Greeks
        greeks = Greeks(model)

        # Store results
        results.append({
            'step': i,
            'spot': S,
            'strike': K,
            'price': model.price(),
            'iv': model.sigma,
            'delta': greeks.delta(),
            'gamma': greeks.gamma(),
            'theta': greeks.theta(),
            'vega': greeks.vega(),
            'intrinsic': model.intrinsic_value(),
            'time_value': model.time_value()
        })

    return results


def demonstrate_caching_performance():
    """Demonstrate the performance benefit of caching."""
    tokens = list_available_tokens()
    if not tokens:
        print("No tokens available in hist_vol_model")
        return

    token = tokens[0]
    print(f"\n{'='*60}")
    print(f"CACHING PERFORMANCE DEMONSTRATION (Token: {token})")
    print(f"{'='*60}\n")

    # Clear cache to start fresh
    clear_iv_cache()

    # Price 20 options with same parameters
    print("Pricing 20 options with identical parameters...")
    start = time.time()

    for i in range(20):
        model = BlackScholesModel.from_iv_model(
            S=1.0,
            K=1.0,
            T=30/365,
            r=0.0,
            token=token,
            tenor_days=30.0,
            strike_percent=100.0
        )
        _ = model.price()

    elapsed = time.time() - start

    # Check cache stats
    cache_info = get_iv_cache_info()

    print(f"✓ Completed in {elapsed:.3f} seconds")
    print(f"\nCache Statistics:")
    print(f"  Cache hits:   {cache_info['hits']}")
    print(f"  Cache misses: {cache_info['misses']}")
    print(f"  Cache size:   {cache_info['size']}/{cache_info['maxsize']}")
    print(f"  Hit rate:     {cache_info['hits']/(cache_info['hits']+cache_info['misses']):.1%}")
    print(f"\nPerformance: First call ~1-2s, subsequent calls ~microseconds")


def backtest_simple_strategy():
    """Backtest a simple options trading strategy."""
    tokens = list_available_tokens()
    if 'HBAR' not in tokens:
        print("HBAR data not available for backtesting")
        return

    print(f"\n{'='*60}")
    print(f"SIMPLE STRATEGY BACKTEST: ATM Call on HBAR")
    print(f"{'='*60}\n")

    # Clear cache for clean test
    clear_iv_cache()

    # Simulate spot price movement (10 time steps)
    initial_spot = 0.15
    spot_prices = [initial_spot * (1 + 0.02 * np.sin(i/2)) for i in range(10)]

    print(f"Strategy: Buy 30D ATM call at S={initial_spot:.4f}")
    print(f"Simulating {len(spot_prices)} time steps...\n")

    # Price options along the path
    start = time.time()
    results = price_option_sequence(
        token='HBAR',
        spot_prices=spot_prices,
        strike_pct=100.0,
        tenor_days=30.0,
        option_type='call',
        quote_type='mid'
    )
    elapsed = time.time() - start

    # Display results
    print(f"{'Step':<6} {'Spot':<10} {'Price':<10} {'Delta':<8} {'Gamma':<8} {'IV':<8}")
    print("-" * 60)

    for r in results:
        print(f"{r['step']:<6} "
              f"{r['spot']:<10.4f} "
              f"{r['price']:<10.6f} "
              f"{r['delta']:<8.4f} "
              f"{r['gamma']:<8.4f} "
              f"{r['iv']:<8.2%}")

    # Calculate P&L
    entry_price = results[0]['price']
    exit_price = results[-1]['price']
    pnl = exit_price - entry_price
    pnl_pct = (pnl / entry_price) * 100

    print(f"\n{'-' * 60}")
    print(f"Entry Price:  ${entry_price:.6f}")
    print(f"Exit Price:   ${exit_price:.6f}")
    print(f"P&L:          ${pnl:+.6f} ({pnl_pct:+.2f}%)")
    print(f"Computation:  {elapsed:.3f}s for {len(results)} pricings")

    # Cache stats
    cache_info = get_iv_cache_info()
    print(f"\nCache Efficiency: {cache_info['hits']} hits, {cache_info['misses']} misses")


def demonstrate_multi_strike_pricing():
    """Demonstrate pricing options across multiple strikes."""
    tokens = list_available_tokens()
    if not tokens:
        print("No tokens available")
        return

    token = tokens[0]
    print(f"\n{'='*60}")
    print(f"MULTI-STRIKE PRICING (Token: {token})")
    print(f"{'='*60}\n")

    clear_iv_cache()

    spot = 1.0
    strikes_pct = [90, 95, 100, 105, 110]  # Various strike levels

    print(f"Spot Price: ${spot:.2f}")
    print(f"Tenor: 30 days\n")
    print(f"{'Strike %':<12} {'Strike':<10} {'Call Price':<12} {'Put Price':<12} {'IV':<8}")
    print("-" * 60)

    start = time.time()

    for strike_pct in strikes_pct:
        K = spot * (strike_pct / 100)

        # Price call
        call = BlackScholesModel.from_iv_model(
            S=spot, K=K, T=30/365, r=0.0,
            token=token, tenor_days=30.0,
            strike_percent=strike_pct,
            option_type='call'
        )

        # Price put
        put = BlackScholesModel.from_iv_model(
            S=spot, K=K, T=30/365, r=0.0,
            token=token, tenor_days=30.0,
            strike_percent=strike_pct,
            option_type='put'
        )

        print(f"{strike_pct:<12} "
              f"{K:<10.2f} "
              f"${call.price():<11.4f} "
              f"${put.price():<11.4f} "
              f"{call.sigma:<8.2%}")

    elapsed = time.time() - start

    print(f"\n{'-' * 60}")
    print(f"Priced {len(strikes_pct)*2} options in {elapsed:.3f}s")

    cache_info = get_iv_cache_info()
    print(f"Cache: {cache_info['hits']} hits, {cache_info['misses']} misses")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CC_OPTIMIZER + HIST_VOL_MODEL INTEGRATION EXAMPLES")
    print("Backtesting and Sequential Option Pricing")
    print("="*60)

    # Check availability
    tokens = list_available_tokens()
    print(f"\nAvailable tokens: {', '.join(tokens)}")

    if not tokens:
        print("\nNo tokens available. Please ensure hist_vol_model has data.")
        return

    # Run examples
    try:
        demonstrate_caching_performance()
        backtest_simple_strategy()
        demonstrate_multi_strike_pricing()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. First IV computation takes ~1-2 seconds (full pipeline)")
    print("  2. Cached lookups are nearly instantaneous")
    print("  3. Cache holds 256 unique (token, tenor, strike, quote) combinations")
    print("  4. Perfect for backtesting where same params are repeated")
    print("  5. Call clear_iv_cache() to force recomputation")


if __name__ == "__main__":
    main()
