# CC Optimizer Documentation

Welcome to the CC Optimizer documentation!

## Available Guides

### [Option Pricer Complete Guide](option_pricer_guide.md)

Comprehensive guide to using the Black-Scholes option pricer for vanilla European options.

**Topics covered:**
- Quick start (30 seconds to first option price)
- Functional vs Class-Based API
- Greeks calculations (Delta, Gamma, Theta, Vega)
- Crypto option examples (BTC, ETH strategies)
- Integration with hist_vol_model
- Common use cases and strategies
- Tips & best practices
- Complete API reference
- Troubleshooting

**Perfect for:**
- Pricing call and put options
- Calculating option Greeks for risk management
- Analyzing option strategies (straddles, spreads, covered calls)
- Building volatility surfaces
- Hedging portfolios

## Quick Start

```python
from cc_optimizer.options import bs_pricer, calculate_greeks

# Price a Bitcoin call option
price = bs_pricer(
    S=50000,           # BTC spot price
    K=50000,           # Strike (ATM)
    T=30/365,          # 30 days
    r=0.0,             # No risk-free rate
    sigma=0.8,         # 80% volatility
    option_type='call'
)

print(f"Option price: ${price:,.2f}")

# Calculate Greeks
greeks = calculate_greeks(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')
print(f"Delta: {greeks['delta']:.4f}")
print(f"Vega: ${greeks['vega']:.2f} per 1% vol")
```

## Examples

See the [example notebook](../notebooks/option_pricer_example.ipynb) for interactive examples with visualizations.

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_black_scholes.py -v
pytest tests/test_greeks.py -v
```

## Project Structure

```
cc_optimizer/
├── src/cc_optimizer/
│   └── options/           # Option pricing module
│       ├── black_scholes.py    # Core BS model
│       ├── greeks.py           # Greeks calculator
│       └── validators.py       # Input validation
├── tests/                 # Test suite (57 tests)
├── notebooks/             # Example notebooks
├── docs/                  # Documentation (you are here)
│   ├── README.md
│   └── option_pricer_guide.md
└── requirements.txt
```

## Features

✅ **Pure NumPy/SciPy** - No heavyweight dependencies
✅ **Dual API** - Simple functions + Advanced classes
✅ **Complete Greeks** - Delta, Gamma, Theta, Vega, Rho
✅ **Validated** - 57 tests, verified against py_vollib
✅ **Crypto-Ready** - High vol, zero rates, short tenors
✅ **Well-Documented** - Complete guide + examples

## Getting Help

1. **Read the guides** - Start with [option_pricer_guide.md](option_pricer_guide.md)
2. **Try the examples** - Open `notebooks/option_pricer_example.ipynb`
3. **Check docstrings** - Use `help(BlackScholesModel)` in Python
4. **Run the tests** - `pytest tests/ -v` to verify everything works

## Additional Resources

- **Source Code**: `src/cc_optimizer/options/`
- **Tests**: `tests/test_black_scholes.py`, `tests/test_greeks.py`
- **Example Notebook**: `notebooks/option_pricer_example.ipynb`
- **Project Info**: `README.md` (project root)

---

**Version**: 1.0
**Last Updated**: 2025-11-17
