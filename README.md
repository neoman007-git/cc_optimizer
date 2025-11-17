# CC Optimizer

Cryptocurrency optimization tools and analytics, featuring a professional-grade option pricer with Greeks calculations.

## Features

### 🎯 Option Pricer
- **Black-Scholes Pricing**: European vanilla calls and puts
- **Complete Greeks**: Delta, Gamma, Theta, Vega, Rho
- **Dual API**: Simple functions + Advanced classes
- **Crypto-Ready**: High volatility, zero rates, short tenors
- **Validated**: 57 tests, verified against py_vollib
- **No Heavy Dependencies**: Pure NumPy/SciPy implementation

## Quick Start

```python
from cc_optimizer.options import bs_pricer, calculate_greeks

# Price a BTC call option
price = bs_pricer(
    S=50000,           # BTC spot
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
print(f"Theta: ${greeks['theta']:.2f} per day")
print(f"Vega: ${greeks['vega']:.2f} per 1% vol")
```

## Setup

1. Create virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install package in editable mode:
   ```bash
   pip install -e .
   ```

## Documentation

📚 **[Complete Option Pricer Guide](docs/option_pricer_guide.md)** - Comprehensive guide with examples

📓 **[Example Notebook](notebooks/option_pricer_example.ipynb)** - Interactive examples with visualizations

📖 **[API Reference](docs/option_pricer_guide.md#api-reference)** - Complete API documentation

## Project Structure

```
cc_optimizer/
├── src/cc_optimizer/         # Main package code
│   └── options/              # Option pricing module ⭐
│       ├── black_scholes.py  # Core BS model
│       ├── greeks.py         # Greeks calculator
│       └── validators.py     # Input validation
├── tests/                    # Test files (57 tests, 100% pass)
│   ├── test_black_scholes.py
│   └── test_greeks.py
├── data/                     # Data files
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
├── notebooks/                # Jupyter notebooks
│   └── option_pricer_example.ipynb
├── scripts/                  # Standalone scripts
├── config/                   # Configuration files
└── docs/                     # Documentation
    ├── README.md
    └── option_pricer_guide.md
```

## Usage Examples

### Basic Pricing
```python
from cc_optimizer.options import BlackScholesModel

# Create model
model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')

# Get pricing info
print(f"Price: ${model.price():.2f}")
print(f"Intrinsic Value: ${model.intrinsic_value():.2f}")
print(f"Time Value: ${model.time_value():.2f}")
print(f"Moneyness: {model.moneyness()}")
```

### Greeks Analysis
```python
from cc_optimizer.options import BlackScholesModel, Greeks

model = BlackScholesModel(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')
greeks = Greeks(model)

# Get all Greeks at once
all_greeks = greeks.all_greeks()
for name, value in all_greeks.items():
    print(f"{name}: {value:.4f}")
```

### Integration with hist_vol_model
```python
# Automatically fetch implied volatility
model = BlackScholesModel.from_iv_model(
    S=50000, K=50000, T=30/365, r=0.0,
    token='BTC', tenor_days=30, strike_percent=100.0,
    option_type='call'
)
```

## Development

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test suite
pytest tests/test_black_scholes.py -v
pytest tests/test_greeks.py -v
```

### Test Results
```
✅ 27 Black-Scholes pricing tests
✅ 30 Greeks calculation tests
✅ 57 total tests - 100% pass rate
✅ Validated against py_vollib reference
```

## Roadmap

- [x] Black-Scholes option pricer
- [x] Complete Greeks calculations
- [x] Comprehensive test suite
- [x] Example notebooks
- [x] Full documentation
- [ ] Implied volatility solver
- [ ] Payoff diagrams and P&L visualization
- [ ] Advanced models (jump-diffusion, stochastic vol)
- [ ] Real-time market data integration

## License

[To be determined]
