# CC Optimizer - Claude Context

## Project Overview

**Project**: CC Optimizer
**Location**: `/Users/neo/Velar/cc_optimizer`
**Purpose**: Cryptocurrency optimization tools and analytics

**Current Features:**
- **Option Pricer**: Professional Black-Scholes pricer for vanilla European options
- **Greeks Calculator**: Complete risk metrics (Delta, Gamma, Theta, Vega, Rho)
- **Crypto-Ready**: Optimized for high volatility and cryptocurrency options
- **Validated**: 57 comprehensive tests, 100% pass rate

## Project Structure

```
cc_optimizer/
├── src/cc_optimizer/              # Main package code
│   ├── __init__.py
│   └── options/                   # Option pricing module ⭐
│       ├── __init__.py            # Package exports
│       ├── black_scholes.py       # Black-Scholes pricing model (402 lines)
│       ├── greeks.py              # Greeks calculator (341 lines)
│       └── validators.py          # Input validation (108 lines)
├── tests/                         # Test files (57 tests, 100% pass)
│   ├── test_black_scholes.py      # 27 pricing tests
│   └── test_greeks.py             # 30 Greeks tests
├── data/                          # Data files
│   ├── raw/                       # Raw data (e.g., historical crypto data)
│   └── processed/                 # Processed data
├── notebooks/                     # Jupyter notebooks for analysis
│   └── option_pricer_example.ipynb # Complete usage examples with visualizations
├── scripts/                       # Standalone scripts
├── config/                        # Configuration files
├── docs/                          # Documentation
│   ├── README.md                  # Documentation index
│   └── option_pricer_guide.md     # Complete option pricer guide (51 pages)
├── .venv/                         # Virtual environment
├── .claude/                       # Claude Code context
│   └── CLAUDE.md                  # This file
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── pyproject.toml                 # Project configuration
```

## Key Technologies

### Core Dependencies
- **Python**: Main programming language (>=3.9)
- **NumPy** (>=1.24.0): Numerical computations and option pricing math
- **SciPy** (>=1.10.0): Statistical distributions (norm.cdf, norm.pdf)
- **Pandas** (>=2.0.0): Data manipulation and analysis
- **Matplotlib** (>=3.7.0): Data visualization
- **Jupyter** (>=1.0.0): Interactive analysis notebooks
- **pytest** (>=7.0.0): Testing framework

### Development Dependencies
- **py_vollib** (>=1.0.0): Validation reference for option pricing

## Modules Overview

### 1. Option Pricer (`src/cc_optimizer/options/`)

A professional-grade Black-Scholes option pricing module with complete Greeks calculations.

#### Key Features
- ✅ European vanilla call and put options
- ✅ Dual API: Functional (simple) + Class-based (advanced)
- ✅ Complete Greeks: Delta, Gamma, Theta, Vega, Rho
- ✅ Crypto-optimized: High volatility, zero rates, short tenors
- ✅ Validated: All calculations verified against py_vollib
- ✅ Pure NumPy/SciPy: No heavyweight dependencies like QuantLib
- ✅ Extensible: Clean architecture for future models

#### Quick Usage

**Functional API (Simple):**
```python
from cc_optimizer.options import bs_pricer, calculate_greeks

# Price an option
price = bs_pricer(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')

# Calculate Greeks
greeks = calculate_greeks(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')
print(f"Delta: {greeks['delta']:.4f}")
print(f"Theta: ${greeks['theta']:.2f} per day")
```

**Class-Based API (Advanced):**
```python
from cc_optimizer.options import BlackScholesModel, Greeks

# Create model
model = BlackScholesModel(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')

# Get pricing details
print(f"Price: ${model.price():.2f}")
print(f"Intrinsic: ${model.intrinsic_value():.2f}")
print(f"Time Value: ${model.time_value():.2f}")
print(f"Moneyness: {model.moneyness()}")

# Calculate Greeks
greeks = Greeks(model)
all_greeks = greeks.all_greeks()
```

#### Integration with hist_vol_model

The option pricer is **fully integrated** with `hist_vol_model` for automatic implied volatility. This integration is production-ready and optimized for backtesting.

**Installation:**
```bash
# Install hist_vol_model as a package (one-time setup)
cd /Users/neo/Velar/hist_vol_model
pip install -e .
```

**Basic Usage:**
```python
from cc_optimizer.options import BlackScholesModel

# Automatic IV from hist_vol_model
model = BlackScholesModel.from_iv_model(
    S=0.15,           # Current HBAR price
    K=0.15,           # Strike price
    T=30/365,         # 30 days in years
    r=0.0,            # Zero rate for crypto
    token='HBAR',     # Token symbol
    tenor_days=30.0,  # 30 day tenor
    strike_percent=100.0,  # ATM
    option_type='call',
    iv_model='model_01',   # Use Model 01
    quote_type='mid'       # Mid-market IV
)

price = model.price()
```

**Available Tokens:**
```python
from cc_optimizer.options import list_available_tokens

tokens = list_available_tokens()
print(tokens)  # ['ENA', 'HBAR']
```

**Performance Optimization (Caching):**
```python
from cc_optimizer.options import (
    BlackScholesModel,
    get_iv_cache_info,
    clear_iv_cache
)

# Caching is automatic - first call computes, subsequent calls use cache
model1 = BlackScholesModel.from_iv_model(S=0.15, K=0.15, T=30/365, r=0.0, token='HBAR', tenor_days=30.0)
model2 = BlackScholesModel.from_iv_model(S=0.15, K=0.15, T=30/365, r=0.0, token='HBAR', tenor_days=30.0)
# Second call is ~1000x faster (cached)

# Check cache performance
info = get_iv_cache_info()
print(f"Hit rate: {info['hits']/(info['hits']+info['misses']):.1%}")

# Clear cache if needed (e.g., after data update)
clear_iv_cache()
```

**Backtesting Example:**
```python
# Price sequence of options (backtesting scenario)
spot_prices = [0.15, 0.152, 0.148, 0.150, 0.155]  # Time series
results = []

for S in spot_prices:
    model = BlackScholesModel.from_iv_model(
        S=S, K=0.15, T=30/365, r=0.0,
        token='HBAR', tenor_days=30.0
    )
    results.append({
        'spot': S,
        'price': model.price(),
        'delta': Greeks(model).delta()
    })

# Fast execution with caching!
```

**Integration Features:**
- ✅ **Clean imports**: No sys.path manipulation
- ✅ **Intelligent caching**: Data-driven automatic cache invalidation
- ✅ **4,000x speedup**: From ~600ms to ~0.05ms on cache hits
- ✅ **Auto-invalidation**: When data files or config changes
- ✅ **Helper utilities**: Check availability, list tokens, cache stats
- ✅ **Comprehensive tests**: 14 integration tests, 100% pass rate
- ✅ **Production ready**: Proper package installation
- ✅ **Backtesting optimized**: Perfect for sequential option pricing

**Enhanced Caching System:**
- **Data-driven invalidation**: Cache automatically invalidates when:
  - Data files are updated (new incremental data)
  - Config files are modified (parameter changes)
  - New tokens are added (separate cache entries)
- **Cache key includes**: (model, token, tenor, strike, quote, data_mtime, config_mtime)
- **Performance**: <1ms overhead for file stat checks (negligible)
- **Zero configuration**: Works automatically, no manual cache management needed
- **Production ready**: No stale cache issues, always uses latest data

**Helper Functions:**
- `is_hist_vol_model_available()`: Check if installed
- `list_available_tokens()`: List available tokens
- `get_hist_vol_model_info()`: Get installation info
- `compute_iv_with_cache()`: Cached IV lookup with auto-invalidation
- `clear_iv_cache()`: Clear IV cache (rarely needed)
- `get_iv_cache_info()`: Get cache statistics

**Example Scripts:**
- `examples/backtesting_example.py`: Complete backtesting demonstration
- `examples/cache_invalidation_demo.py`: Shows automatic cache invalidation in action

#### Module Files

**black_scholes.py:**
- `BlackScholesModel` class: Core pricing model
- `bs_pricer()` function: Simple functional interface
- Methods: `price()`, `intrinsic_value()`, `time_value()`, `moneyness()`
- Class method: `from_iv_model()` for hist_vol_model integration

**greeks.py:**
- `Greeks` class: Greeks calculator
- `calculate_greeks()` function: Simple functional interface
- Greeks: `delta()`, `gamma()`, `theta()`, `vega()`, `rho()`
- Method: `all_greeks()` for efficient batch calculation

**validators.py:**
- Input validation utilities
- Type checking and range validation
- Parameter normalization (option_type, units)

**iv_helpers.py:**
- hist_vol_model integration utilities with intelligent caching
- `is_hist_vol_model_available()`: Check installation
- `list_available_tokens()`: List available tokens
- `get_hist_vol_model_info()`: Get installation info
- `compute_iv_with_cache()`: Cached IV computation with auto-invalidation
- `_get_data_modification_time()`: Get data file mtime (for cache invalidation)
- `_get_config_modification_time()`: Get config file mtime (for cache invalidation)
- `_compute_iv_cached()`: LRU cache (256 entries) with data-driven invalidation
- `clear_iv_cache()`: Clear cache (rarely needed)
- `get_iv_cache_info()`: Cache statistics and hit rates

## Development Setup

### Virtual Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test suites
pytest tests/test_black_scholes.py -v  # 27 tests
pytest tests/test_greeks.py -v         # 30 tests

# With coverage
pytest tests/ --cov=src/cc_optimizer/options
```

**Test Results:**
- ✅ 27 Black-Scholes pricing tests (put-call parity, boundary conditions, edge cases)
- ✅ 30 Greeks tests (bounds, properties, validation against py_vollib)
- ✅ 57 total tests - 100% pass rate
- ✅ All calculations validated against py_vollib reference

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open example notebook
# notebooks/option_pricer_example.ipynb
```

## Data Management

- **Raw Data**: Store original/raw data files in `data/raw/`
- **Processed Data**: Store cleaned/processed data in `data/processed/`
- Data files are excluded from git by default (see `.gitignore`)

## Documentation

### Available Guides

1. **Complete Option Pricer Guide** (`docs/option_pricer_guide.md`)
   - 51-page comprehensive guide
   - Setup, quick start, parameter guide
   - Functional vs class-based API
   - Greeks calculations with interpretations
   - Crypto option examples (BTC, ETH strategies)
   - Integration with hist_vol_model
   - Common use cases and strategies
   - Tips, best practices, troubleshooting
   - Complete API reference

2. **Example Notebook** (`notebooks/option_pricer_example.ipynb`)
   - Interactive examples with visualizations
   - Greeks behavior across strikes and time
   - Volatility surface analysis
   - Real trading scenarios

3. **Project README** (`README.md`)
   - Quick start examples
   - Feature highlights
   - Usage examples
   - Development instructions

## Working with Claude Code

### General Guidelines

When working with this project:
1. Always activate the virtual environment
2. Keep data files in appropriate directories
3. Update documentation as the project evolves
4. Use notebooks for exploratory analysis
5. Move production code to `src/cc_optimizer/`
6. Run tests before committing changes
7. Follow the existing code style and patterns

### Option Pricer Specific

**When modifying option pricing code:**
1. Read the existing implementation in `src/cc_optimizer/options/`
2. Understand the Black-Scholes formulas and Greeks
3. Add tests for any new functionality
4. Validate against py_vollib if adding new calculations
5. Update documentation if changing APIs
6. Consider edge cases (T=0, extreme volatility, etc.)

**Parameter Conventions:**
- `S`: Spot price (must be positive)
- `K`: Strike price (must be positive)
- `T`: Time to expiration in **years** (not days!)
- `r`: Risk-free rate as decimal (0.05 = 5%)
- `sigma`: Volatility as decimal (0.8 = 80%)
- `option_type`: 'call', 'put', 'c', or 'p'

**Common Pitfalls:**
- ❌ `T=30` (30 YEARS) → ✅ `T=30/365` (30 days)
- ❌ `sigma=80` (8000% vol) → ✅ `sigma=0.80` (80% vol)
- ❌ `r=5` (500% rate) → ✅ `r=0.05` (5% rate)

## Project Status

### Completed Features
- [x] Initial project structure created
- [x] Virtual environment configured
- [x] Basic dependencies installed
- [x] **Black-Scholes option pricer implemented**
- [x] **Complete Greeks calculator implemented**
- [x] **Comprehensive test suite (71 tests: 57 pricing + 14 integration)**
- [x] **Functional and class-based APIs**
- [x] **Production-ready hist_vol_model integration**
- [x] **Intelligent caching with automatic invalidation (4,000x speedup)**
- [x] **Data-driven cache invalidation (auto-detects data/config changes)**
- [x] **Integration helper utilities (availability, token listing, cache management)**
- [x] **Backtesting-optimized architecture**
- [x] **Complete documentation (51 pages + integration guide)**
- [x] **Example Jupyter notebook with visualizations**
- [x] **Backtesting example script + cache invalidation demo**
- [x] **Validation against py_vollib reference**

### Roadmap

**Near-term (Next 1-2 months):**
- [ ] Implied volatility solver (Newton-Raphson method)
- [ ] Payoff diagrams and P&L visualization
- [x] ~~Batch pricing optimization~~ (Achieved via caching - 100-1000x speedup)
- [ ] Real-time market data integration (Deribit API)
- [ ] Compare BS prices with actual market prices
- [ ] Extend backtesting framework with more strategies

**Medium-term (3-6 months):**
- [ ] Advanced Greeks (Vanna, Volga, Charm, etc.)
- [ ] American option pricing (binomial tree)
- [ ] Jump-diffusion models (Merton, Kou)
- [ ] Stochastic volatility models (Heston)
- [ ] Greeks surfaces and risk analytics

**Long-term (6+ months):**
- [ ] Machine learning-based pricing corrections
- [ ] Multi-asset portfolio optimization
- [ ] Options strategy backtesting framework
- [ ] Real-time portfolio Greeks monitoring
- [ ] Integration with trading execution

## Related Projects

### hist_vol_model
**Location**: `/Users/neo/Velar/hist_vol_model/`
**Purpose**: Historical volatility and implied volatility modeling for cryptocurrency options
**Integration Status**: ✅ Production-ready, fully integrated as Python package
**Key Features**:
- Provides model_01: Percentile-based IV estimation from historical data
- Supports HBAR, ENA, and 50+ additional tokens
- Handles bid/mid/ask quotes, tenor interpolation, volatility smile
**Integration Details**:
- Installed via: `pip install -e /Users/neo/Velar/hist_vol_model`
- Accessed via: `BlackScholesModel.from_iv_model()` class method
- Cached for performance: 100-1000x speedup on repeated parameters
- Helper utilities: `list_available_tokens()`, `get_iv_cache_info()`, etc.
- See `examples/backtesting_example.py` for usage

## Notes

- Option pricer is production-ready for vanilla European options
- All calculations validated against industry-standard py_vollib library
- Designed for cryptocurrency options but works for any asset class
- Pure Python implementation avoids complex C++ dependencies (QuantLib)
- Extensible architecture allows easy addition of new pricing models
- Greeks are calculated analytically (exact formulas, not numerical approximations)

## Technical Details

### Black-Scholes Implementation

**Pricing Formulas:**
- Call: C = S·N(d₁) - K·e^(-rT)·N(d₂)
- Put: P = K·e^(-rT)·N(-d₂) - S·N(-d₁)

**Greeks Formulas:**
- Delta: Δ = N(d₁) [calls], N(d₁) - 1 [puts]
- Gamma: Γ = N'(d₁) / (S·σ·√T)
- Theta: Θ = -[S·N'(d₁)·σ / (2√T)] ± r·K·e^(-rT)·N(±d₂)
- Vega: ν = S·N'(d₁)·√T
- Rho: ρ = K·T·e^(-rT)·N(±d₂)

Where:
- d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
- d₂ = d₁ - σ√T
- N(·) = cumulative standard normal distribution
- N'(·) = standard normal probability density function

### Performance Characteristics

- **Single pricing**: ~0.1ms
- **Greeks calculation**: ~0.2ms
- **Batch pricing (1000 options)**: ~100ms
- **Memory usage**: Minimal (< 1MB for typical use)

### Numerical Stability

The implementation handles edge cases:
- T → 0: Returns intrinsic value
- S → 0: Correct limiting behavior
- K → 0: Correct limiting behavior
- σ → 0: Handled gracefully
- σ → ∞: Handled gracefully

All edge cases tested in `tests/test_black_scholes.py`.

## Backtest Framework

### Overview

A comprehensive backtesting framework for systematic option selling strategies on cryptocurrency assets. Implements a **covered call** strategy with rolling 7-day positions.

**Key Features:**
- ✅ Dollar-based accounting (not percentage-based)
- ✅ Multiple parallel scenarios to remove start-date bias
- ✅ Cash-settled options (no physical delivery)
- ✅ Regime-based analysis across different market periods
- ✅ Comprehensive statistics and visualization

### Strategy Description

**Covered Call Strategy:**
1. Hold 1 HBAR throughout the entire period
2. Every 7 days, sell 1 call option at strike% above spot
3. Collect premium upfront (cash credit)
4. At maturity:
   - If OTM: Keep premium, roll to new option
   - If ITM: Pay cash settlement, keep HBAR, roll to new option
5. Portfolio NAV = HBAR value + cumulative cash

**Key Parameters:**
- `rv_lookback_days`: Option maturity in days (default: 7)
- `strike_percent`: Strike as % of spot (e.g., 1.10 = 110% strike)
- `vol_spread`: Spread subtracted from realized vol to get implied vol (e.g., 0.10 = 10%)
- `risk_free_rate`: Discount rate (typically 0 for crypto)
- `cost_of_carry`: Cost of carry parameter (typically 0 for crypto)

### Files and Structure

```
scripts/
├── backtest.py                      # Original implementation (percentage-based, deprecated)
├── backtest_covered_call.py         # Corrected dollar-based covered call backtest ⭐
├── run_hbar_backtest.py            # Single period HBAR backtest
├── sanity_check_backtest.py        # Sanity check with 200% strikes
├── multi_period_sanity_check.py    # Multi-period regime analysis
└── multi_period_covered_call.py    # Multi-period with corrected logic ⭐

data/processed/
└── HBAR_daily.parquet              # Resampled daily HBAR price data

docs/
└── BACKTEST_INSTRUCTIONS.md        # Detailed backtest documentation

output/
├── HBAR_all_scenarios_nav.png      # NAV chart for all scenarios
├── HBAR_sanity_check_nav.png       # Sanity check results
└── multi_period/                    # Multi-period analysis outputs
    ├── Jan_2021_scenarios.png
    ├── Jan_2022_scenarios.png
    ├── Jan_2023_scenarios.png
    ├── Jan_2024_scenarios.png
    ├── comparison_chart.png
    └── comparison_summary.csv
```

### NAV Calculation Logic (Corrected)

**Dollar-Based Approach:**
```python
# Initial state
hbar_quantity = 1.0  # Fixed
cash_balance = 0.0
initial_spot = S0

# For each option roll
# Day 0 (Inception):
premium_received = BS_premium(spot, strike, T, r, sigma)
cash_balance += premium_received

# Day 7 (Maturity):
payoff = -max(0, fixing_price - strike_price)
cash_balance += payoff

# Portfolio value
hbar_value = hbar_quantity * current_spot_price
portfolio_value = hbar_value + cash_balance

# Final NAV (normalized to initial spot)
nav = portfolio_value / initial_spot
```

**Key Insight:** This approach avoids the multiplicative compounding issue of percentage-based calculations.

### Multiple Scenario Design

To remove start-date bias, the backtest runs N scenarios (where N = `rv_lookback_days`):
- Scenario 1 starts on day 7
- Scenario 2 starts on day 8
- ...
- Scenario N starts on day (7 + N - 1)

Each scenario operates independently with the same parameters. Results are averaged across scenarios to get debiased performance metrics.

### Usage Example

```python
from scripts.backtest_covered_call import backtest_covered_call_strategy

# Load HBAR daily data
df = pd.read_parquet('data/processed/HBAR_daily.parquet')

# Run backtest
results = backtest_covered_call_strategy(
    df,
    rv_lookback_days=7,       # 7-day options
    strike_percent=1.10,      # 110% strike (10% OTM)
    vol_spread=0.10,          # IV = RV - 10%
    risk_free_rate=0.0,       # 0% for crypto
    cost_of_carry=0.0         # 0% for crypto
)

# View averaged statistics
print_summary(results)
plot_all_scenarios(results)
```

### Key Learnings from Analysis

**1. HBAR Extreme Volatility:**
- HBAR can gain 100-200% in 7 days (observed multiple times)
- Even 200% OTM strikes get exercised during explosive rallies
- Feb 2020: 3.05x gain in 7 days (-105% payoff on short call)
- Nov 2024: 2.47x gain in 7 days (+147% in 7 days)

**2. Multi-Period Analysis:**
- Tested 4 starting periods: Jan 2021, Jan 2022, Jan 2023, Jan 2024
- All periods ran to Oct 2025 (same end date)
- All periods unprofitable even with 200% strikes (avg NAV ~0.90-0.93)
- Late 2024 explosion affected all periods, demonstrating tail risk

**3. Strategy Implications:**
- Covered calls on high-volatility assets require very wide strikes
- Single extreme event can wipe out years of premium collection
- Risk controls essential (stop losses, position limits, volatility filters)

### Comparison: Old vs New Implementation

**Old (Incorrect - Percentage-Based):**
- Calculated P&L as percentage of spot at inception
- Used multiplicative compounding: `nav = nav * (1 + pnl_pct)`
- Led to NAV going negative when payoff > spot_at_inception
- Conceptually unclear (mixing different percentage bases)

**New (Correct - Dollar-Based):**
- Tracks actual dollar cashflows (premiums + payoffs)
- Portfolio value = HBAR value + cash
- NAV normalized at the end: `nav = portfolio_value / initial_spot`
- Clean, intuitive accounting

### Output Files

**Per-Scenario Data:**
- Date, inception date, maturity date
- Spot at inception, strike price, fixing price
- Premium received, payoff paid
- Cumulative cash, HBAR value, portfolio value, NAV

**Summary Statistics (Averaged Across Scenarios):**
- Number of trades
- Final NAV
- Total return %, annualized return %
- Maximum drawdown %
- Win rate %
- Average P&L per trade
- Best/worst trade

### Future Enhancements

- [ ] Implement stop-loss logic
- [ ] Add position sizing (scale with capital)
- [ ] Volatility regime filters
- [ ] Test on multiple tokens (BTC, ETH, etc.)
- [ ] Credit spreads (buy protection to cap max loss)
- [ ] Dynamic strike selection based on volatility
- [ ] Real-time Greeks tracking
- [ ] Transaction costs and slippage modeling
