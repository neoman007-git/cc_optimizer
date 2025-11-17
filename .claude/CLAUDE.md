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

The option pricer can integrate with the `hist_vol_model` project for automatic implied volatility:

```python
# Fetch IV automatically from hist_vol_model
model = BlackScholesModel.from_iv_model(
    S=50000,
    K=50000,
    T=30/365,
    r=0.0,
    token='BTC',
    tenor_days=30,
    strike_percent=100.0,
    option_type='call',
    iv_model='model_01'
)
```

**Integration Notes:**
- hist_vol_model path: `/Users/neo/Velar/hist_vol_model/`
- Uses `compute_iv()` API from hist_vol_model
- Automatically resolves path and imports
- Falls back to manual sigma if hist_vol_model unavailable

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
- [x] **Comprehensive test suite (57 tests)**
- [x] **Functional and class-based APIs**
- [x] **hist_vol_model integration**
- [x] **Complete documentation (51 pages)**
- [x] **Example Jupyter notebook with visualizations**
- [x] **Validation against py_vollib reference**

### Roadmap

**Near-term (Next 1-2 months):**
- [ ] Implied volatility solver (Newton-Raphson method)
- [ ] Payoff diagrams and P&L visualization
- [ ] Batch pricing optimization
- [ ] Real-time market data integration (Deribit API)
- [ ] Compare BS prices with actual market prices

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
**Purpose**: Historical volatility and implied volatility modeling
**Integration**: Option pricer uses `compute_iv()` from hist_vol_model via `from_iv_model()` class method

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
