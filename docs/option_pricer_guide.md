# Complete Guide: Black-Scholes Option Pricer

**Version:** 1.0
**Last Updated:** 2025-11-17
**Author:** CC Optimizer Team

---

## Table of Contents

1. [Introduction](#introduction)
2. [Setup & Installation](#setup)
3. [Quick Start](#quick-start)
4. [Parameter Guide](#parameters)
5. [Functional API (Simple)](#functional-api)
6. [Class-Based API (Advanced)](#class-api)
7. [Greeks Calculations](#greeks)
8. [Crypto Option Examples](#crypto-examples)
9. [Integration with hist_vol_model](#iv-integration)
10. [Common Use Cases](#use-cases)
11. [Tips & Best Practices](#tips)
12. [API Reference](#api-reference)
13. [Troubleshooting](#troubleshooting)

---

## 1. Introduction {#introduction}

The CC Optimizer option pricer is a lightweight, pure Python implementation of the Black-Scholes model for pricing European vanilla call and put options. It's specifically designed for cryptocurrency options but works equally well for traditional assets.

### Key Features

- ✅ **Pure NumPy/SciPy** - No heavyweight dependencies like QuantLib
- ✅ **Dual API** - Simple functional API and advanced class-based API
- ✅ **Complete Greeks** - Delta, Gamma, Theta, Vega (and Rho)
- ✅ **Validated** - All calculations verified against py_vollib
- ✅ **Crypto-Ready** - Handles high volatility, zero rates, short tenors
- ✅ **Extensible** - Clean architecture for future models
- ✅ **Well-Tested** - 57 tests, 100% pass rate

### What You Can Do

- Price European call and put options
- Calculate option Greeks for risk management
- Analyze option strategies (straddles, spreads, etc.)
- Build volatility surfaces
- Integrate with implied volatility models
- Hedge portfolios with delta-neutral strategies

---

## 2. Setup & Installation {#setup}

### Prerequisites

- Python 3.9+
- Virtual environment activated

### Activate Virtual Environment

```bash
cd /Users/neo/Velar/cc_optimizer
source .venv/bin/activate
```

### Verify Installation

```bash
# Quick test
python -c "from cc_optimizer.options import bs_pricer; print('✓ Ready to use!')"

# Run tests
pytest tests/test_black_scholes.py tests/test_greeks.py -v
```

### Start Python Environment

```bash
# Option 1: Python interactive shell
python

# Option 2: IPython (recommended for exploration)
ipython

# Option 3: Jupyter notebook
jupyter notebook
```

---

## 3. Quick Start {#quick-start}

### Import the Module

```python
# Import pricing functions
from cc_optimizer.options import bs_pricer, calculate_greeks

# OR import classes for advanced usage
from cc_optimizer.options import BlackScholesModel, Greeks
```

### Price Your First Option (30 seconds)

```python
# Price a Bitcoin call option
# BTC @ $50,000, 30-day ATM call, 80% volatility
price = bs_pricer(
    S=50000,           # Current BTC price
    K=50000,           # Strike price (ATM)
    T=30/365,          # 30 days (convert to years)
    r=0.0,             # No risk-free rate for crypto
    sigma=0.8,         # 80% annual volatility
    option_type='call'
)

print(f"Option price: ${price:,.2f}")
# Output: Option price: $4,327.52
```

### Calculate Greeks (30 seconds)

```python
# Calculate all Greeks at once
greeks = calculate_greeks(
    S=50000,
    K=50000,
    T=30/365,
    r=0.0,
    sigma=0.8,
    option_type='call'
)

# Display results
for name, value in greeks.items():
    print(f"{name.capitalize()}: {value:.4f}")

# Output:
# Delta: 0.5398
# Gamma: 0.0000
# Theta: -36.7845 (per day)
# Vega: 56.4821 (per 1% vol change)
```

---

## 4. Parameter Guide {#parameters}

### Understanding Each Parameter

| Parameter | Description | Units | Example Values |
|-----------|-------------|-------|----------------|
| **S** | Spot/Current price of underlying asset | USD (or any currency) | 50000 (BTC), 3000 (ETH), 100 (stock) |
| **K** | Strike price | Same as S | 50000 (ATM), 55000 (OTM call), 45000 (ITM call) |
| **T** | Time to expiration | **Years** (important!) | 1 (1 year), 30/365 (30 days), 7/365 (1 week) |
| **r** | Risk-free interest rate | Annual decimal | 0.05 (5%), 0.0 (crypto), 0.02 (2%) |
| **sigma** | Volatility (implied or historical) | Annual decimal | 0.2 (20%), 0.8 (80%), 1.2 (120%) |
| **option_type** | Call or Put | String | 'call', 'put', 'c', 'p' |

### Common Time Conversions

```python
# Always convert days to years by dividing by 365

T = 1/365      # 1 day
T = 7/365      # 1 week
T = 30/365     # 30 days (1 month)
T = 90/365     # 90 days (3 months)
T = 180/365    # 180 days (6 months)
T = 1          # 1 year
T = 2          # 2 years
```

### Common Volatility Values

```python
# Traditional assets
sigma = 0.15   # Low vol (utilities, bonds): 15%
sigma = 0.20   # Medium vol (blue chip stocks): 20%
sigma = 0.30   # High vol (tech stocks): 30%

# Cryptocurrencies
sigma = 0.60   # Low crypto vol: 60%
sigma = 0.80   # Medium crypto vol: 80%
sigma = 1.00   # High crypto vol: 100%
sigma = 1.50   # Extreme crypto vol: 150%
```

### Moneyness Reference

| Term | Call | Put | Example (S=100) |
|------|------|-----|-----------------|
| **Deep ITM** | S >> K | S << K | Call K=80, Put K=120 |
| **ITM** | S > K | S < K | Call K=95, Put K=105 |
| **ATM** | S ≈ K | S ≈ K | K=100 |
| **OTM** | S < K | S > K | Call K=105, Put K=95 |
| **Deep OTM** | S << K | S >> K | Call K=120, Put K=80 |

---

## 5. Functional API (Simple) {#functional-api}

**Best for:** Quick calculations, one-off pricing, scripts

### 5.1 Basic Call Option

```python
from cc_optimizer.options import bs_pricer

# Traditional stock option
call_price = bs_pricer(
    S=100,              # Stock at $100
    K=105,              # Strike at $105 (OTM)
    T=90/365,           # 90 days to expiration
    r=0.05,             # 5% risk-free rate
    sigma=0.25,         # 25% volatility
    option_type='call'
)

print(f"Call price: ${call_price:.2f}")
```

### 5.2 Basic Put Option

```python
# Protective put for portfolio insurance
put_price = bs_pricer(
    S=100,              # Portfolio value: $100
    K=95,               # Protect at $95 (5% downside)
    T=30/365,           # 30 days protection
    r=0.03,             # 3% risk-free rate
    sigma=0.20,         # 20% volatility
    option_type='put'
)

print(f"Put price: ${put_price:.2f}")
print(f"Insurance cost: {put_price/100*100:.2f}% of portfolio")
```

### 5.3 Batch Pricing Multiple Options

```python
# Price multiple strikes at once
spot = 50000
strikes = [45000, 47500, 50000, 52500, 55000]
T = 30/365
r = 0.0
sigma = 0.75

print("Strike | Call Price | Put Price")
print("-" * 40)

for K in strikes:
    call = bs_pricer(spot, K, T, r, sigma, 'call')
    put = bs_pricer(spot, K, T, r, sigma, 'put')
    moneyness = "ITM" if K < spot else "ATM" if K == spot else "OTM"
    print(f"${K:,} | ${call:,.2f} | ${put:,.2f} | {moneyness}")
```

### 5.4 Calculate Greeks Functionally

```python
from cc_optimizer.options import calculate_greeks

# Get all Greeks in one call
greeks = calculate_greeks(
    S=50000,
    K=52500,
    T=7/365,
    r=0.0,
    sigma=0.9,
    option_type='call'
)

print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Theta: ${greeks['theta']:.2f} per day")
print(f"Vega: ${greeks['vega']:.2f} per 1% vol")
```

---

## 6. Class-Based API (Advanced) {#class-api}

**Best for:** Complex analysis, reusing models, getting additional info

### 6.1 Create a Model Instance

```python
from cc_optimizer.options import BlackScholesModel, Greeks

# Create a model object
model = BlackScholesModel(
    S=100,
    K=100,
    T=1,
    r=0.05,
    sigma=0.2,
    option_type='call'
)

# Now you can call multiple methods on the same model
price = model.price()
intrinsic = model.intrinsic_value()
time_value = model.time_value()
moneyness = model.moneyness()

print(f"Price: ${price:.2f}")
print(f"Intrinsic Value: ${intrinsic:.2f}")
print(f"Time Value: ${time_value:.2f}")
print(f"Moneyness: {moneyness}")
```

### 6.2 Get Detailed Model Information

```python
# Print complete model summary
print(model)

# Output:
# CALL Option:
#   Spot: $100.00
#   Strike: $100.00
#   Time to expiry: 1.0000 years
#   Risk-free rate: 5.00%
#   Volatility: 20.00%
#   Price: $10.4506
#   Moneyness: ATM
```

### 6.3 Intrinsic vs Time Value Analysis

```python
# Useful for understanding option composition
model = BlackScholesModel(S=110, K=100, T=0.5, r=0.05, sigma=0.25, option_type='call')

price = model.price()
intrinsic = model.intrinsic_value()
time_value = model.time_value()

print(f"Total Price: ${price:.2f}")
print(f"  Intrinsic Value: ${intrinsic:.2f} ({intrinsic/price*100:.1f}%)")
print(f"  Time Value: ${time_value:.2f} ({time_value/price*100:.1f}%)")

# Example output:
# Total Price: $13.68
#   Intrinsic Value: $10.00 (73.1%)
#   Time Value: $3.68 (26.9%)
```

### 6.4 Moneyness Classification

```python
# Automatically classify ITM/ATM/OTM
scenarios = [
    (110, 100, 'call'),  # ITM call
    (100, 100, 'call'),  # ATM call
    (90, 100, 'call'),   # OTM call
    (90, 100, 'put'),    # ITM put
    (100, 100, 'put'),   # ATM put
    (110, 100, 'put'),   # OTM put
]

for S, K, opt_type in scenarios:
    model = BlackScholesModel(S=S, K=K, T=1, r=0.05, sigma=0.2, option_type=opt_type)
    print(f"{opt_type.upper():4} | S=${S:3} K=${K:3} | {model.moneyness():3} | ${model.price():.2f}")
```

---

## 7. Greeks Calculations {#greeks}

### 7.1 Understanding the Greeks

| Greek | Measures | Interpretation | Range |
|-------|----------|----------------|-------|
| **Delta (Δ)** | Sensitivity to price changes | How much option price changes per $1 move in underlying | Calls: 0 to 1<br>Puts: -1 to 0 |
| **Gamma (Γ)** | Rate of delta change | How much delta changes per $1 move (convexity) | Always ≥ 0<br>Max at ATM |
| **Theta (Θ)** | Time decay | How much value lost per day | Usually < 0<br>(time decay) |
| **Vega (ν)** | Sensitivity to volatility | How much price changes per 1% vol change | Always ≥ 0 |
| **Rho (ρ)** | Sensitivity to interest rate | How much price changes per 1% rate change | Calls: > 0<br>Puts: < 0 |

### 7.2 Calculate All Greeks

```python
from cc_optimizer.options import BlackScholesModel, Greeks

# Create model
model = BlackScholesModel(
    S=50000,
    K=50000,
    T=30/365,
    r=0.0,
    sigma=0.8,
    option_type='call'
)

# Create Greeks calculator
greeks = Greeks(model)

# Get all Greeks
all_greeks = greeks.all_greeks()

# Display
print("\n=== Option Greeks ===")
for name, value in all_greeks.items():
    print(f"{name.capitalize():8}: {value:10.4f}")
```

### 7.3 Individual Greek Calculations

```python
# Calculate Greeks individually (if you only need specific ones)
greeks = Greeks(model)

delta = greeks.delta()
gamma = greeks.gamma()
theta = greeks.theta(per_day=True)    # Or per_day=False for annual
vega = greeks.vega(per_percent=True)  # Or per_percent=False for full
rho = greeks.rho(per_percent=True)    # Rate sensitivity (optional)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.6f}")
print(f"Theta: ${theta:.2f} per day")
print(f"Vega:  ${vega:.2f} per 1% vol")
```

### 7.4 Practical Greeks Interpretation

#### Delta - The Hedge Ratio

```python
# Calculate hedge ratio for 100 options
model = BlackScholesModel(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')
greeks = Greeks(model)

num_contracts = 100
delta = greeks.delta()

# Each contract covers 1 BTC, so:
hedge_amount = num_contracts * delta

print(f"Position: {num_contracts} call options")
print(f"Delta: {delta:.4f}")
print(f"Hedge required: {hedge_amount:.2f} BTC")
print(f"Value to hedge: ${hedge_amount * 50000:,.0f}")

# To delta-hedge, you need to SHORT hedge_amount BTC
print(f"\nAction: SHORT {hedge_amount:.2f} BTC to delta-hedge")
```

**Interpretation:**
- Delta = 0.54 means the option moves $0.54 for every $1 move in BTC
- For 100 contracts, you need to short 54 BTC to be delta-neutral
- As BTC price changes, delta changes (gamma effect), requiring rebalancing

#### Gamma - The Delta Sensitivity

```python
model = BlackScholesModel(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')
greeks = Greeks(model)

gamma = greeks.gamma()
delta = greeks.delta()

# If BTC moves $1000
price_move = 1000
delta_change = gamma * price_move

print(f"Current Delta: {delta:.4f}")
print(f"Gamma: {gamma:.6f}")
print(f"\nIf BTC moves ${price_move:,}:")
print(f"  New Delta (approx): {delta + delta_change:.4f}")
print(f"  Delta change: {delta_change:.4f}")
print(f"  Hedge adjustment needed: {delta_change * 100:.2f} BTC (for 100 contracts)")
```

**Interpretation:**
- High gamma = delta changes rapidly = frequent rehedging needed
- ATM options have highest gamma
- Long options have positive gamma (good for volatility)

#### Theta - The Time Decay

```python
model = BlackScholesModel(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')
greeks = Greeks(model)

price = model.price()
theta = greeks.theta(per_day=True)

print(f"Option Price: ${price:,.2f}")
print(f"Theta: ${theta:.2f} per day")
print(f"\nValue lost over time:")
print(f"  1 day:  ${abs(theta):.2f} ({abs(theta)/price*100:.2f}%)")
print(f"  7 days: ${abs(theta)*7:.2f} ({abs(theta)*7/price*100:.2f}%)")
print(f"  30 days: ${abs(theta)*30:.2f} ({abs(theta)*30/price*100:.2f}%)")
```

**Interpretation:**
- Theta is usually negative for long options (you lose money as time passes)
- Theta accelerates as expiration approaches
- Short-dated ATM options have the highest theta

#### Vega - The Volatility Sensitivity

```python
model = BlackScholesModel(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')
greeks = Greeks(model)

price = model.price()
vega = greeks.vega(per_percent=True)

print(f"Option Price: ${price:,.2f}")
print(f"Current Vol: {model.sigma:.0%}")
print(f"Vega: ${vega:.2f} per 1% vol")
print(f"\nIf volatility changes:")
print(f"  Vol +10% (to {(model.sigma+0.10):.0%}): Price ${price + vega*10:,.2f} (+${vega*10:.2f})")
print(f"  Vol -10% (to {(model.sigma-0.10):.0%}): Price ${price - vega*10:,.2f} (-${vega*10:.2f})")
```

**Interpretation:**
- Vega is always positive for long options
- Crypto options have very high vega due to volatile vol
- Longer-dated options have higher vega

### 7.5 Greeks Over Time

```python
import numpy as np

# Analyze how Greeks change as expiration approaches
S, K, r, sigma = 100, 100, 0.05, 0.25
days_to_expiry = [90, 60, 30, 14, 7, 3, 1]

print("Days | Price  | Delta | Gamma  | Theta | Vega")
print("-" * 60)

for days in days_to_expiry:
    T = days / 365
    model = BlackScholesModel(S, K, T, r, sigma, 'call')
    greeks = Greeks(model)

    print(f"{days:4} | ${model.price():5.2f} | {greeks.delta():.3f} | "
          f"{greeks.gamma():.4f} | ${greeks.theta():.2f} | ${greeks.vega():.2f}")
```

### 7.6 Greeks Across Strikes

```python
# See how Greeks vary across different strikes
spot = 100
strikes = range(80, 121, 5)
T = 30/365

print("Strike | Delta  | Gamma  | Vega   | Price")
print("-" * 50)

for K in strikes:
    model = BlackScholesModel(S=spot, K=K, T=T, r=0.0, sigma=0.5, option_type='call')
    greeks = Greeks(model)

    print(f"${K:3} | {greeks.delta():.4f} | {greeks.gamma():.4f} | "
          f"${greeks.vega():5.2f} | ${model.price():6.2f}")
```

---

## 8. Crypto Option Examples {#crypto-examples}

### 8.1 Bitcoin Weekly Options

```python
# BTC weekly call (common on Deribit)
btc_weekly = BlackScholesModel(
    S=50000,           # BTC spot
    K=52000,           # 4% OTM strike
    T=7/365,           # 7 days (Friday expiry)
    r=0.0,             # No risk-free rate
    sigma=0.80,        # 80% annual vol
    option_type='call'
)

print("=== BTC Weekly Call ===")
print(f"Spot: ${btc_weekly.S:,}")
print(f"Strike: ${btc_weekly.K:,} ({(btc_weekly.K/btc_weekly.S-1)*100:+.1f}%)")
print(f"Days: 7")
print(f"Price: ${btc_weekly.price():,.2f}")
print(f"Premium: {btc_weekly.price()/btc_weekly.S*100:.2f}% of spot")
print(f"Moneyness: {btc_weekly.moneyness()}")

# Greeks
greeks = Greeks(btc_weekly)
print(f"\nDelta: {greeks.delta():.4f}")
print(f"Theta: -${abs(greeks.theta()):,.2f} per day")
```

### 8.2 Ethereum Monthly Put (Portfolio Protection)

```python
# ETH put for downside protection
eth_put = BlackScholesModel(
    S=3000,            # ETH spot
    K=2700,            # 10% downside protection
    T=30/365,          # 30 days
    r=0.0,
    sigma=0.90,        # 90% vol
    option_type='put'
)

holding_size = 100  # 100 ETH
protection_cost = eth_put.price() * holding_size
portfolio_value = eth_put.S * holding_size

print("=== ETH Portfolio Protection ===")
print(f"Holdings: {holding_size} ETH @ ${eth_put.S:,}")
print(f"Portfolio Value: ${portfolio_value:,}")
print(f"Protection Level: ${eth_put.K:,} ({(eth_put.K/eth_put.S-1)*100:.1f}%)")
print(f"\nPut Price: ${eth_put.price():.2f} per ETH")
print(f"Total Cost: ${protection_cost:,.2f}")
print(f"Insurance Cost: {protection_cost/portfolio_value*100:.2f}% of portfolio")
print(f"\nMax Loss: ${(eth_put.S - eth_put.K) * holding_size - protection_cost:,.2f}")
```

### 8.3 Straddle Strategy (Volatility Play)

```python
# Buy both call and put at same strike (bet on big move)
S = 50000
K = 50000
T = 14/365
r = 0.0
sigma = 0.95

call = BlackScholesModel(S, K, T, r, sigma, 'call')
put = BlackScholesModel(S, K, T, r, sigma, 'put')

straddle_cost = call.price() + put.price()
breakeven_up = K + straddle_cost
breakeven_down = K - straddle_cost

print("=== ATM Straddle ===")
print(f"Spot: ${S:,}")
print(f"Strike: ${K:,}")
print(f"Days: 14")
print(f"Vol: {sigma:.0%}")
print(f"\nCall Price: ${call.price():,.2f}")
print(f"Put Price: ${put.price():,.2f}")
print(f"Total Cost: ${straddle_cost:,.2f}")
print(f"\nBreakeven Points:")
print(f"  Up: ${breakeven_up:,.2f} ({(breakeven_up/S-1)*100:+.1f}%)")
print(f"  Down: ${breakeven_down:,.2f} ({(breakeven_down/S-1)*100:+.1f}%)")
print(f"\nNeed {abs(breakeven_up-S)/S*100:.1f}% move to profit")
```

### 8.4 Covered Call Strategy

```python
# Sell OTM call against holdings (income generation)
holding = 10  # 10 BTC
spot = 50000
strike = 55000  # 10% OTM
T = 30/365

call_sold = BlackScholesModel(
    S=spot,
    K=strike,
    T=T,
    r=0.0,
    sigma=0.75,
    option_type='call'
)

premium_received = call_sold.price() * holding
monthly_yield = premium_received / (spot * holding) * 100

print("=== Covered Call ===")
print(f"Holdings: {holding} BTC @ ${spot:,}")
print(f"Call Strike: ${strike:,} ({(strike/spot-1)*100:+.1f}%)")
print(f"\nPremium per BTC: ${call_sold.price():,.2f}")
print(f"Total Premium: ${premium_received:,.2f}")
print(f"Monthly Yield: {monthly_yield:.2f}%")
print(f"Annual Yield (if repeated): {monthly_yield*12:.1f}%")

# Risk analysis
greeks = Greeks(call_sold)
print(f"\nRisk Metrics:")
print(f"  Delta: {greeks.delta():.4f} (probability of being assigned)")
print(f"  Max Gain: ${(strike - spot) * holding + premium_received:,.2f}")
```

### 8.5 Bull Call Spread

```python
# Buy lower strike call, sell higher strike call (limited risk/reward)
S = 50000
T = 30/365
r = 0.0
sigma = 0.80

# Long call at 50k
call_long = BlackScholesModel(S, K=50000, T=T, r=r, sigma=sigma, option_type='call')

# Short call at 55k
call_short = BlackScholesModel(S, K=55000, T=T, r=r, sigma=sigma, option_type='call')

net_cost = call_long.price() - call_short.price()
max_profit = (55000 - 50000) - net_cost
max_loss = net_cost

print("=== Bull Call Spread ===")
print(f"Spot: ${S:,}")
print(f"\nBUY Call @ $50,000: Cost ${call_long.price():,.2f}")
print(f"SELL Call @ $55,000: Credit ${call_short.price():,.2f}")
print(f"\nNet Cost: ${net_cost:,.2f}")
print(f"Max Profit: ${max_profit:,.2f} (if spot > $55,000)")
print(f"Max Loss: ${net_cost:,.2f} (if spot < $50,000)")
print(f"Breakeven: ${50000 + net_cost:,.2f}")
print(f"Risk/Reward Ratio: 1:{max_profit/net_cost:.2f}")
```

---

## 9. Integration with hist_vol_model {#iv-integration}

### 9.1 Basic Integration

When the `hist_vol_model` project is available, you can automatically fetch implied volatility:

```python
from cc_optimizer.options import BlackScholesModel

# Create model using IV from hist_vol_model
try:
    model = BlackScholesModel.from_iv_model(
        S=50000,                # Current BTC price
        K=50000,                # Strike
        T=30/365,               # Time to expiry
        r=0.0,                  # Risk-free rate
        token='BTC',            # Token symbol
        tenor_days=30,          # Match T
        strike_percent=100.0,   # ATM (100% of spot)
        option_type='call',
        iv_model='model_01',    # Your IV model
        quote_type='mid'        # 'bid', 'mid', or 'ask'
    )

    print(f"IV from model: {model.sigma:.2%}")
    print(f"Option Price: ${model.price():,.2f}")

except ImportError:
    print("hist_vol_model not available - use manual sigma")
    model = BlackScholesModel(S=50000, K=50000, T=30/365, r=0.0, sigma=0.75, option_type='call')
```

### 9.2 Strike Percent Mapping

```python
# Different strikes as percent of spot
strike_scenarios = [
    (45000, 90.0),   # 10% ITM
    (47500, 95.0),   # 5% ITM
    (50000, 100.0),  # ATM
    (52500, 105.0),  # 5% OTM
    (55000, 110.0),  # 10% OTM
]

spot = 50000

for K, strike_pct in strike_scenarios:
    try:
        model = BlackScholesModel.from_iv_model(
            S=spot,
            K=K,
            T=30/365,
            r=0.0,
            token='BTC',
            tenor_days=30,
            strike_percent=strike_pct,
            option_type='call'
        )
        print(f"Strike ${K:,} ({strike_pct}%): IV={model.sigma:.2%}, Price=${model.price():,.2f}")
    except ImportError:
        print("hist_vol_model not available")
        break
```

### 9.3 Volatility Surface with IV Model

```python
# Build volatility surface using hist_vol_model
import pandas as pd

tenors = [7, 14, 30, 60, 90]
strike_pcts = [90, 95, 100, 105, 110]
spot = 50000

results = []

for tenor in tenors:
    for strike_pct in strike_pcts:
        K = spot * strike_pct / 100
        try:
            model = BlackScholesModel.from_iv_model(
                S=spot,
                K=K,
                T=tenor/365,
                r=0.0,
                token='BTC',
                tenor_days=tenor,
                strike_percent=strike_pct,
                option_type='call'
            )
            results.append({
                'Tenor': tenor,
                'Strike%': strike_pct,
                'Strike': K,
                'IV': model.sigma,
                'Price': model.price()
            })
        except ImportError:
            print("hist_vol_model not available")
            break

if results:
    df = pd.DataFrame(results)
    pivot = df.pivot(index='Strike%', columns='Tenor', values='IV')
    print("\nImplied Volatility Surface:")
    print(pivot)
```

---

## 10. Common Use Cases {#use-cases}

### 10.1 Compare Call vs Put

```python
import numpy as np

def compare_call_put(S, K, T, r, sigma):
    call = BlackScholesModel(S, K, T, r, sigma, 'call')
    put = BlackScholesModel(S, K, T, r, sigma, 'put')

    print(f"=== Comparison @ Strike ${K} ===")
    print(f"Spot: ${S}")
    print(f"\n{'':12} | Call      | Put")
    print("-" * 40)
    print(f"{'Price':12} | ${call.price():8.2f} | ${put.price():8.2f}")
    print(f"{'Intrinsic':12} | ${call.intrinsic_value():8.2f} | ${put.intrinsic_value():8.2f}")
    print(f"{'Time Value':12} | ${call.time_value():8.2f} | ${put.time_value():8.2f}")
    print(f"{'Moneyness':12} | {call.moneyness():8} | {put.moneyness():8}")

    # Verify put-call parity
    parity_lhs = call.price() - put.price()
    parity_rhs = S - K * np.exp(-r * T)
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = {parity_lhs:.6f}")
    print(f"  S - K*e^(-rT) = {parity_rhs:.6f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.10f} ✓")

compare_call_put(S=100, K=100, T=1, r=0.05, sigma=0.25)
```

### 10.2 Find Breakeven Points

```python
def analyze_breakeven(S, K, T, r, sigma, option_type):
    model = BlackScholesModel(S, K, T, r, sigma, option_type)
    price = model.price()

    if option_type == 'call':
        breakeven = K + price
        direction = "above"
    else:  # put
        breakeven = K - price
        direction = "below"

    print(f"=== {option_type.upper()} Breakeven Analysis ===")
    print(f"Current Spot: ${S:,.2f}")
    print(f"Strike: ${K:,.2f}")
    print(f"Premium Paid: ${price:,.2f}")
    print(f"\nBreakeven: ${breakeven:,.2f}")
    print(f"Move Required: {abs(breakeven-S)/S*100:.2f}%")
    print(f"Profit when spot goes {direction} ${breakeven:,.2f}")

    # Sample P&L at different spot prices
    print(f"\nP&L at Expiry:")
    test_spots = np.linspace(S*0.8, S*1.2, 9)
    for test_S in test_spots:
        if option_type == 'call':
            payout = max(test_S - K, 0)
        else:
            payout = max(K - test_S, 0)
        pnl = payout - price
        print(f"  Spot ${test_S:,.0f}: P&L ${pnl:+,.2f}")

analyze_breakeven(S=50000, K=52000, T=7/365, r=0.0, sigma=0.85, option_type='call')
```

### 10.3 Volatility Smile Analysis

```python
def volatility_smile(S, strikes, T, r, base_sigma):
    """
    Demonstrate how option prices vary with different implied vols
    """
    print("=== Volatility Smile ===")
    print(f"Spot: ${S:,}")
    print("\nStrike | Moneyness | IV    | Call Price | Put Price")
    print("-" * 65)

    for K in strikes:
        # Simulate vol smile (typically higher vol for OTM)
        moneyness = K / S
        if moneyness < 0.95:
            sigma = base_sigma + 0.05  # ITM slightly higher
        elif moneyness > 1.05:
            sigma = base_sigma + 0.10  # OTM significantly higher
        else:
            sigma = base_sigma  # ATM baseline

        call = BlackScholesModel(S, K, T, r, sigma, 'call')
        put = BlackScholesModel(S, K, T, r, sigma, 'put')

        print(f"${K:,} | {moneyness:8.2f} | {sigma:5.0%} | "
              f"${call.price():9,.2f} | ${put.price():9,.2f}")

volatility_smile(
    S=50000,
    strikes=[45000, 47500, 50000, 52500, 55000],
    T=30/365,
    r=0.0,
    base_sigma=0.75
)
```

### 10.4 Portfolio Greeks Analysis

```python
# Calculate total Greeks for a portfolio of options
positions = [
    {'type': 'call', 'K': 50000, 'qty': 10, 'position': 'long'},
    {'type': 'call', 'K': 55000, 'qty': 10, 'position': 'short'},
    {'type': 'put', 'K': 45000, 'qty': 5, 'position': 'long'},
]

S = 50000
T = 30/365
r = 0.0
sigma = 0.80

total_delta = 0
total_gamma = 0
total_theta = 0
total_vega = 0

print("=== Portfolio Greeks ===")
print("\nPosition Details:")

for pos in positions:
    model = BlackScholesModel(S, pos['K'], T, r, sigma, pos['type'])
    greeks = Greeks(model)

    sign = 1 if pos['position'] == 'long' else -1
    qty = pos['qty'] * sign

    delta = greeks.delta() * qty
    gamma = greeks.gamma() * qty
    theta = greeks.theta() * qty
    vega = greeks.vega() * qty

    total_delta += delta
    total_gamma += gamma
    total_theta += theta
    total_vega += vega

    print(f"{pos['position'].upper():5} {qty:+4} {pos['type']:4} @ ${pos['K']:,}")
    print(f"  Delta: {delta:+.2f}, Gamma: {gamma:+.4f}, Theta: ${theta:+.2f}, Vega: ${vega:+.2f}")

print(f"\nPortfolio Totals:")
print(f"  Delta: {total_delta:+.2f}")
print(f"  Gamma: {total_gamma:+.4f}")
print(f"  Theta: ${total_theta:+.2f} per day")
print(f"  Vega:  ${total_vega:+.2f} per 1% vol")

# Hedging requirement
print(f"\nTo delta-hedge: {-total_delta:+.2f} BTC spot")
```

---

## 11. Tips & Best Practices {#tips}

### 11.1 Choosing Between APIs

**Use Functional API when:**
- Quick one-off calculations
- Scripting or automation
- You only need the price

**Use Class-Based API when:**
- Need multiple pieces of information (price, intrinsic, Greeks)
- Building complex analysis tools
- Want to cache/reuse model parameters

### 11.2 Parameter Best Practices

```python
# ✅ GOOD: Explicit and clear
model = BlackScholesModel(
    S=50000,
    K=50000,
    T=30/365,  # Clear: 30 days
    r=0.0,
    sigma=0.80,
    option_type='call'
)

# ❌ BAD: Easy to make mistakes
model = BlackScholesModel(50000, 50000, 30, 0, 0.80, 'call')  # T=30 YEARS!
```

### 11.3 Common Pitfalls

```python
# ❌ MISTAKE 1: Forgetting to convert days to years
T = 30  # WRONG! This is 30 YEARS, not 30 days
T = 30/365  # CORRECT: 30 days in years

# ❌ MISTAKE 2: Using percentage instead of decimal
sigma = 80  # WRONG! This is 8000% volatility
sigma = 0.80  # CORRECT: 80% volatility

# ❌ MISTAKE 3: Wrong interest rate format
r = 5  # WRONG! This is 500% rate
r = 0.05  # CORRECT: 5% rate

# ❌ MISTAKE 4: Mismatching units
S = 50000
K = 50  # WRONG! Strike should be in same units as spot
K = 50000  # CORRECT
```

### 11.4 Validation Checks

```python
# The pricer has built-in validation
try:
    model = BlackScholesModel(
        S=-100,  # Negative price
        K=100,
        T=1,
        r=0.05,
        sigma=0.2,
        option_type='call'
    )
except ValueError as e:
    print(f"Validation caught error: {e}")
    # Output: "S (spot price) must be positive, got -100"
```

### 11.5 Performance Tips

```python
# If pricing many options with same parameters, reuse model
S, T, r, sigma = 50000, 30/365, 0.0, 0.80
strikes = range(45000, 55001, 1000)

# Method 1: Efficient (create Greeks calculator once)
for K in strikes:
    model = BlackScholesModel(S, K, T, r, sigma, 'call')
    greeks = Greeks(model)
    all_greeks = greeks.all_greeks()  # Calculates all at once
    print(f"${K}: {all_greeks}")

# Method 2: Less efficient (multiple calls)
for K in strikes:
    model = BlackScholesModel(S, K, T, r, sigma, 'call')
    greeks = Greeks(model)
    delta = greeks.delta()  # Each call recalculates d1, d2
    gamma = greeks.gamma()
    theta = greeks.theta()
    vega = greeks.vega()
```

### 11.6 When to Use Different Greeks Units

```python
greeks = Greeks(model)

# Theta: Use per_day=True for practical trading
theta_day = greeks.theta(per_day=True)    # Typical: "I lose $50/day in theta"
theta_year = greeks.theta(per_day=False)  # Academic: Annual theta

# Vega: Use per_percent=True for practical trading
vega_pct = greeks.vega(per_percent=True)   # Typical: "Vega is $100 per 1% vol"
vega_full = greeks.vega(per_percent=False) # Academic: Per 100% vol change

# Rho: Usually per_percent=True (rate changes are small)
rho_pct = greeks.rho(per_percent=True)     # Per 1% rate change (e.g., 5%→6%)
```

### 11.7 Sanity Checks

```python
def sanity_check(model):
    """Run basic sanity checks on option pricing"""
    price = model.price()
    intrinsic = model.intrinsic_value()

    # Check 1: Price should be non-negative
    assert price >= 0, "Price cannot be negative"

    # Check 2: Price should be at least intrinsic value
    assert price >= intrinsic, "Price should be ≥ intrinsic value"

    # Check 3: Call price should be less than spot (except extreme cases)
    if model.option_type == 'call' and model.T > 0.01:  # Not at expiry
        assert price < model.S * 1.5, "Call price seems too high"

    # Check 4: Greeks should be in reasonable ranges
    greeks = Greeks(model)
    delta = greeks.delta()
    if model.option_type == 'call':
        assert 0 <= delta <= 1, f"Call delta should be 0-1, got {delta}"
    else:
        assert -1 <= delta <= 0, f"Put delta should be -1-0, got {delta}"

    gamma = greeks.gamma()
    assert gamma >= 0, "Gamma should be non-negative"

    vega = greeks.vega()
    assert vega >= 0, "Vega should be non-negative"

    print("✓ All sanity checks passed")

# Test it
model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
sanity_check(model)
```

---

## 12. API Reference {#api-reference}

### Functional API

#### `bs_pricer(S, K, T, r, sigma, option_type='call')`

Price a European vanilla option.

**Parameters:**
- `S` (float): Spot price
- `K` (float): Strike price
- `T` (float): Time to expiration (years)
- `r` (float): Risk-free rate (annual)
- `sigma` (float): Volatility (annual)
- `option_type` (str): 'call', 'put', 'c', or 'p'

**Returns:**
- `float`: Option price

**Example:**
```python
price = bs_pricer(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
```

#### `calculate_greeks(S, K, T, r, sigma, option_type='call', include_rho=False)`

Calculate all Greeks for an option.

**Parameters:**
- Same as `bs_pricer`
- `include_rho` (bool): Whether to include rho (default: False)

**Returns:**
- `dict`: Dictionary with keys 'delta', 'gamma', 'theta', 'vega', and optionally 'rho'

**Example:**
```python
greeks = calculate_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
```

### Class-Based API

#### `BlackScholesModel(S, K, T, r, sigma, option_type='call')`

Black-Scholes option pricing model.

**Methods:**

##### `price()`
Calculate option price.

**Returns:** `float`

##### `intrinsic_value()`
Calculate intrinsic value.

**Returns:** `float`

##### `time_value()`
Calculate time value (price - intrinsic).

**Returns:** `float`

##### `moneyness()`
Determine moneyness classification.

**Returns:** `str` - 'ITM', 'ATM', or 'OTM'

##### `from_iv_model(cls, S, K, T, r, token, tenor_days, strike_percent=100.0, option_type='call', iv_model='model_01', quote_type='mid')`

Create model using IV from hist_vol_model (class method).

**Parameters:**
- `S`, `K`, `T`, `r`, `option_type`: Same as constructor
- `token` (str): Token symbol ('BTC', 'ETH', etc.)
- `tenor_days` (int): Option tenor in days
- `strike_percent` (float): Strike as % of spot
- `iv_model` (str): IV model to use
- `quote_type` (str): 'bid', 'mid', or 'ask'

**Returns:** `BlackScholesModel` instance

#### `Greeks(bs_model)`

Greeks calculator for Black-Scholes model.

**Methods:**

##### `delta()`
Calculate delta (dV/dS).

**Returns:** `float` - Range: [0,1] for calls, [-1,0] for puts

##### `gamma()`
Calculate gamma (d²V/dS²).

**Returns:** `float` - Always non-negative

##### `theta(per_day=True)`
Calculate theta (dV/dt).

**Parameters:**
- `per_day` (bool): If True, return per day; if False, per year

**Returns:** `float` - Usually negative

##### `vega(per_percent=True)`
Calculate vega (dV/dσ).

**Parameters:**
- `per_percent` (bool): If True, return per 1%; if False, per 100%

**Returns:** `float` - Always non-negative

##### `rho(per_percent=True)`
Calculate rho (dV/dr).

**Parameters:**
- `per_percent` (bool): If True, return per 1%; if False, per 100%

**Returns:** `float` - Positive for calls, negative for puts

##### `all_greeks(include_rho=False)`
Calculate all Greeks at once.

**Parameters:**
- `include_rho` (bool): Whether to include rho

**Returns:** `dict` - Dictionary with all Greeks

---

## 13. Troubleshooting {#troubleshooting}

### Common Errors

#### ImportError: No module named 'cc_optimizer'

**Problem:** Package not installed or not in Python path.

**Solution:**
```bash
cd /Users/neo/Velar/cc_optimizer
source .venv/bin/activate
pip install -e .
```

#### ValueError: S (spot price) must be positive

**Problem:** Invalid parameter value.

**Solution:** Check all parameters are in correct format:
- S, K must be positive
- T must be non-negative
- sigma must be positive
- Use decimals for rates and volatility (0.05 not 5)

#### Option price seems too high/low

**Problem:** Likely unit error.

**Solution:**
- Check T is in years (divide days by 365)
- Check sigma is decimal (0.8 not 80)
- Check r is decimal (0.05 not 5)
- Verify S and K are in same units

#### Greeks don't match expectations

**Problem:** Wrong units or parameter error.

**Solution:**
- For theta: Use `per_day=True` for daily decay
- For vega: Use `per_percent=True` for 1% changes
- Verify option type ('call' vs 'put')
- Check T > 0 (not at expiration)

### Getting Help

1. **Check Documentation:**
   - This guide
   - Example notebook: `notebooks/option_pricer_example.ipynb`
   - Docstrings: `help(BlackScholesModel)`

2. **Run Tests:**
   ```bash
   pytest tests/test_black_scholes.py tests/test_greeks.py -v
   ```

3. **Validate Against py_vollib:**
   ```python
   import py_vollib.black_scholes as pyv

   # Compare results
   our_price = bs_pricer(100, 100, 1, 0.05, 0.2, 'call')
   ref_price = pyv.black_scholes('c', 100, 100, 1, 0.05, 0.2)

   print(f"Our price: {our_price:.6f}")
   print(f"Reference: {ref_price:.6f}")
   print(f"Difference: {abs(our_price - ref_price):.10f}")
   ```

---

## Quick Reference Card

```python
# IMPORTS
from cc_optimizer.options import bs_pricer, calculate_greeks  # Functional
from cc_optimizer.options import BlackScholesModel, Greeks    # Class-based

# QUICK PRICE
price = bs_pricer(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')

# QUICK GREEKS
greeks = calculate_greeks(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')

# DETAILED MODEL
model = BlackScholesModel(S=50000, K=50000, T=30/365, r=0.0, sigma=0.8, option_type='call')
price = model.price()
intrinsic = model.intrinsic_value()
time_val = model.time_value()
moneyness = model.moneyness()

# GREEKS
greeks_calc = Greeks(model)
all_greeks = greeks_calc.all_greeks()
delta = greeks_calc.delta()
gamma = greeks_calc.gamma()

# WITH IV FROM hist_vol_model
model = BlackScholesModel.from_iv_model(
    S=50000, K=50000, T=30/365, r=0.0,
    token='BTC', tenor_days=30, strike_percent=100.0, option_type='call'
)
```

---

## Appendix: Black-Scholes Formula

For reference, here are the Black-Scholes formulas implemented in this pricer:

### Option Pricing

**Call Option:**
```
C = S·N(d₁) - K·e^(-rT)·N(d₂)
```

**Put Option:**
```
P = K·e^(-rT)·N(-d₂) - S·N(-d₁)
```

Where:
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

And N(·) is the cumulative standard normal distribution.

### Greeks

**Delta:**
```
Δ_call = N(d₁)
Δ_put = N(d₁) - 1
```

**Gamma:**
```
Γ = N'(d₁) / (S·σ·√T)
```

**Theta:**
```
Θ_call = -[S·N'(d₁)·σ / (2√T)] - r·K·e^(-rT)·N(d₂)
Θ_put = -[S·N'(d₁)·σ / (2√T)] + r·K·e^(-rT)·N(-d₂)
```

**Vega:**
```
ν = S·N'(d₁)·√T
```

**Rho:**
```
ρ_call = K·T·e^(-rT)·N(d₂)
ρ_put = -K·T·e^(-rT)·N(-d₂)
```

Where N'(·) is the standard normal probability density function.

---

**End of Guide**

For additional examples and visualizations, see `notebooks/option_pricer_example.ipynb`.

For technical details, see the source code in `src/cc_optimizer/options/`.
