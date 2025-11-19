# Short Call Option Strategy Backtest - Documentation

## Overview

This project backtests a systematic short call option selling strategy on cryptocurrency daily price data. The strategy involves selling short-dated call options and rolling them at maturity, creating multiple parallel scenarios that start on different days.

## Trading Strategy Logic

### Core Concept
- **Strategy**: Sell short call options at a strike above spot, hold until maturity, collect premium, pay any ITM payoff, then roll into a new position
- **Option Maturity**: Rolling basis with `rv_lookback_days` (typically 7 days)
- **Multiple Scenarios**: Since options mature every N days, there are N parallel scenarios running simultaneously, each starting on a different day

### Key Parameters

| Parameter          | Description                         | Typical Value |
| ------------------ | ----------------------------------- | ------------- |
| `rv_lookback_days` | Days to maturity for each option    | 7             |
| `strike_percent`   | Strike as % of spot price           | 110% (1.10)   |
| `vol_spread`       | Spread subtracted from RV to get IV | 10% (0.10)    |
| `risk_free_rate`   | Risk-free rate for discounting      | 10% (0.10)    |
| `cost_of_carry`    | Cost of carry (b parameter)         | 10% (0.10)    |

## Mathematical Formulas

### 1. Log Returns
```
ln_return[t] = ln(close[t] / close[t-1])
```

### 2. Realized Volatility (Annualized)
```
RV[t] = std(ln_returns[t-n:t]) × sqrt(365)
```
where n = `rv_lookback_days`

### 3. Implied Volatility
```
IV[t] = RV[t] - vol_spread
```

### 4. Option Pricing (Black-Scholes-Merton)

For a call option:
```
S = spot price (close price)
X = strike price = S × strike_percent
T = rv_lookback_days / 365
r = risk_free_rate
b = cost_of_carry
v = implied_vol

d1 = (ln(S/X) + (b + v²/2) × T) / (v × sqrt(T))
d2 = d1 - v × sqrt(T)

Call Premium = S × exp((b-r)×T) × N(d1) - X × exp(-r×T) × N(d2)
```

where N(x) is the cumulative standard normal distribution

### 5. Payoff at Maturity
```
Short Call Payoff = -max(0, Fixing_Price - Strike_Price)
```
(Negative because we're short the call)

### 6. P&L Calculation (in % terms)
```
Premium_pct = Premium_received / Spot_at_inception
Payoff_pct = Payoff / Spot_at_inception
PnL_pct = Premium_pct + Payoff_pct
```

### 7. NAV Evolution
```
NAV[0] = 1.0 (100%)
NAV[i] = NAV[i-1] × (1 + PnL_pct[i])
```

## Data Requirements

### Input Data Format
Parquet files with the following columns:
- `date`: datetime (daily frequency)
- `open`: opening price
- `high`: highest price
- `low`: lowest price
- `close`: closing price
- `volume`: trading volume

### Data Location
```
project_folder/
├── backtest_short_call.py
├── BACKTEST_INSTRUCTIONS.md
└── data/
    ├── BTC_daily.parquet
    ├── ETH_daily.parquet
    └── SOL_daily.parquet
```

## Strategy Timeline Example (7-day lookback)
```
Day 0-6:   Accumulate returns to calculate first RV
Day 7:     Strategy can start - first RV available
           - Scenario 1: Sell option, matures Day 14
Day 8:     - Scenario 2: Sell option, matures Day 15
Day 9:     - Scenario 3: Sell option, matures Day 16
...
Day 13:    - Scenario 7: Sell option, matures Day 20
Day 14:    - Scenario 1: Option matures, pay payoff, roll to new option (matures Day 21)
Day 15:    - Scenario 2: Option matures, pay payoff, roll to new option (matures Day 22)
...
```

Each scenario operates independently but in parallel, creating 7 separate NAV paths.

## Backtest Flow

1. **Load Data**: Read parquet file with daily OHLCV data
2. **Calculate Returns**: Compute log returns from close prices
3. **Calculate RV**: Rolling realized volatility over `rv_lookback_days`
4. **Calculate IV**: RV minus vol_spread
5. **Price Options**: For each day (after warmup), calculate BS call premium
6. **Run Scenarios**: For each of N scenarios (where N = rv_lookback_days):
   - Start on day `rv_lookback_days + scenario_num`
   - Every N days:
     - Record premium received (% of spot)
     - N days later: calculate payoff (% of spot)
     - Calculate PnL and update NAV
7. **Output Results**: Summary statistics and NAV paths for all scenarios

## Code Structure

### Main Functions

#### `vanilla_bs(opt_type, S, X, T, r, b, v)`
Calculates Black-Scholes-Merton option price.

**Parameters:**
- `opt_type`: 'C' for call, 'P' for put
- `S`: spot price
- `X`: strike price
- `T`: time to maturity (years)
- `r`: risk-free rate
- `b`: cost of carry
- `v`: implied volatility

**Returns:** Option premium (in dollar terms)

#### `calculate_realized_vol(returns, lookback_days)`
Calculates annualized realized volatility.

**Parameters:**
- `returns`: pandas Series of log returns
- `lookback_days`: rolling window size

**Returns:** pandas Series of annualized volatility

#### `backtest_short_call_strategy(df, **params)`
Main backtest function that runs all scenarios.

**Parameters:**
- `df`: DataFrame with OHLCV data
- `rv_lookback_days`: option maturity in days
- `strike_percent`: strike as multiple of spot
- `vol_spread`: spread to subtract from RV
- `risk_free_rate`: discount rate
- `cost_of_carry`: cost of carry rate

**Returns:** Dictionary containing:
- `price_data`: DataFrame with all calculations
- `scenarios`: Dict of DataFrames, one per scenario
- `summary`: Dict of summary statistics per scenario
- `parameters`: Dict of input parameters

#### `plot_all_scenarios(results)`
Plots NAV paths for all scenarios on one chart.

#### `print_summary(results)`
Prints formatted summary statistics table.

## Expected Output

### Summary Statistics (per scenario)
- Number of trades executed
- Final NAV (as multiple of initial capital)
- Total return %
- Average P&L per trade %
- Win rate %
- Best trade %
- Worst trade %

### Scenario DataFrames
Each scenario DataFrame contains:
- `date`: maturity date
- `option_inception_date`: when option was sold
- `maturity_date`: when option matured
- `spot_at_inception`: spot price when sold
- `strike_px`: strike price
- `fixing_px`: spot price at maturity
- `premium_received`: dollar premium
- `payoff`: dollar payoff (negative for short)
- `premium_pct`: premium as % of spot
- `payoff_pct`: payoff as % of spot
- `pnl_pct`: total P&L as % of spot
- `nav`: cumulative NAV

### Visualization
Line chart showing NAV evolution over time for all 7 scenarios, color-coded.

## Usage Example
```python
import pandas as pd
from backtest_short_call import backtest_short_call_strategy, plot_all_scenarios, print_summary

# Load data
df = pd.read_parquet('data/BTC_daily.parquet')

# Run backtest
results = backtest_short_call_strategy(
    df,
    rv_lookback_days=7,
    strike_percent=1.10,
    vol_spread=0.10,
    risk_free_rate=0.10,
    cost_of_carry=0.10
)

# View results
print_summary(results)
plot_all_scenarios(results)

# Export scenario data
for scenario_name, scenario_df in results['scenarios'].items():
    scenario_df.to_csv(f'output/{scenario_name}.csv', index=False)
```

## Important Notes

1. **Warmup Period**: The strategy requires `rv_lookback_days` of historical data before the first option can be priced. This is because we need that many days of returns to calculate the first realized volatility.

2. **Multiple Scenarios**: With a 7-day maturity, there are 7 independent scenarios. This is NOT ensemble modeling - it's the reality of how a rolling strategy works. On any given day, you might have 7 different option positions open, each at different points in their lifecycle.

3. **NAV Compounding**: Each scenario's NAV compounds multiplicatively. If you make 2% on trade 1 and 3% on trade 2:
```
   NAV = 1.0 × 1.02 × 1.03 = 1.0506 (not 1.05)
```

4. **Percentage Returns**: All P&L calculations are done in percentage terms relative to the spot price at option inception. This normalizes returns across different price levels.

5. **Negative Payoffs**: Short call payoffs are negative when ITM (we pay the counterparty). The strategy makes money from collecting premiums upfront and hopes the options expire OTM or only slightly ITM.

## Validation Checklist

To verify the backtest is working correctly:

- [ ] First option is priced on day `rv_lookback_days` (not before)
- [ ] Each scenario has approximately `(total_days - rv_lookback_days) / rv_lookback_days` trades
- [ ] NAV starts at 1.0 for all scenarios
- [ ] Premium values are positive and reasonable (typically 1-5% of spot)
- [ ] Payoffs are negative (or zero) since we're short
- [ ] NAV paths show realistic evolution (no sudden jumps, smooth compounding)
- [ ] All 7 scenarios have similar but offset timing

## Common Issues & Solutions

**Issue**: "Not enough data for realized volatility"
**Solution**: Ensure your dataset has at least `rv_lookback_days + 1` rows

**Issue**: "All NAVs end at 1.0"
**Solution**: Check that options are actually being priced (IV > 0, all parameters valid)

**Issue**: "NAV paths are identical"
**Solution**: Verify that scenarios are starting on different days (offset by 1 day each)

**Issue**: "Extremely high/low returns"
**Solution**: Check vol_spread parameter - if IV becomes negative or too high, option pricing breaks down

## Extensions & Modifications

Possible enhancements:
- Add transaction costs
- Include bid-ask spreads
- Add position sizing limits
- Calculate Sharpe ratio and maximum drawdown
- Add delta hedging simulation
- Support multiple strike levels
- Add early exit logic based on Greeks