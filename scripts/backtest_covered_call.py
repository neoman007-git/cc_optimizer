"""
Covered Call Backtest - Dollar-Based Accounting (CORRECTED)

This implements the CORRECT covered call strategy using dollar-based accounting:
- Track HBAR holdings (fixed at 1.0)
- Track cash balance (premiums + payoffs)
- Portfolio value = HBAR value + cash
- NAV = portfolio_value / initial_spot

This replaces the incorrect percentage-based approach in backtest.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path

# Black-Scholes pricing functions
def CND(x):
    """Cumulative normal distribution"""
    return norm.cdf(x)

def vanilla_bs(opt_type, S, X, T, r, b, v):
    """
    Black-Scholes-Merton option pricing

    Parameters:
    opt_type: 'C' for call, 'P' for put
    S: spot price
    X: strike price
    T: time to maturity in years
    r: risk-free rate
    b: cost of carry
    v: volatility (implied vol)

    Returns:
    Option premium in DOLLARS
    """
    if T == 0:
        if opt_type == 'C':
            return max(0, S - X)
        elif opt_type == 'P':
            return max(0, X - S)

    d1 = (np.log(S / X) + (b + v**2 / 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)

    if opt_type == 'C':
        premium = S * np.exp((b - r) * T) * CND(d1) - X * np.exp(-r * T) * CND(d2)
        return premium
    elif opt_type == 'P':
        premium = X * np.exp(-r * T) * CND(-d2) - S * np.exp((b - r) * T) * CND(-d1)
        return premium

def calculate_realized_vol(returns, lookback_days):
    """
    Calculate realized volatility (annualized)

    Parameters:
    returns: pandas Series of log returns
    lookback_days: number of days to look back
    """
    rolling_std = returns.rolling(window=lookback_days).std()
    annualized_vol = rolling_std * np.sqrt(365)
    return annualized_vol

def backtest_covered_call_strategy(
    df,
    rv_lookback_days=7,
    strike_percent=1.10,
    vol_spread=0.10,
    risk_free_rate=0.10,
    cost_of_carry=0.10
):
    """
    Backtest covered call strategy with DOLLAR-BASED accounting

    Strategy:
    - Hold 1 HBAR throughout
    - Sell 1 call every rv_lookback_days
    - Collect premium (cash credit)
    - Pay ITM settlement (cash debit)
    - Portfolio value = HBAR value + cash
    - NAV = portfolio_value / initial_spot

    Parameters:
    df: DataFrame with columns ['date', 'close']
    rv_lookback_days: lookback period for realized volatility
    strike_percent: strike as % of spot (e.g., 1.10 = 110%)
    vol_spread: spread to subtract from realized vol to get implied vol
    risk_free_rate: risk-free rate
    cost_of_carry: cost of carry rate

    Returns:
    dict with scenarios data and summary statistics
    """

    # Make a copy and ensure date column
    df = df.copy()
    if 'date' not in df.columns and df.index.name == 'date':
        df = df.reset_index()

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Loaded {len(df)} days of price data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Store initial spot for NAV normalization
    initial_spot = df.loc[rv_lookback_days, 'close']  # First tradeable day

    # Calculate log returns
    df['ln_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Calculate realized volatility
    df['realized_vol'] = calculate_realized_vol(df['ln_returns'], rv_lookback_days)

    # Calculate implied vol (realized vol - vol spread)
    df['implied_vol'] = df['realized_vol'] - vol_spread

    # Time to maturity in years
    T = rv_lookback_days / 365

    # Initialize columns for option premium
    df['option_premium'] = np.nan
    df['strike_price'] = np.nan

    # Calculate option premium for each day (after warmup period)
    print(f"\nCalculating option premiums...")
    valid_premium_count = 0
    for i in range(rv_lookback_days, len(df)):
        if pd.notna(df.loc[i, 'implied_vol']) and df.loc[i, 'implied_vol'] > 0:
            S = df.loc[i, 'close']
            X = S * strike_percent
            v = df.loc[i, 'implied_vol']

            premium = vanilla_bs('C', S, X, T, risk_free_rate, cost_of_carry, v)
            df.loc[i, 'option_premium'] = premium
            df.loc[i, 'strike_price'] = X
            valid_premium_count += 1

    print(f"Calculated {valid_premium_count} option premiums")

    # Now run multiple scenarios
    scenarios = {}

    print(f"\nRunning {rv_lookback_days} scenarios...")
    for scenario_num in range(rv_lookback_days):
        # Each scenario starts selling options on a different day
        first_option_day = rv_lookback_days + scenario_num

        # Portfolio state
        hbar_quantity = 1.0  # Fixed: always own 1 HBAR
        cash_balance = 0.0   # Cumulative cash from premiums + payoffs

        scenario_data = []

        # Track positions
        for day_idx in range(first_option_day, len(df), rv_lookback_days):
            maturity_idx = day_idx + rv_lookback_days

            if maturity_idx >= len(df):
                break

            # Day 0 (Inception): Sell call, receive premium
            premium_received = df.loc[day_idx, 'option_premium']
            strike_px = df.loc[day_idx, 'strike_price']
            spot_at_inception = df.loc[day_idx, 'close']

            # Skip if no valid premium
            if pd.isna(premium_received) or pd.isna(strike_px):
                continue

            # Cash increases by premium
            cash_balance += premium_received

            # Day 7 (Maturity): Calculate payoff
            fixing_px = df.loc[maturity_idx, 'close']
            payoff = -max(0, fixing_px - strike_px)  # Negative because we're short

            # Cash changes by payoff
            cash_balance += payoff

            # Portfolio value at maturity
            hbar_value = hbar_quantity * fixing_px
            portfolio_value = hbar_value + cash_balance

            # NAV (normalized to initial spot)
            nav = portfolio_value / initial_spot

            scenario_data.append({
                'date': df.loc[maturity_idx, 'date'],
                'option_inception_date': df.loc[day_idx, 'date'],
                'maturity_date': df.loc[maturity_idx, 'date'],
                'spot_at_inception': spot_at_inception,
                'strike_px': strike_px,
                'fixing_px': fixing_px,
                'premium_received': premium_received,
                'payoff': payoff,
                'cash_balance': cash_balance,
                'hbar_value': hbar_value,
                'portfolio_value': portfolio_value,
                'nav': nav
            })

        scenarios[f'Scenario_{scenario_num + 1}'] = pd.DataFrame(scenario_data)
        print(f"  Scenario {scenario_num + 1}: {len(scenario_data)} trades")

    # Create summary statistics
    summary = {}
    for scenario_name, scenario_df in scenarios.items():
        if len(scenario_df) > 0:
            initial_value = initial_spot  # Started with 1 HBAR worth initial_spot
            final_nav = scenario_df['nav'].iloc[-1]
            total_return = (final_nav - 1) * 100

            # Calculate max drawdown
            scenario_df['cummax_nav'] = scenario_df['nav'].expanding(min_periods=1).max()
            scenario_df['drawdown'] = (scenario_df['nav'] - scenario_df['cummax_nav']) / scenario_df['cummax_nav']
            max_dd = scenario_df['drawdown'].min() * 100

            summary[scenario_name] = {
                'num_trades': len(scenario_df),
                'initial_spot': initial_spot,
                'final_portfolio_value': scenario_df['portfolio_value'].iloc[-1],
                'final_nav': final_nav,
                'total_return': total_return,
                'avg_premium': scenario_df['premium_received'].mean(),
                'total_cash': scenario_df['cash_balance'].iloc[-1],
                'win_rate': (scenario_df['payoff'] >= 0).sum() / len(scenario_df) * 100,
                'max_drawdown': max_dd,
                'num_itm': (scenario_df['payoff'] < 0).sum()
            }

    return {
        'price_data': df,
        'scenarios': scenarios,
        'summary': summary,
        'parameters': {
            'initial_spot': initial_spot,
            'rv_lookback_days': rv_lookback_days,
            'strike_percent': strike_percent,
            'vol_spread': vol_spread,
            'risk_free_rate': risk_free_rate,
            'cost_of_carry': cost_of_carry
        }
    }

def plot_all_scenarios(results, save_path=None):
    """
    Plot NAV paths for all scenarios
    """
    plt.figure(figsize=(14, 8))

    colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#9C27B0', '#00BCD4', '#FF5722']

    for i, (scenario_name, scenario_df) in enumerate(results['scenarios'].items()):
        if len(scenario_df) > 0:
            plt.plot(scenario_df['date'], scenario_df['nav'],
                    label=scenario_name, color=colors[i % len(colors)], linewidth=2, alpha=0.8)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('NAV (Multiple of Initial)', fontsize=12)
    plt.title('Covered Call Strategy: All Scenarios NAV Paths', fontsize=14, fontweight='bold')
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to: {save_path}")

    plt.show()

def print_summary(results):
    """
    Print summary statistics for all scenarios
    """
    print("\n" + "="*90)
    print("COVERED CALL BACKTEST SUMMARY (Dollar-Based Accounting)")
    print("="*90)

    params = results['parameters']
    print(f"\nParameters:")
    print(f"  Initial Spot: ${params['initial_spot']:.6f}")
    print(f"  RV Lookback Days: {params['rv_lookback_days']}")
    print(f"  Strike: {params['strike_percent']*100:.0f}% of spot")
    print(f"  Vol Spread: {params['vol_spread']*100:.1f}%")
    print(f"  Risk-free Rate: {params['risk_free_rate']*100:.0f}%")

    print(f"\n{'Scenario':<15} {'Trades':<8} {'Final NAV':<12} {'Return %':<12} {'Total Cash $':<15} {'Win Rate %':<12} {'Max DD %':<12}")
    print("-" * 90)

    for scenario_name, stats in results['summary'].items():
        print(f"{scenario_name:<15} {stats['num_trades']:<8} "
              f"{stats['final_nav']:<12.4f} {stats['total_return']:<12.2f} "
              f"{stats['total_cash']:<15.6f} {stats['win_rate']:<12.1f} {stats['max_drawdown']:<12.2f}")

    print("\n" + "="*90)

    # Average statistics
    avg_nav = np.mean([s['final_nav'] for s in results['summary'].values()])
    avg_return = np.mean([s['total_return'] for s in results['summary'].values()])

    print(f"\nAveraged Across All Scenarios:")
    print(f"  Average Final NAV: {avg_nav:.4f}")
    print(f"  Average Return: {avg_return:.2f}%")
    print("="*90)

def load_data_from_parquet(file_path):
    """Load price data from parquet file"""
    df = pd.read_parquet(file_path)

    # Ensure required columns exist
    required_cols = ['date', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df

# Main execution
if __name__ == "__main__":
    # Example: Load data from parquet file
    data_file = "data/processed/HBAR_daily.parquet"

    print(f"Loading data from: {data_file}")
    df = load_data_from_parquet(data_file)

    # Run backtest
    results = backtest_covered_call_strategy(
        df,
        rv_lookback_days=7,
        strike_percent=1.10,
        vol_spread=0.10,
        risk_free_rate=0.0,
        cost_of_carry=0.0
    )

    # Print summary
    print_summary(results)

    # Plot results
    plot_all_scenarios(results, save_path='output/covered_call_nav.png')

    # Export scenario data
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    for scenario_name, scenario_df in results['scenarios'].items():
        output_file = output_dir / f'covered_call_{scenario_name}.csv'
        scenario_df.to_csv(output_file, index=False)
        print(f"Exported: {output_file}")

    print("\n✓ Backtest complete!")
