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
    # Rolling standard deviation of returns
    rolling_std = returns.rolling(window=lookback_days).std()
    # Annualize: multiply by sqrt(365)
    annualized_vol = rolling_std * np.sqrt(365)
    return annualized_vol

def backtest_short_call_strategy(
    df,
    rv_lookback_days=7,
    strike_percent=1.10,
    vol_spread=0.10,
    risk_free_rate=0.10,
    cost_of_carry=0.10
):
    """
    Backtest short call option strategy with multiple scenarios
    
    Parameters:
    df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
    rv_lookback_days: lookback period for realized volatility
    strike_percent: strike as % of spot (e.g., 1.10 = 110%)
    vol_spread: spread to subtract from realized vol to get implied vol
    risk_free_rate: risk-free rate (e.g., 0.10 = 10%)
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
        
        scenario_data = []
        
        # Track positions: each position tracks when it was opened and its strike
        for day_idx in range(first_option_day, len(df), rv_lookback_days):
            maturity_idx = day_idx + rv_lookback_days
            
            if maturity_idx >= len(df):
                break
            
            # Premium received at option inception (day_idx)
            premium_received = df.loc[day_idx, 'option_premium']
            strike_px = df.loc[day_idx, 'strike_price']
            spot_at_inception = df.loc[day_idx, 'close']
            
            # Skip if no valid premium
            if pd.isna(premium_received) or pd.isna(strike_px):
                continue
            
            # Payoff at maturity (maturity_idx)
            fixing_px = df.loc[maturity_idx, 'close']
            payoff = -max(0, fixing_px - strike_px)  # Negative because we're short
            
            # Calculate payoff in % terms (relative to spot at inception)
            payoff_pct = payoff / spot_at_inception
            premium_pct = premium_received / spot_at_inception
            
            # PnL = premium + payoff (both in % terms)
            pnl_pct = premium_pct + payoff_pct
            
            scenario_data.append({
                'date': df.loc[maturity_idx, 'date'],
                'option_inception_date': df.loc[day_idx, 'date'],
                'maturity_date': df.loc[maturity_idx, 'date'],
                'spot_at_inception': spot_at_inception,
                'strike_px': strike_px,
                'fixing_px': fixing_px,
                'premium_received': premium_received,
                'payoff': payoff,
                'premium_pct': premium_pct,
                'payoff_pct': payoff_pct,
                'pnl_pct': pnl_pct
            })
        
        # Calculate cumulative NAV for this scenario
        # NAV starts at 1 (100%)
        nav = 1.0
        nav_path = [nav]
        
        for i, row in enumerate(scenario_data):
            # NAV compounds with each rolled position
            nav = nav * (1 + row['pnl_pct'])
            nav_path.append(nav)
            scenario_data[i]['nav'] = nav
        
        scenarios[f'Scenario_{scenario_num + 1}'] = pd.DataFrame(scenario_data)
        print(f"  Scenario {scenario_num + 1}: {len(scenario_data)} trades")
    
    # Create summary statistics
    summary = {}
    for scenario_name, scenario_df in scenarios.items():
        if len(scenario_df) > 0:
            summary[scenario_name] = {
                'num_trades': len(scenario_df),
                'final_nav': scenario_df['nav'].iloc[-1] if len(scenario_df) > 0 else 1.0,
                'total_return': (scenario_df['nav'].iloc[-1] - 1) * 100 if len(scenario_df) > 0 else 0,
                'avg_pnl_pct': scenario_df['pnl_pct'].mean() * 100,
                'win_rate': (scenario_df['pnl_pct'] > 0).sum() / len(scenario_df) * 100,
                'best_trade': scenario_df['pnl_pct'].max() * 100,
                'worst_trade': scenario_df['pnl_pct'].min() * 100
            }
    
    return {
        'price_data': df,
        'scenarios': scenarios,
        'summary': summary,
        'parameters': {
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
    plt.ylabel('NAV (% of Initial)', fontsize=12)
    plt.title('Short Call Strategy: All Scenarios NAV Paths', fontsize=14, fontweight='bold')
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
    print("BACKTEST SUMMARY")
    print("="*90)
    
    params = results['parameters']
    print(f"\nParameters:")
    print(f"  RV Lookback Days: {params['rv_lookback_days']}")
    print(f"  Strike: {params['strike_percent']*100:.0f}% of spot")
    print(f"  Vol Spread: {params['vol_spread']*100:.1f}%")
    print(f"  Risk-free Rate: {params['risk_free_rate']*100:.0f}%")
    
    print(f"\n{'Scenario':<15} {'Trades':<8} {'Final NAV':<12} {'Return %':<12} {'Avg PnL %':<12} {'Win Rate %':<12}")
    print("-" * 90)
    
    for scenario_name, stats in results['summary'].items():
        print(f"{scenario_name:<15} {stats['num_trades']:<8} "
              f"{stats['final_nav']:<12.4f} {stats['total_return']:<12.2f} "
              f"{stats['avg_pnl_pct']:<12.3f} {stats['win_rate']:<12.1f}")
    
    print("\n" + "="*90)

def load_data_from_parquet(file_path):
    """
    Load price data from parquet file
    
    Expected columns: date, open, high, low, close, volume
    """
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
    data_file = "data/BTC_daily.parquet"  # Update this path
    
    print(f"Loading data from: {data_file}")
    df = load_data_from_parquet(data_file)
    
    # Run backtest
    results = backtest_short_call_strategy(
        df,
        rv_lookback_days=7,
        strike_percent=1.10,
        vol_spread=0.10,
        risk_free_rate=0.10,
        cost_of_carry=0.10
    )
    
    # Print summary
    print_summary(results)
    
    # Plot results
    plot_all_scenarios(results, save_path='output/all_scenarios_nav.png')
    
    # Export scenario data
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    for scenario_name, scenario_df in results['scenarios'].items():
        output_file = output_dir / f'{scenario_name}.csv'
        scenario_df.to_csv(output_file, index=False)
        print(f"Exported: {output_file}")
    
    print("\n✓ Backtest complete!")