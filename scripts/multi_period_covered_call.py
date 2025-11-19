"""
Multi-Period Covered Call Analysis - Dollar-Based Accounting

Run corrected dollar-based backtest for multiple starting periods and
generate a 4-subplot comparison chart.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_covered_call import backtest_covered_call_strategy

def run_period_backtest(df_daily, start_date, period_name, strike_percent=1.10):
    """Run backtest for a specific starting period"""
    # Filter data to start from specified date
    df_period = df_daily[df_daily['date'] >= start_date].copy()
    df_period = df_period.reset_index(drop=True)

    print(f"\n{'='*80}")
    print(f"Period: {period_name}")
    print(f"{'='*80}")
    print(f"Start date: {start_date}")
    print(f"End date: {df_period['date'].max()}")
    print(f"Total days: {len(df_period)}")
    print(f"Price range: ${df_period['close'].min():.6f} - ${df_period['close'].max():.6f}")

    # Run backtest
    results = backtest_covered_call_strategy(
        df_period,
        rv_lookback_days=7,
        strike_percent=strike_percent,
        vol_spread=0.10,
        risk_free_rate=0.0,
        cost_of_carry=0.0
    )

    return results

def plot_multi_period_comparison(all_results, output_path='output/multi_period_covered_call.png'):
    """Create 2x2 subplot chart with all periods"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#9C27B0', '#00BCD4', '#FF5722']

    for idx, (period_name, results) in enumerate(all_results.items()):
        ax = axes[idx]

        # Plot all scenarios
        for i, (scenario_name, scenario_df) in enumerate(results['scenarios'].items()):
            if len(scenario_df) > 0:
                ax.plot(scenario_df['date'], scenario_df['nav'],
                       label=scenario_name, color=colors[i % len(colors)],
                       linewidth=2, alpha=0.7)

        # Add horizontal lines
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
        ax.axhline(y=0, color='red', linestyle=':', linewidth=1, alpha=0.3)

        # Calculate average stats
        avg_nav = np.mean([s['final_nav'] for s in results['summary'].values()])
        avg_return = np.mean([s['total_return'] for s in results['summary'].values()])

        # Labels and formatting
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('NAV (Multiple of Initial)', fontsize=11)
        ax.set_title(f"{period_name}\nAvg NAV: {avg_nav:.2f} | Avg Return: {avg_return:.1f}%",
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Covered Call Strategy (110% Strikes): Multi-Period Comparison\nDollar-Based Accounting',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Multi-period chart saved to: {output_path}")
    plt.close()

def print_comparison_summary(all_results):
    """Print summary table comparing all periods"""
    print("\n" + "="*100)
    print("MULTI-PERIOD SUMMARY: Covered Call Strategy (110% Strikes)")
    print("="*100)

    print(f"\n{'Period':<15} {'Avg NAV':<12} {'Avg Return %':<15} {'Avg Max DD %':<15} {'Profitable':<12}")
    print("-"*100)

    for period_name, results in all_results.items():
        avg_nav = np.mean([s['final_nav'] for s in results['summary'].values()])
        avg_return = np.mean([s['total_return'] for s in results['summary'].values()])
        avg_dd = np.mean([s['max_drawdown'] for s in results['summary'].values()])
        num_profitable = sum(1 for s in results['summary'].values() if s['final_nav'] > 1.0)

        print(f"{period_name:<15} {avg_nav:<12.2f} {avg_return:<15.1f} {avg_dd:<15.1f} {num_profitable}/7")

    print("="*100)

def main():
    # Load HBAR data
    data_path = Path('data/processed/HBAR_daily.parquet')

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    print(f"Loading data from: {data_path}")
    df_daily = pd.read_parquet(data_path)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    print(f"Loaded {len(df_daily)} days of HBAR price data")

    # Define test periods
    test_periods = [
        ('2021-01-01', 'Jan 2021'),
        ('2022-01-01', 'Jan 2022'),
        ('2023-01-01', 'Jan 2023'),
        ('2024-01-01', 'Jan 2024')
    ]

    print("\n" + "="*100)
    print("RUNNING MULTI-PERIOD ANALYSIS WITH CORRECTED DOLLAR-BASED BACKTEST")
    print("="*100)
    print("\nStrategy: Covered Call")
    print("Strike: 110% of spot (10% OTM)")
    print("Lookback: 7 days")
    print("Vol Spread: 10%")

    # Run backtests for each period
    all_results = {}

    for start_date, period_name in test_periods:
        results = run_period_backtest(df_daily, start_date, period_name, strike_percent=1.10)
        all_results[period_name] = results

    # Print comparison summary
    print_comparison_summary(all_results)

    # Generate 4-subplot chart
    plot_multi_period_comparison(all_results)

    print("\n✓ Multi-period analysis complete!")
    print("\nAll 4 periods displayed in a single chart with subplots.")

if __name__ == "__main__":
    main()
