"""
Multi-Period Sanity Check: Test Strategy Across Different Market Regimes

This script runs the short call backtest (200% strikes) starting from different
dates to identify which market regimes are profitable vs unprofitable.

Key Design:
- Run 7 scenarios for each starting period
- Calculate AVERAGED statistics across scenarios (not individual NAV paths)
- Compare performance across different starting periods
- Validate if strategy works in calm periods
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtest import backtest_short_call_strategy

def calculate_annualized_return(final_nav, num_days):
    """Calculate annualized return from final NAV and number of days"""
    if final_nav <= 0 or num_days <= 0:
        return np.nan
    years = num_days / 365
    return (final_nav ** (1/years) - 1) * 100

def calculate_max_drawdown(nav_series):
    """Calculate maximum drawdown from NAV series"""
    if len(nav_series) == 0:
        return np.nan

    cummax = nav_series.expanding(min_periods=1).max()
    drawdown = (nav_series - cummax) / cummax
    return drawdown.min() * 100

def calculate_scenario_statistics(scenario_df, start_date, end_date):
    """Calculate statistics for a single scenario"""
    if len(scenario_df) == 0:
        return None

    # Handle timezone-aware vs timezone-naive datetime comparison
    if hasattr(end_date, 'tz') and end_date.tz is not None:
        end_date = end_date.tz_localize(None)
    if hasattr(start_date, 'tz') and start_date.tz is not None:
        start_date = start_date.tz_localize(None)

    num_days = (end_date - start_date).days
    final_nav = scenario_df['nav'].iloc[-1]

    stats = {
        'num_trades': len(scenario_df),
        'final_nav': final_nav,
        'total_return': (final_nav - 1) * 100,
        'annualized_return': calculate_annualized_return(final_nav, num_days),
        'max_drawdown': calculate_max_drawdown(scenario_df['nav']),
        'avg_pnl_pct': scenario_df['pnl_pct'].mean() * 100,
        'win_rate': (scenario_df['pnl_pct'] > 0).sum() / len(scenario_df) * 100,
        'best_trade': scenario_df['pnl_pct'].max() * 100,
        'worst_trade': scenario_df['pnl_pct'].min() * 100,
        'num_exercises': (scenario_df['payoff'] < 0).sum(),
        'exercise_rate': (scenario_df['payoff'] < 0).sum() / len(scenario_df) * 100
    }

    return stats

def run_period_backtest(df_daily, start_date, period_name):
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
    results = backtest_short_call_strategy(
        df_period,
        rv_lookback_days=7,
        strike_percent=2.00,  # 200% strikes (sanity check)
        vol_spread=0.10,
        risk_free_rate=0.0,
        cost_of_carry=0.0
    )

    # Calculate statistics for each scenario
    scenario_stats = {}
    for scenario_name, scenario_df in results['scenarios'].items():
        stats = calculate_scenario_statistics(
            scenario_df,
            pd.to_datetime(start_date),
            df_period['date'].max()
        )
        if stats:
            scenario_stats[scenario_name] = stats

    # Calculate averaged statistics across all scenarios
    if scenario_stats:
        avg_stats = {
            'period_name': period_name,
            'start_date': start_date,
            'num_scenarios': len(scenario_stats),
            'avg_num_trades': np.mean([s['num_trades'] for s in scenario_stats.values()]),
            'avg_final_nav': np.mean([s['final_nav'] for s in scenario_stats.values()]),
            'avg_total_return': np.mean([s['total_return'] for s in scenario_stats.values()]),
            'avg_annualized_return': np.mean([s['annualized_return'] for s in scenario_stats.values() if not np.isnan(s['annualized_return'])]),
            'avg_max_drawdown': np.mean([s['max_drawdown'] for s in scenario_stats.values() if not np.isnan(s['max_drawdown'])]),
            'avg_win_rate': np.mean([s['win_rate'] for s in scenario_stats.values()]),
            'avg_exercise_rate': np.mean([s['exercise_rate'] for s in scenario_stats.values()]),
            'avg_pnl_per_trade': np.mean([s['avg_pnl_pct'] for s in scenario_stats.values()]),
            'std_final_nav': np.std([s['final_nav'] for s in scenario_stats.values()]),
            'min_final_nav': np.min([s['final_nav'] for s in scenario_stats.values()]),
            'max_final_nav': np.max([s['final_nav'] for s in scenario_stats.values()])
        }
    else:
        avg_stats = None

    return {
        'results': results,
        'scenario_stats': scenario_stats,
        'avg_stats': avg_stats
    }

def plot_period_scenarios(period_results, output_path):
    """Plot NAV paths for all scenarios in a period"""
    results = period_results['results']
    avg_stats = period_results['avg_stats']

    plt.figure(figsize=(14, 8))

    colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#9C27B0', '#00BCD4', '#FF5722']

    for i, (scenario_name, scenario_df) in enumerate(results['scenarios'].items()):
        if len(scenario_df) > 0:
            plt.plot(scenario_df['date'], scenario_df['nav'],
                    label=scenario_name, color=colors[i % len(colors)],
                    linewidth=2, alpha=0.7)

    # Add horizontal line at NAV = 1.0
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')

    # Add average final NAV line
    if avg_stats:
        plt.axhline(y=avg_stats['avg_final_nav'], color='red', linestyle=':',
                   linewidth=2, alpha=0.7, label=f"Avg Final NAV: {avg_stats['avg_final_nav']:.3f}")

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('NAV (% of Initial)', fontsize=12)
    plt.title(f"Period: {avg_stats['period_name']} | Strike: 200% | Avg Return: {avg_stats['avg_total_return']:.1f}%",
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_table(all_period_stats):
    """Create comparison table of averaged statistics across periods"""
    print("\n" + "="*120)
    print("MULTI-PERIOD COMPARISON: AVERAGED STATISTICS ACROSS SCENARIOS")
    print("="*120)
    print("\nStrategy: Short Call | Strike: 200% of Spot | Lookback: 7 days")
    print("\nKey Metrics (Averaged Across 7 Scenarios per Period):")
    print("-"*120)

    header = f"{'Period':<15} {'Avg Final NAV':<15} {'Avg Return %':<15} {'Avg Ann Ret %':<15} {'Avg Max DD %':<15} {'Avg Win Rate %':<15} {'Profitable?':<12}"
    print(header)
    print("-"*120)

    for stats in all_period_stats:
        period = stats['period_name']
        avg_nav = stats['avg_final_nav']
        avg_ret = stats['avg_total_return']
        avg_ann_ret = stats['avg_annualized_return']
        avg_dd = stats['avg_max_drawdown']
        avg_wr = stats['avg_win_rate']
        profitable = '✓ YES' if avg_nav > 1.0 else '✗ NO'

        row = f"{period:<15} {avg_nav:<15.4f} {avg_ret:<15.2f} {avg_ann_ret:<15.2f} {avg_dd:<15.2f} {avg_wr:<15.1f} {profitable:<12}"
        print(row)

    print("="*120)

    # Summary
    profitable_periods = [s['period_name'] for s in all_period_stats if s['avg_final_nav'] > 1.0]
    print(f"\nProfitable Periods (Avg NAV > 1.0): {', '.join(profitable_periods) if profitable_periods else 'NONE'}")
    print(f"Unprofitable Periods: {', '.join([s['period_name'] for s in all_period_stats if s['avg_final_nav'] <= 1.0])}")

    # Best and worst periods
    best_period = max(all_period_stats, key=lambda x: x['avg_final_nav'])
    worst_period = min(all_period_stats, key=lambda x: x['avg_final_nav'])

    print(f"\nBest Period: {best_period['period_name']} (Avg NAV: {best_period['avg_final_nav']:.4f})")
    print(f"Worst Period: {worst_period['period_name']} (Avg NAV: {worst_period['avg_final_nav']:.4f})")
    print("="*120)

def plot_comparison_chart(all_period_results, output_path):
    """Plot comparison of average NAV across periods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Chart 1: Average Final NAV by Period
    periods = [r['avg_stats']['period_name'] for r in all_period_results]
    avg_navs = [r['avg_stats']['avg_final_nav'] for r in all_period_results]
    colors_map = ['green' if nav > 1.0 else 'red' for nav in avg_navs]

    bars = ax1.bar(periods, avg_navs, color=colors_map, alpha=0.7, edgecolor='black')
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Break-even')
    ax1.set_xlabel('Starting Period', fontsize=12)
    ax1.set_ylabel('Average Final NAV', fontsize=12)
    ax1.set_title('Average Final NAV by Starting Period\n(Averaged Across 7 Scenarios)',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, nav in zip(bars, avg_navs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{nav:.3f}', ha='center', va='bottom' if nav > 1.0 else 'top', fontsize=10)

    # Chart 2: Average Return % by Period
    avg_returns = [r['avg_stats']['avg_total_return'] for r in all_period_results]
    bars2 = ax2.bar(periods, avg_returns, color=colors_map, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Break-even')
    ax2.set_xlabel('Starting Period', fontsize=12)
    ax2.set_ylabel('Average Total Return (%)', fontsize=12)
    ax2.set_title('Average Total Return by Starting Period\n(Averaged Across 7 Scenarios)',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, ret in zip(bars2, avg_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ret:.1f}%', ha='center', va='bottom' if ret > 0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Paths
    daily_data_path = Path('/Users/neo/Velar/cc_optimizer/data/processed/HBAR_daily.parquet')
    output_dir = Path('/Users/neo/Velar/cc_optimizer/output/multi_period')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MULTI-PERIOD SANITY CHECK: REGIME-BASED ANALYSIS")
    print("="*80)
    print("\nTesting short call strategy (200% strikes) across different market regimes")
    print("Methodology: Average statistics across 7 scenarios for each starting period")

    # Load data
    df_daily = pd.read_parquet(daily_data_path)
    df_daily['date'] = pd.to_datetime(df_daily['date'])

    # Define test periods
    test_periods = [
        ('2021-01-01', 'Jan 2021'),
        ('2022-01-01', 'Jan 2022'),
        ('2023-01-01', 'Jan 2023'),
        ('2024-01-01', 'Jan 2024')
    ]

    # Run backtests for each period
    all_period_results = []
    all_period_stats = []

    for start_date, period_name in test_periods:
        period_result = run_period_backtest(df_daily, start_date, period_name)

        if period_result['avg_stats']:
            all_period_results.append(period_result)
            all_period_stats.append(period_result['avg_stats'])

            # Generate individual period chart
            chart_path = output_dir / f'{period_name.replace(" ", "_")}_scenarios.png'
            plot_period_scenarios(period_result, chart_path)
            print(f"\n✓ Chart saved: {chart_path}")

            # Export scenario data
            for scenario_name, scenario_df in period_result['results']['scenarios'].items():
                csv_path = output_dir / f'{period_name.replace(" ", "_")}_{scenario_name}.csv'
                scenario_df.to_csv(csv_path, index=False)

    # Create comparison outputs
    print("\n" + "="*80)
    print("GENERATING COMPARISON OUTPUTS")
    print("="*80)

    # Comparison table
    create_comparison_table(all_period_stats)

    # Comparison chart
    comparison_chart_path = output_dir / 'comparison_chart.png'
    plot_comparison_chart(all_period_results, comparison_chart_path)
    print(f"\n✓ Comparison chart saved: {comparison_chart_path}")

    # Export comparison table to CSV
    comparison_df = pd.DataFrame(all_period_stats)
    comparison_csv_path = output_dir / 'comparison_summary.csv'
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"✓ Comparison summary saved: {comparison_csv_path}")

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - Individual period charts: {len(test_periods)} files")
    print(f"  - Scenario CSVs: {len(test_periods) * 7} files")
    print(f"  - Comparison chart: comparison_chart.png")
    print(f"  - Comparison summary: comparison_summary.csv")
    print()

if __name__ == "__main__":
    main()
