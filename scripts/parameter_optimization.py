"""
Parameter Optimization Grid for Covered Call Strategy

Tests multiple combinations of strike_percent and rv_lookback_days
to find optimal parameters for the covered call strategy.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from backtest_covered_call import backtest_covered_call_strategy, load_data_from_parquet

def filter_data_by_date(df, start_date, end_date):
    """
    Filter dataframe by date range

    Parameters:
    df: DataFrame with 'date' column
    start_date: Start date (string or datetime)
    end_date: End date (string or datetime)

    Returns:
    Filtered DataFrame
    """
    df = df.copy()

    # Ensure date is datetime
    if 'date' not in df.columns and df.index.name == 'date':
        df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])

    # Convert start/end dates to timezone-aware if needed
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # If df['date'] is timezone-aware, make start/end timezone-aware too
    if df['date'].dt.tz is not None:
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize('UTC')
        if end_dt.tz is None:
            end_dt = end_dt.tz_localize('UTC')

    # Filter by date range
    mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
    filtered_df = df.loc[mask].copy()

    print(f"Filtered data: {len(filtered_df)} days ({filtered_df['date'].min()} to {filtered_df['date'].max()})")

    return filtered_df

def run_parameter_optimization(
    df,
    strike_percents,
    rv_lookback_days_list,
    vol_spread=0.10,
    risk_free_rate=0.0,
    cost_of_carry=0.0
):
    """
    Run parameter optimization grid

    Parameters:
    df: DataFrame with price data
    strike_percents: List of strike percentages to test
    rv_lookback_days_list: List of lookback days to test
    vol_spread: Volatility spread (constant)
    risk_free_rate: Risk-free rate (constant)
    cost_of_carry: Cost of carry (constant)

    Returns:
    DataFrame with results for all parameter combinations
    """
    results = []
    total_combinations = len(strike_percents) * len(rv_lookback_days_list)

    print(f"\n{'='*80}")
    print(f"PARAMETER OPTIMIZATION GRID")
    print(f"{'='*80}")
    print(f"Total combinations to test: {total_combinations}")
    print(f"Strike percentages: {strike_percents}")
    print(f"Lookback days: {rv_lookback_days_list}")
    print(f"{'='*80}\n")

    iteration = 0
    for strike_pct, lookback_days in product(strike_percents, rv_lookback_days_list):
        iteration += 1
        print(f"[{iteration}/{total_combinations}] Testing: strike={strike_pct*100:.0f}%, lookback={lookback_days} days")

        try:
            # Run backtest
            backtest_results = backtest_covered_call_strategy(
                df,
                rv_lookback_days=lookback_days,
                strike_percent=strike_pct,
                vol_spread=vol_spread,
                risk_free_rate=risk_free_rate,
                cost_of_carry=cost_of_carry
            )

            # Extract summary stats (averaged across scenarios)
            summary = backtest_results['summary']

            # Calculate averages across all scenarios
            avg_nav = np.mean([s['final_nav'] for s in summary.values()])
            avg_return = np.mean([s['total_return'] for s in summary.values()])
            avg_win_rate = np.mean([s['win_rate'] for s in summary.values()])
            avg_max_dd = np.mean([s['max_drawdown'] for s in summary.values()])
            avg_num_trades = np.mean([s['num_trades'] for s in summary.values()])
            avg_total_cash = np.mean([s['total_cash'] for s in summary.values()])

            # Profitability metrics
            num_profitable_scenarios = sum([s['final_nav'] > 1.0 for s in summary.values()])
            pct_profitable_scenarios = num_profitable_scenarios / len(summary) * 100

            results.append({
                'strike_percent': strike_pct,
                'strike_label': f"{strike_pct*100:.0f}%",
                'rv_lookback_days': lookback_days,
                'avg_nav': avg_nav,
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'avg_max_drawdown': avg_max_dd,
                'avg_num_trades': avg_num_trades,
                'avg_total_cash': avg_total_cash,
                'num_scenarios': len(summary),
                'num_profitable_scenarios': num_profitable_scenarios,
                'pct_profitable_scenarios': pct_profitable_scenarios
            })

            print(f"  → Avg NAV: {avg_nav:.4f} | Avg Return: {avg_return:.2f}% | Win Rate: {avg_win_rate:.1f}%")

        except Exception as e:
            print(f"  → ERROR: {str(e)}")
            # Still record the failure
            results.append({
                'strike_percent': strike_pct,
                'strike_label': f"{strike_pct*100:.0f}%",
                'rv_lookback_days': lookback_days,
                'avg_nav': np.nan,
                'avg_return': np.nan,
                'avg_win_rate': np.nan,
                'avg_max_drawdown': np.nan,
                'avg_num_trades': np.nan,
                'avg_total_cash': np.nan,
                'num_scenarios': 0,
                'num_profitable_scenarios': 0,
                'pct_profitable_scenarios': 0
            })

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}\n")

    return pd.DataFrame(results)

def plot_heatmap(results_df, metric='avg_return', save_path=None):
    """
    Plot heatmap of optimization results

    Parameters:
    results_df: DataFrame with optimization results
    metric: Metric to plot ('avg_return', 'avg_nav', 'avg_win_rate', etc.)
    save_path: Path to save the plot
    """
    # Create pivot table for heatmap
    pivot_table = results_df.pivot(
        index='rv_lookback_days',
        columns='strike_label',
        values=metric
    )

    # Sort columns by strike percentage (numeric)
    strike_order = sorted(results_df['strike_percent'].unique())
    strike_labels = [f"{s*100:.0f}%" for s in strike_order]
    pivot_table = pivot_table[strike_labels]

    # Create figure
    plt.figure(figsize=(12, 8))

    # Choose colormap based on metric
    if 'return' in metric.lower() or 'nav' in metric.lower():
        cmap = 'RdYlGn'  # Red (bad) to Yellow to Green (good)
        center = 0 if 'return' in metric.lower() else 1
    else:
        cmap = 'viridis'
        center = None

    # Create heatmap
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=center,
        linewidths=0.5,
        cbar_kws={'label': metric.replace('_', ' ').title()},
        vmin=pivot_table.min().min() if center is None else None,
        vmax=pivot_table.max().max() if center is None else None
    )

    plt.xlabel('Strike (% of Spot)', fontsize=12, fontweight='bold')
    plt.ylabel('Lookback Days', fontsize=12, fontweight='bold')
    plt.title(f'Covered Call Strategy: {metric.replace("_", " ").title()} by Parameters',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    plt.show()

def print_top_combinations(results_df, top_n=5, sort_by='avg_return'):
    """
    Print top N parameter combinations

    Parameters:
    results_df: DataFrame with optimization results
    top_n: Number of top combinations to show
    sort_by: Metric to sort by
    """
    print(f"\n{'='*100}")
    print(f"TOP {top_n} PARAMETER COMBINATIONS (sorted by {sort_by.replace('_', ' ').title()})")
    print(f"{'='*100}")

    # Sort by specified metric (descending)
    top_results = results_df.sort_values(by=sort_by, ascending=False).head(top_n)

    print(f"\n{'Rank':<6} {'Strike':<10} {'Lookback':<12} {'Avg NAV':<12} {'Avg Return':<14} "
          f"{'Win Rate':<12} {'Max DD':<12} {'# Trades':<10}")
    print("-" * 100)

    for rank, (idx, row) in enumerate(top_results.iterrows(), start=1):
        print(f"{rank:<6} {row['strike_label']:<10} {row['rv_lookback_days']:<12.0f} "
              f"{row['avg_nav']:<12.4f} {row['avg_return']:<14.2f} "
              f"{row['avg_win_rate']:<12.1f} {row['avg_max_drawdown']:<12.2f} "
              f"{row['avg_num_trades']:<10.1f}")

    print("="*100)

def main():
    """Main execution"""

    # Configuration
    data_file = "data/processed/HBAR_daily.parquet"
    start_date = "2023-01-01"
    end_date = "2025-10-31"

    # Parameter grid
    strike_percents = [1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00]
    rv_lookback_days_list = [3, 5, 7, 10, 14, 21, 30]

    # Constants
    vol_spread = 0.10
    risk_free_rate = 0.0
    cost_of_carry = 0.0

    # Load and filter data
    print(f"Loading data from: {data_file}")
    df = load_data_from_parquet(data_file)
    df_filtered = filter_data_by_date(df, start_date, end_date)

    # Run optimization
    results_df = run_parameter_optimization(
        df_filtered,
        strike_percents,
        rv_lookback_days_list,
        vol_spread=vol_spread,
        risk_free_rate=risk_free_rate,
        cost_of_carry=cost_of_carry
    )

    # Save results to CSV
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / 'parameter_optimization_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Plot heatmap for average return
    heatmap_path = output_dir / 'parameter_optimization_heatmap.png'
    plot_heatmap(results_df, metric='avg_return', save_path=heatmap_path)

    # Print top 5 combinations
    print_top_combinations(results_df, top_n=5, sort_by='avg_return')

    # Also show top by Sharpe-like metric (return / abs(drawdown))
    results_df['sharpe_like'] = results_df['avg_return'] / results_df['avg_max_drawdown'].abs()
    print("\n")
    print_top_combinations(results_df, top_n=5, sort_by='sharpe_like')

    print("\n✓ Parameter optimization complete!")

if __name__ == "__main__":
    main()
