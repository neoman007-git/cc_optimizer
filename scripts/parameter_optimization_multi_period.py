"""
Multi-Period Parameter Optimization Grid for Covered Call Strategy

Runs optimization for multiple starting periods and creates comparison heatmaps.
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
    period_name,
    vol_spread=0.10,
    risk_free_rate=0.0,
    cost_of_carry=0.0
):
    """
    Run parameter optimization grid for a single period

    Parameters:
    df: DataFrame with price data
    strike_percents: List of strike percentages to test
    rv_lookback_days_list: List of lookback days to test
    period_name: Name of the period (for logging)
    vol_spread: Volatility spread (constant)
    risk_free_rate: Risk-free rate (constant)
    cost_of_carry: Cost of carry (constant)

    Returns:
    DataFrame with results for all parameter combinations
    """
    results = []
    total_combinations = len(strike_percents) * len(rv_lookback_days_list)

    print(f"\n{'='*80}")
    print(f"PERIOD: {period_name}")
    print(f"{'='*80}")
    print(f"Total combinations to test: {total_combinations}")
    print(f"{'='*80}\n")

    iteration = 0
    for strike_pct, lookback_days in product(strike_percents, rv_lookback_days_list):
        iteration += 1
        print(f"[{iteration}/{total_combinations}] Testing: strike={strike_pct*100:.0f}%, lookback={lookback_days} days", end=' ')

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

            print(f"→ Avg Return: {avg_return:+.1f}%")

        except Exception as e:
            print(f"→ ERROR: {str(e)}")
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

    return pd.DataFrame(results)

def create_multi_period_heatmaps(period_results, save_path=None):
    """
    Create 2x2 subplot of heatmaps for multiple periods

    Parameters:
    period_results: Dict of {period_name: results_df}
    save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    # Common colormap settings
    cmap = 'RdYlGn'
    center = 0

    # Find global min/max for consistent color scale
    all_returns = []
    for results_df in period_results.values():
        all_returns.extend(results_df['avg_return'].dropna().values)

    vmin = min(all_returns)
    vmax = max(all_returns)

    for idx, (period_name, results_df) in enumerate(period_results.items()):
        ax = axes[idx]

        # Create pivot table for heatmap
        pivot_table = results_df.pivot(
            index='rv_lookback_days',
            columns='strike_label',
            values='avg_return'
        )

        # Sort columns by strike percentage (numeric)
        strike_order = sorted(results_df['strike_percent'].unique())
        strike_labels = [f"{s*100:.0f}%" for s in strike_order]
        pivot_table = pivot_table[strike_labels]

        # Create heatmap
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            center=center,
            linewidths=0.5,
            cbar_kws={'label': 'Avg Return (%)'},
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=True
        )

        # Calculate summary stats for title
        avg_return_all = results_df['avg_return'].mean()
        best_return = results_df['avg_return'].max()
        num_profitable = (results_df['avg_return'] > 0).sum()
        pct_profitable = num_profitable / len(results_df) * 100

        ax.set_title(f'Year {period_name} (1-Year Period)\nAvg: {avg_return_all:+.1f}% | Best: {best_return:+.1f}% | Profitable: {pct_profitable:.0f}%',
                    fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('Strike (% of Spot)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Lookback Days', fontsize=10, fontweight='bold')

    plt.suptitle('Covered Call Strategy: 1-Year Period Comparison (Jan-Dec Each Year)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nMulti-period heatmap saved to: {save_path}")

    plt.show()

def print_period_summary(period_results):
    """
    Print summary comparison across periods

    Parameters:
    period_results: Dict of {period_name: results_df}
    """
    print(f"\n{'='*100}")
    print("MULTI-PERIOD SUMMARY COMPARISON")
    print(f"{'='*100}\n")

    print(f"{'Period':<20} {'Avg Return':<15} {'Best Return':<15} {'Worst Return':<15} {'Best Params':<25}")
    print("-" * 100)

    for period_name, results_df in period_results.items():
        avg_return = results_df['avg_return'].mean()
        best_return = results_df['avg_return'].max()
        worst_return = results_df['avg_return'].min()

        # Find best parameters
        best_row = results_df.loc[results_df['avg_return'].idxmax()]
        best_params = f"{best_row['strike_label']}, {int(best_row['rv_lookback_days'])}d"

        print(f"{period_name:<20} {avg_return:>13.1f}% {best_return:>13.1f}% {worst_return:>13.1f}% {best_params:<25}")

    print("=" * 100)

def main():
    """Main execution"""

    # Configuration
    data_file = "data/processed/HBAR_daily.parquet"

    # Define periods to test (1 year each for fair comparison)
    periods = {
        '2021': ('2021-01-01', '2021-12-31'),
        '2022': ('2022-01-01', '2022-12-31'),
        '2023': ('2023-01-01', '2023-12-31'),
        '2024': ('2024-01-01', '2024-12-31')
    }

    # Parameter grid
    strike_percents = [1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00]
    rv_lookback_days_list = [3, 5, 7, 10, 14, 21, 30]

    # Constants
    vol_spread = 0.10
    risk_free_rate = 0.0
    cost_of_carry = 0.0

    # Load full dataset
    print(f"Loading data from: {data_file}")
    df_full = load_data_from_parquet(data_file)

    # Run optimization for each period
    period_results = {}

    for period_name, (start_date, end_date) in periods.items():
        print(f"\n\n{'#'*80}")
        print(f"PROCESSING: {period_name} ({start_date} to {end_date})")
        print(f"{'#'*80}\n")

        # Filter data for this period
        df_period = filter_data_by_date(df_full, start_date, end_date)

        # Skip if insufficient data
        if len(df_period) < 100:
            print(f"⚠️  Insufficient data for {period_name} (only {len(df_period)} days), skipping...")
            continue

        # Run optimization
        results_df = run_parameter_optimization(
            df_period,
            strike_percents,
            rv_lookback_days_list,
            period_name=period_name,
            vol_spread=vol_spread,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry
        )

        period_results[period_name] = results_df

        # Save individual period results
        output_dir = Path('output/multi_period')
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f'{period_name.replace(" ", "_")}_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to: {csv_path}")

    # Create multi-period heatmap comparison
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    heatmap_path = output_dir / 'parameter_optimization_multi_period_heatmaps.png'
    create_multi_period_heatmaps(period_results, save_path=heatmap_path)

    # Print summary comparison
    print_period_summary(period_results)

    # Save summary comparison to CSV
    summary_data = []
    for period_name, results_df in period_results.items():
        summary_data.append({
            'period': period_name,
            'avg_return': results_df['avg_return'].mean(),
            'best_return': results_df['avg_return'].max(),
            'worst_return': results_df['avg_return'].min(),
            'median_return': results_df['avg_return'].median(),
            'std_return': results_df['avg_return'].std(),
            'num_profitable': (results_df['avg_return'] > 0).sum(),
            'pct_profitable': (results_df['avg_return'] > 0).sum() / len(results_df) * 100
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'multi_period_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved to: {summary_path}")

    print("\n✓ Multi-period parameter optimization complete!")

if __name__ == "__main__":
    main()
