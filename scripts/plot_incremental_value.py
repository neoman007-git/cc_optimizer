"""
Plot Incremental Value Added by Covered Call Strategy

Shows the value added/lost relative to buy-and-hold HBAR for each year.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_buy_hold_returns():
    """Calculate buy-and-hold HBAR returns for each year"""
    df = pd.read_parquet('data/processed/HBAR_daily.parquet')
    df['date'] = pd.to_datetime(df['date'])

    years = {
        '2021': ('2021-01-01', '2021-12-31'),
        '2022': ('2022-01-01', '2022-12-31'),
        '2023': ('2023-01-01', '2023-12-31'),
        '2024': ('2024-01-01', '2024-12-31')
    }

    buy_hold_returns = {}

    for year, (start_date, end_date) in years.items():
        year_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if len(year_data) > 0:
            start_price = year_data.iloc[0]['close']
            end_price = year_data.iloc[-1]['close']
            bh_return = (end_price / start_price - 1) * 100
            buy_hold_returns[year] = bh_return

    return buy_hold_returns

def create_incremental_value_heatmaps(save_path=None):
    """
    Create 2x2 subplot of heatmaps showing incremental value added
    """
    # Load buy-and-hold returns
    buy_hold_returns = calculate_buy_hold_returns()

    # Load strategy results for each period
    output_dir = Path('output/multi_period')

    period_results = {}
    for year in ['2021', '2022', '2023', '2024']:
        csv_path = output_dir / f'{year}_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Calculate incremental value: strategy return - buy-hold return
            df['incremental_value'] = df['avg_return'] - buy_hold_returns[year]
            period_results[year] = df
        else:
            print(f"Warning: {csv_path} not found")

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    # Common colormap settings (centered at 0)
    cmap = 'RdYlGn'
    center = 0

    # Find global min/max for consistent color scale
    all_incremental = []
    for results_df in period_results.values():
        all_incremental.extend(results_df['incremental_value'].dropna().values)

    vmin = min(all_incremental)
    vmax = max(all_incremental)

    for idx, (year, results_df) in enumerate(period_results.items()):
        ax = axes[idx]

        # Create pivot table for heatmap
        pivot_table = results_df.pivot(
            index='rv_lookback_days',
            columns='strike_label',
            values='incremental_value'
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
            cbar_kws={'label': 'Incremental Value (%)'},
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=True
        )

        # Calculate summary stats
        avg_incremental = results_df['incremental_value'].mean()
        best_incremental = results_df['incremental_value'].max()
        worst_incremental = results_df['incremental_value'].min()

        # Count how many beat buy-hold
        num_beat_bh = (results_df['incremental_value'] > 0).sum()
        pct_beat_bh = num_beat_bh / len(results_df) * 100

        # Get HBAR buy-hold return for reference
        bh_return = buy_hold_returns[year]

        ax.set_title(
            f'Year {year} (HBAR: {bh_return:+.1f}%)\n'
            f'Avg Incremental: {avg_incremental:+.1f}% | Best: {best_incremental:+.1f}% | Beat B&H: {pct_beat_bh:.0f}%',
            fontsize=11, fontweight='bold', pad=15
        )
        ax.set_xlabel('Strike (% of Spot)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Lookback Days', fontsize=10, fontweight='bold')

    plt.suptitle('Covered Call Strategy: Incremental Value vs Buy-and-Hold HBAR\n(Positive = Strategy Outperformed, Negative = Strategy Underperformed)',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nIncremental value heatmap saved to: {save_path}")

    plt.show()

def print_incremental_summary():
    """Print summary of incremental value by year"""
    buy_hold_returns = calculate_buy_hold_returns()

    output_dir = Path('output/multi_period')

    print(f"\n{'='*100}")
    print("INCREMENTAL VALUE ANALYSIS: Covered Call Strategy vs Buy-and-Hold")
    print(f"{'='*100}\n")

    print(f"{'Year':<8} {'HBAR B&H':<12} {'Avg Added':<12} {'Best Added':<12} {'Worst Added':<12} {'% Beat B&H':<12}")
    print("-" * 100)

    for year in ['2021', '2022', '2023', '2024']:
        csv_path = output_dir / f'{year}_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            bh_return = buy_hold_returns[year]
            df['incremental_value'] = df['avg_return'] - bh_return

            avg_incr = df['incremental_value'].mean()
            best_incr = df['incremental_value'].max()
            worst_incr = df['incremental_value'].min()
            pct_beat_bh = (df['incremental_value'] > 0).sum() / len(df) * 100

            print(f"{year:<8} {bh_return:>10.1f}% {avg_incr:>10.1f}% {best_incr:>10.1f}% {worst_incr:>10.1f}% {pct_beat_bh:>10.0f}%")

    print("=" * 100)

def main():
    """Main execution"""

    # Create incremental value heatmaps
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    heatmap_path = output_dir / 'incremental_value_heatmaps.png'
    create_incremental_value_heatmaps(save_path=heatmap_path)

    # Print summary
    print_incremental_summary()

    print("\n✓ Incremental value analysis complete!")

if __name__ == "__main__":
    main()
