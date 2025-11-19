"""
Compare Old (Percentage-Based) vs New (Dollar-Based) Backtest Implementations

This script runs both implementations side-by-side and compares results to demonstrate
the correction.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest import backtest_short_call_strategy as old_backtest
from backtest_covered_call import backtest_covered_call_strategy as new_backtest

def compare_implementations(df, strike_percent=2.00):
    """
    Run both old and new implementations and compare results

    Args:
        df: DataFrame with HBAR daily price data
        strike_percent: Strike as % of spot (2.00 = 200% for sanity check)
    """
    print("="*100)
    print("COMPARISON: OLD (Percentage-Based) vs NEW (Dollar-Based) IMPLEMENTATION")
    print("="*100)
    print(f"\nTest Parameters:")
    print(f"  Strike: {strike_percent*100:.0f}% of spot")
    print(f"  Lookback: 7 days")
    print(f"  Vol Spread: 10%")

    # Run old implementation
    print("\n" + "-"*100)
    print("Running OLD implementation (percentage-based)...")
    print("-"*100)
    old_results = old_backtest(
        df,
        rv_lookback_days=7,
        strike_percent=strike_percent,
        vol_spread=0.10,
        risk_free_rate=0.0,
        cost_of_carry=0.0
    )

    # Run new implementation
    print("\n" + "-"*100)
    print("Running NEW implementation (dollar-based)...")
    print("-"*100)
    new_results = new_backtest(
        df,
        rv_lookback_days=7,
        strike_percent=strike_percent,
        vol_spread=0.10,
        risk_free_rate=0.0,
        cost_of_carry=0.0
    )

    # Compare results
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)

    print(f"\n{'Scenario':<15} {'Old Final NAV':<15} {'New Final NAV':<15} {'Difference':<15} {'Old Return %':<15} {'New Return %':<15}")
    print("-"*100)

    for scenario_num in range(1, 8):
        scenario_name = f'Scenario_{scenario_num}'

        old_summary = old_results['summary'][scenario_name]
        new_summary = new_results['summary'][scenario_name]

        old_nav = old_summary['final_nav']
        new_nav = new_summary['final_nav']
        diff = new_nav - old_nav

        old_return = old_summary['total_return']
        new_return = new_summary['total_return']

        print(f"{scenario_name:<15} {old_nav:<15.4f} {new_nav:<15.4f} {diff:<15.4f} {old_return:<15.2f} {new_return:<15.2f}")

    # Average comparison
    old_avg_nav = np.mean([s['final_nav'] for s in old_results['summary'].values()])
    new_avg_nav = np.mean([s['final_nav'] for s in new_results['summary'].values()])
    old_avg_return = np.mean([s['total_return'] for s in old_results['summary'].values()])
    new_avg_return = np.mean([s['total_return'] for s in new_results['summary'].values()])

    print("-"*100)
    print(f"{'AVERAGE':<15} {old_avg_nav:<15.4f} {new_avg_nav:<15.4f} {new_avg_nav - old_avg_nav:<15.4f} {old_avg_return:<15.2f} {new_avg_return:<15.2f}")
    print("="*100)

    # Key insights
    print("\nKEY INSIGHTS:")
    print("-"*100)

    # Check for negative NAVs
    old_negative_count = sum(1 for s in old_results['summary'].values() if s['final_nav'] < 0)
    new_negative_count = sum(1 for s in new_results['summary'].values() if s['final_nav'] < 0)

    print(f"1. Negative NAVs:")
    print(f"   Old implementation: {old_negative_count}/7 scenarios ended with negative NAV")
    print(f"   New implementation: {new_negative_count}/7 scenarios ended with negative NAV")

    if old_negative_count > 0 and new_negative_count == 0:
        print(f"   ✓ FIXED: New implementation correctly prevents impossible negative NAVs")
    elif new_negative_count > 0:
        print(f"   ⚠ WARNING: New implementation still has negative NAVs (extreme case)")

    # Check profitability
    old_profitable = sum(1 for s in old_results['summary'].values() if s['final_nav'] > 1.0)
    new_profitable = sum(1 for s in new_results['summary'].values() if s['final_nav'] > 1.0)

    print(f"\n2. Profitability:")
    print(f"   Old implementation: {old_profitable}/7 scenarios profitable (NAV > 1.0)")
    print(f"   New implementation: {new_profitable}/7 scenarios profitable (NAV > 1.0)")

    # Return difference
    return_diff = new_avg_return - old_avg_return
    print(f"\n3. Average Return Difference:")
    print(f"   Old: {old_avg_return:.2f}%")
    print(f"   New: {new_avg_return:.2f}%")
    print(f"   Difference: {return_diff:+.2f}%")

    if abs(return_diff) < 5:
        print(f"   → Returns are similar (difference < 5%)")
    else:
        print(f"   → Significant difference in returns!")

    print("="*100)

    return old_results, new_results

def plot_comparison(old_results, new_results, output_path='output/comparison_old_vs_new.png'):
    """Plot side-by-side comparison of old vs new NAV paths"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#9C27B0', '#00BCD4', '#FF5722']

    # Old implementation
    for i, (scenario_name, scenario_df) in enumerate(old_results['scenarios'].items()):
        if len(scenario_df) > 0:
            ax1.plot(scenario_df['date'], scenario_df['nav'],
                    label=scenario_name, color=colors[i % len(colors)],
                    linewidth=2, alpha=0.7)

    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('NAV', fontsize=12)
    ax1.set_title('OLD: Percentage-Based (Incorrect)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # New implementation
    for i, (scenario_name, scenario_df) in enumerate(new_results['scenarios'].items()):
        if len(scenario_df) > 0:
            ax2.plot(scenario_df['date'], scenario_df['nav'],
                    label=scenario_name, color=colors[i % len(colors)],
                    linewidth=2, alpha=0.7)

    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('NAV', fontsize=12)
    ax2.set_title('NEW: Dollar-Based (Correct)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved to: {output_path}")
    plt.close()

def main():
    # Load HBAR data
    data_path = Path('data/processed/HBAR_daily.parquet')

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please run run_hbar_backtest.py first to generate the daily data.")
        return

    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} days of HBAR price data")

    # Run comparison with 200% strikes (sanity check scenario)
    old_results, new_results = compare_implementations(df, strike_percent=2.00)

    # Plot comparison
    plot_comparison(old_results, new_results)

    print("\n✓ Comparison complete!")
    print("\nSummary:")
    print("  - Old implementation uses percentage-based calculation (incorrect)")
    print("  - New implementation uses dollar-based calculation (correct)")
    print("  - Both should show similar patterns but new implementation is conceptually clearer")
    print("  - New implementation properly tracks HBAR + cash portfolio value")

if __name__ == "__main__":
    main()
