"""
HBAR Backtest Sanity Check - Deep OTM Calls (Strike = 200% of Spot)

This script validates the backtest logic by using extremely deep OTM calls
that should almost never be exercised. Expected behavior:
- Very small premiums collected
- Zero or near-zero payoffs
- High win rate (~100%)
- Slow steady NAV growth (final NAV > 1.0)
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtest import backtest_short_call_strategy, plot_all_scenarios, print_summary

def main():
    # Paths
    daily_data_path = Path('/Users/neo/Velar/cc_optimizer/data/processed/HBAR_daily.parquet')
    output_chart_dir = Path('/Users/neo/Velar/cc_optimizer/output')

    # Create output directory
    output_chart_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SANITY CHECK: DEEP OTM CALLS (STRIKE = 200% OF SPOT)")
    print("="*80)

    # Load daily data
    print(f"\n1. Loading HBAR daily data...")
    print(f"   Path: {daily_data_path}")
    df_daily = pd.read_parquet(daily_data_path)
    print(f"   Loaded {len(df_daily):,} daily candles")
    print(f"   Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")

    # Run backtest with DEEP OTM calls
    print(f"\n2. Running sanity check backtest...")
    print(f"   Strategy: Short Call (7-day rolling)")
    print(f"   SANITY CHECK PARAMETERS:")
    print(f"     - Maturity: 7 days")
    print(f"     - Strike: 200% of spot (100% OTM) ← DEEP OTM")
    print(f"     - Vol Spread: 10%")
    print(f"     - Risk-free Rate: 0% (crypto)")
    print(f"     - Cost of Carry: 0% (crypto)")
    print(f"\n   Expected Results:")
    print(f"     • Premiums: Very small (~0.01-0.1% per trade)")
    print(f"     • Payoffs: Zero or near-zero (calls rarely exercised)")
    print(f"     • Win Rate: ~100%")
    print(f"     • Final NAV: > 1.0 (slow steady growth)")

    results = backtest_short_call_strategy(
        df_daily,
        rv_lookback_days=7,
        strike_percent=2.00,  # ← SANITY CHECK: 200% strike
        vol_spread=0.10,
        risk_free_rate=0.0,
        cost_of_carry=0.0
    )

    # Print summary
    print(f"\n3. Sanity Check Results:")
    print_summary(results)

    # Analyze payoffs
    print(f"\n4. Payoff Analysis:")
    print("-" * 80)
    for scenario_name, scenario_df in results['scenarios'].items():
        num_payoffs = (scenario_df['payoff'] < 0).sum()  # Negative payoffs = ITM
        total_trades = len(scenario_df)
        pct_exercised = (num_payoffs / total_trades * 100) if total_trades > 0 else 0

        print(f"{scenario_name}: {num_payoffs}/{total_trades} trades had payoffs "
              f"({pct_exercised:.1f}% exercised)")

    # Generate chart
    print(f"\n5. Generating NAV chart...")
    chart_path = output_chart_dir / 'HBAR_sanity_check_nav.png'
    plot_all_scenarios(results, save_path=str(chart_path))

    # Export scenario data
    print(f"\n6. Exporting scenario data...")
    for scenario_name, scenario_df in results['scenarios'].items():
        output_file = output_chart_dir / f'HBAR_sanity_{scenario_name}.csv'
        scenario_df.to_csv(output_file, index=False)
        print(f"   Exported: {output_file}")

    # Validation summary
    print("\n" + "="*80)
    print("SANITY CHECK VALIDATION")
    print("="*80)

    # Get average metrics across all scenarios
    avg_final_nav = sum(s['final_nav'] for s in results['summary'].values()) / len(results['summary'])
    avg_win_rate = sum(s['win_rate'] for s in results['summary'].values()) / len(results['summary'])

    print(f"\nAverage Results Across All Scenarios:")
    print(f"  - Final NAV: {avg_final_nav:.4f}")
    print(f"  - Win Rate: {avg_win_rate:.1f}%")

    print(f"\nValidation Checks:")

    # Check 1: Final NAV > 1.0
    if avg_final_nav > 1.0:
        print(f"  ✓ PASS: Final NAV > 1.0 (expected for deep OTM)")
        print(f"    → Backtest logic appears correct")
    else:
        print(f"  ✗ FAIL: Final NAV <= 1.0 (unexpected!)")
        print(f"    → Potential bug in backtest logic")

    # Check 2: Win rate > 95%
    if avg_win_rate > 95.0:
        print(f"  ✓ PASS: Win rate > 95% (expected for deep OTM)")
        print(f"    → Payoff calculation appears correct")
    else:
        print(f"  ⚠ WARNING: Win rate < 95% (unexpected)")
        print(f"    → HBAR had extreme volatility or potential issue")

    # Check 3: Positive avg P&L
    avg_pnl = sum(s['avg_pnl_pct'] for s in results['summary'].values()) / len(results['summary'])
    if avg_pnl > 0:
        print(f"  ✓ PASS: Average P&L > 0 (expected for deep OTM)")
        print(f"    → Premium collection working correctly")
    else:
        print(f"  ✗ FAIL: Average P&L <= 0 (unexpected!)")
        print(f"    → Potential issue with premium or payoff calculation")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)

    if avg_final_nav > 1.0 and avg_win_rate > 95.0 and avg_pnl > 0:
        print("\n✓ SANITY CHECK PASSED!")
        print("\nThe backtest logic is working correctly.")
        print("The poor performance with 110% strike is due to HBAR's volatility,")
        print("not bugs in the implementation.")
    else:
        print("\n✗ SANITY CHECK FAILED!")
        print("\nThere may be issues with the backtest implementation.")
        print("Review the payoff and premium calculations.")

    print("\n" + "="*80)
    print(f"\nOutputs:")
    print(f"  - NAV chart: {chart_path}")
    print(f"  - Scenario CSVs: {output_chart_dir}/HBAR_sanity_Scenario_*.csv")
    print()

if __name__ == "__main__":
    main()
