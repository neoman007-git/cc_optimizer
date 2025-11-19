"""
HBAR Backtest - Resample hourly data to daily and run backtest
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtest import backtest_short_call_strategy, plot_all_scenarios, print_summary

def resample_to_daily(df):
    """
    Resample hourly OHLCV data to daily candles

    Parameters:
    df: DataFrame with hourly OHLCV data

    Returns:
    DataFrame with daily OHLCV data
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index(pd.to_datetime(df['timestamp']))
        elif 'date' in df.columns:
            df = df.set_index(pd.to_datetime(df['date']))
        else:
            raise ValueError("No datetime column or index found")

    # If index is already datetime, use it directly

    # Resample to daily
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Remove any rows with NaN (incomplete days)
    daily_df = daily_df.dropna()

    # Reset index to have date as column
    daily_df = daily_df.reset_index()

    # Rename index column to 'date' if it's not already named that
    if daily_df.columns[0] != 'date':
        daily_df = daily_df.rename(columns={daily_df.columns[0]: 'date'})

    return daily_df

def main():
    # Paths
    raw_data_path = Path('/Users/neo/Velar/hist_vol_model/data/raw/BINANCE_HBAR_OHLCV_1HRS_2019-01-01_2025-11-17.parquet')
    output_data_dir = Path('/Users/neo/Velar/cc_optimizer/data/processed')
    output_chart_dir = Path('/Users/neo/Velar/cc_optimizer/output')

    # Create output directories
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_chart_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HBAR BACKTEST - SHORT CALL STRATEGY")
    print("="*80)

    # Load hourly data
    print(f"\n1. Loading hourly HBAR data from hist_vol_model...")
    print(f"   Path: {raw_data_path}")
    df_hourly = pd.read_parquet(raw_data_path)
    print(f"   Loaded {len(df_hourly):,} hourly candles")
    print(f"   Columns: {list(df_hourly.columns)}")
    print(f"   Index: {df_hourly.index.name}")
    print(f"   Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")

    # Resample to daily
    print(f"\n2. Resampling to daily candles...")
    df_daily = resample_to_daily(df_hourly)
    print(f"   Resampled to {len(df_daily):,} daily candles")
    print(f"   Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
    print(f"   Price range: ${df_daily['close'].min():.6f} to ${df_daily['close'].max():.6f}")

    # Save daily data
    daily_data_path = output_data_dir / 'HBAR_daily.parquet'
    df_daily.to_parquet(daily_data_path, index=False)
    print(f"   Saved daily data to: {daily_data_path}")

    # Run backtest
    print(f"\n3. Running backtest...")
    print(f"   Strategy: Short Call (7-day rolling)")
    print(f"   Parameters:")
    print(f"     - Maturity: 7 days")
    print(f"     - Strike: 110% of spot (10% OTM)")
    print(f"     - Vol Spread: 10%")
    print(f"     - Risk-free Rate: 0% (crypto)")
    print(f"     - Cost of Carry: 0% (crypto)")

    results = backtest_short_call_strategy(
        df_daily,
        rv_lookback_days=7,
        strike_percent=1.10,
        vol_spread=0.10,
        risk_free_rate=0.0,  # Changed to 0 for crypto
        cost_of_carry=0.0    # Changed to 0 for crypto
    )

    # Print summary
    print(f"\n4. Backtest Results:")
    print_summary(results)

    # Generate chart
    print(f"\n5. Generating NAV chart...")
    chart_path = output_chart_dir / 'HBAR_all_scenarios_nav.png'
    plot_all_scenarios(results, save_path=str(chart_path))

    # Export scenario data
    print(f"\n6. Exporting scenario data...")
    for scenario_name, scenario_df in results['scenarios'].items():
        output_file = output_chart_dir / f'HBAR_{scenario_name}.csv'
        scenario_df.to_csv(output_file, index=False)
        print(f"   Exported: {output_file}")

    print("\n" + "="*80)
    print("✓ HBAR BACKTEST COMPLETE!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - Daily data: {daily_data_path}")
    print(f"  - NAV chart: {chart_path}")
    print(f"  - Scenario CSVs: {output_chart_dir}/HBAR_Scenario_*.csv")
    print()

if __name__ == "__main__":
    main()
