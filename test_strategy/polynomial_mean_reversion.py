import alpaca_trade_api as alpaca
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Setup API
YOUR_API_SECRET_KEY = os.getenv('ALPACA_PAPER_KEY')
ALPACA_ID = os.getenv('ALPACA_ID')
api = alpaca.REST(ALPACA_ID, YOUR_API_SECRET_KEY, 'https://paper-api.alpaca.markets/v2')

print("="*80)
print("POLYNOMIAL MEAN REVERSION STRATEGY")
print("Quadratic Regression Bands with Standard Deviation Entry/Exit")
print("="*80)

# ==========================================
# FETCH DATA
# ==========================================
print("\n1. Fetching multi-timeframe data...")

end_date = "2026-04-05"
end_dt = datetime.strptime(end_date, "%Y-%m-%d")

# 1-minute: last 7 days
start_1min = (end_dt - timedelta(days=7)).strftime("%Y-%m-%d")
df_1min = api.get_bars(symbol="SPY", start=start_1min, end=end_date, timeframe="1Min").df
print(f"   1-minute bars (7 days): {len(df_1min)}")

# 15-minute: last 2 weeks
start_15min = (end_dt - timedelta(days=14)).strftime("%Y-%m-%d")
df_15min = api.get_bars(symbol="SPY", start=start_15min, end=end_date, timeframe="15Min").df
print(f"   15-minute bars (14 days): {len(df_15min)}")

# 1-hour: last 2 months
start_1hour = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
df_1hour = api.get_bars(symbol="SPY", start=start_1hour, end=end_date, timeframe="1Hour").df
print(f"   1-hour bars (60 days): {len(df_1hour)}")

# Daily: last 365 days
start_daily = (end_dt - timedelta(days=365)).strftime("%Y-%m-%d")
df_daily = api.get_bars(symbol="SPY", start=start_daily, end=end_date, timeframe="1Day").df
print(f"   Daily bars (365 days): {len(df_daily)}")

# ==========================================
# POLYNOMIAL REGRESSION FUNCTIONS
# ==========================================

def calculate_polynomial_bands(df, window_size, poly_degree=2):
    """
    Calculate polynomial regression mean and standard deviation bands

    Args:
        df: DataFrame with OHLC data
        window_size: Number of bars to use for polynomial fit
        poly_degree: Degree of polynomial (2 = quadratic)

    Returns:
        DataFrame with polynomial fit and bands
    """
    df = df.copy()

    # Initialize columns
    df['poly_mean'] = np.nan
    df['poly_std'] = np.nan
    df['upper_1std'] = np.nan
    df['lower_1std'] = np.nan
    df['upper_2std'] = np.nan
    df['lower_2std'] = np.nan
    df['upper_3std'] = np.nan
    df['lower_3std'] = np.nan
    df['distance_from_mean'] = np.nan
    df['std_distance'] = np.nan  # Distance in terms of std devs

    # Need at least window_size + poly_degree points
    min_points = max(window_size, poly_degree + 5)

    for i in range(min_points, len(df)):
        # Get window of data
        window = df.iloc[i-window_size:i+1]

        # Create x values (time indices)
        x = np.arange(len(window))
        y = window['close'].values

        # Fit polynomial
        try:
            coeffs = np.polyfit(x, y, poly_degree)
            poly_fit = np.polyval(coeffs, x)

            # Calculate residuals
            residuals = y - poly_fit
            std_dev = np.std(residuals)

            # Current point (last in window)
            current_poly_value = poly_fit[-1]
            current_price = y[-1]

            # Store values
            df.iloc[i, df.columns.get_loc('poly_mean')] = current_poly_value
            df.iloc[i, df.columns.get_loc('poly_std')] = std_dev
            df.iloc[i, df.columns.get_loc('upper_1std')] = current_poly_value + std_dev
            df.iloc[i, df.columns.get_loc('lower_1std')] = current_poly_value - std_dev
            df.iloc[i, df.columns.get_loc('upper_2std')] = current_poly_value + 2 * std_dev
            df.iloc[i, df.columns.get_loc('lower_2std')] = current_poly_value - 2 * std_dev
            df.iloc[i, df.columns.get_loc('upper_3std')] = current_poly_value + 3 * std_dev
            df.iloc[i, df.columns.get_loc('lower_3std')] = current_poly_value - 3 * std_dev

            # Distance from mean
            distance = current_price - current_poly_value
            df.iloc[i, df.columns.get_loc('distance_from_mean')] = distance

            # Distance in terms of standard deviations
            if std_dev > 0:
                df.iloc[i, df.columns.get_loc('std_distance')] = distance / std_dev

        except Exception as e:
            # If polynomial fit fails, skip
            continue

    return df

# ==========================================
# MEAN REVERSION TRADING STRATEGY
# ==========================================

def execute_mean_reversion_strategy(df, timeframe_name, poly_degree=2):
    """
    Execute mean reversion strategy based on polynomial bands

    Entry Rules:
    - Long: Price 2+ std devs below mean, size = floor(std_distance) x base position
    - Short: Price 2+ std devs above mean, size = floor(std_distance) x base position

    Exit Rules:
    - Close when price returns to mean (crosses polynomial)

    Args:
        df: DataFrame with polynomial bands already calculated
        timeframe_name: Name for reporting
        poly_degree: Degree of polynomial

    Returns:
        trades_df: DataFrame with all trades
    """

    print(f"\n{'='*80}")
    print(f"TRADING STRATEGY: {timeframe_name}")
    print(f"{'='*80}")

    TRANSACTION_COST = 0.0001
    BASE_POSITION_SIZE = 1.0  # 1x base position

    trades = []
    position = None

    # Start after bands are calculated
    start_idx = df['poly_mean'].first_valid_index()
    if start_idx is None:
        print(f"   No valid polynomial data for {timeframe_name}")
        return pd.DataFrame()

    start_loc = df.index.get_loc(start_idx)

    for i in range(start_loc, len(df)):
        current_idx = df.index[i]
        current_price = df.loc[current_idx, 'close']
        poly_mean = df.loc[current_idx, 'poly_mean']
        std_distance = df.loc[current_idx, 'std_distance']

        # Skip if no valid data
        if pd.isna(poly_mean) or pd.isna(std_distance):
            continue

        # Manage existing position
        if position is not None:
            position_type = position['type']
            position_size = position['size']
            exit_reason = None
            exit_price = current_price

            # Calculate current P&L
            if position_type == 'long':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                # Exit condition: price returns to mean (crosses above)
                if current_price >= poly_mean:
                    exit_reason = 'mean_reversion'

            else:  # short
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

                # Exit condition: price returns to mean (crosses below)
                if current_price <= poly_mean:
                    exit_reason = 'mean_reversion'

            # Close position if exit triggered
            if exit_reason:
                gross_pnl = pnl_pct * position_size  # Scale by position size
                net_pnl = gross_pnl - (2 * TRANSACTION_COST * position_size)

                bars_held = i - position['entry_idx']

                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'entry_std_distance': position['entry_std_distance'],
                    'exit_time': current_idx,
                    'exit_price': exit_price,
                    'bars_held': bars_held,
                    'position_type': position_type,
                    'position_size': position_size,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason
                })

                position = None

        # Enter new position if no position exists
        if position is None:
            # Long entry: price is 2+ std devs BELOW mean
            if std_distance <= -2.0:
                # Position size = floor of abs(std_distance)
                position_size = max(1.0, np.floor(abs(std_distance)))

                position = {
                    'entry_idx': i,
                    'entry_time': current_idx,
                    'entry_price': current_price,
                    'entry_std_distance': std_distance,
                    'type': 'long',
                    'size': position_size
                }

            # Short entry: price is 2+ std devs ABOVE mean
            elif std_distance >= 2.0:
                # Position size = floor of std_distance
                position_size = max(1.0, np.floor(std_distance))

                position = {
                    'entry_idx': i,
                    'entry_time': current_idx,
                    'entry_price': current_price,
                    'entry_std_distance': std_distance,
                    'type': 'short',
                    'size': position_size
                }

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    if len(trades_df) > 0:
        print(f"\n   Total trades: {len(trades_df)}")

        long_trades = trades_df[trades_df['position_type'] == 'long']
        short_trades = trades_df[trades_df['position_type'] == 'short']

        print(f"   Long: {len(long_trades)}, Short: {len(short_trades)}")

        winning = (trades_df['net_pnl'] > 0).sum()
        print(f"   Win rate: {winning/len(trades_df)*100:.1f}%")

        total_pnl = trades_df['net_pnl'].sum()
        print(f"   Total P&L: {total_pnl*100:.2f}%")

        print(f"   Avg position size: {trades_df['position_size'].mean():.2f}x")
        print(f"   Max position size: {trades_df['position_size'].max():.1f}x")

        if len(long_trades) > 0:
            print(f"\n   LONG Performance:")
            print(f"      Trades: {len(long_trades)}")
            print(f"      Win rate: {(long_trades['net_pnl']>0).sum()/len(long_trades)*100:.1f}%")
            print(f"      Total P&L: {long_trades['net_pnl'].sum()*100:.2f}%")
            print(f"      Avg entry: {long_trades['entry_std_distance'].mean():.2f} std devs below mean")

        if len(short_trades) > 0:
            print(f"\n   SHORT Performance:")
            print(f"      Trades: {len(short_trades)}")
            print(f"      Win rate: {(short_trades['net_pnl']>0).sum()/len(short_trades)*100:.1f}%")
            print(f"      Total P&L: {short_trades['net_pnl'].sum()*100:.2f}%")
            print(f"      Avg entry: {short_trades['entry_std_distance'].mean():.2f} std devs above mean")
    else:
        print(f"   No trades executed")

    return trades_df

# ==========================================
# RUN STRATEGY ON 1-MINUTE BARS
# ==========================================
print("\n2. Running polynomial mean reversion strategy on 1-MINUTE bars...")

# Calculate polynomial bands (use full window)
window_size_1min = len(df_1min) // 2  # Use half of available data as rolling window
print(f"   Using rolling window: {window_size_1min} bars")

df_1min_with_bands = calculate_polynomial_bands(df_1min, window_size=window_size_1min, poly_degree=2)

# Execute strategy
trades_1min = execute_mean_reversion_strategy(df_1min_with_bands, "1-MINUTE (7 days)")

# ==========================================
# RUN ON OTHER TIMEFRAMES
# ==========================================
print("\n3. Running strategy on additional timeframes...")

# 15-minute
print("\n   15-MINUTE timeframe:")
window_size_15min = len(df_15min) // 2
df_15min_with_bands = calculate_polynomial_bands(df_15min, window_size=window_size_15min, poly_degree=2)
trades_15min = execute_mean_reversion_strategy(df_15min_with_bands, "15-MINUTE (14 days)")

# 1-hour
print("\n   1-HOUR timeframe:")
window_size_1hour = len(df_1hour) // 2
df_1hour_with_bands = calculate_polynomial_bands(df_1hour, window_size=window_size_1hour, poly_degree=2)
trades_1hour = execute_mean_reversion_strategy(df_1hour_with_bands, "1-HOUR (60 days)")

# Daily
print("\n   DAILY timeframe:")
window_size_daily = len(df_daily) // 2
df_daily_with_bands = calculate_polynomial_bands(df_daily, window_size=window_size_daily, poly_degree=2)
trades_daily = execute_mean_reversion_strategy(df_daily_with_bands, "DAILY (365 days)")

# ==========================================
# SAVE RESULTS
# ==========================================
print("\n4. Saving results...")

if len(trades_1min) > 0:
    trades_1min.to_csv('mean_reversion_1min.csv', index=False)
    print("   Saved: mean_reversion_1min.csv")

if len(trades_15min) > 0:
    trades_15min.to_csv('mean_reversion_15min.csv', index=False)
    print("   Saved: mean_reversion_15min.csv")

if len(trades_1hour) > 0:
    trades_1hour.to_csv('mean_reversion_1hour.csv', index=False)
    print("   Saved: mean_reversion_1hour.csv")

if len(trades_daily) > 0:
    trades_daily.to_csv('mean_reversion_daily.csv', index=False)
    print("   Saved: mean_reversion_daily.csv")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*80)
print("MULTI-TIMEFRAME SUMMARY")
print("="*80)

summary_data = []

for name, trades_df in [
    ('1-Minute', trades_1min),
    ('15-Minute', trades_15min),
    ('1-Hour', trades_1hour),
    ('Daily', trades_daily)
]:
    if len(trades_df) > 0:
        summary_data.append({
            'Timeframe': name,
            'Trades': len(trades_df),
            'Win Rate': f"{(trades_df['net_pnl']>0).sum()/len(trades_df)*100:.1f}%",
            'Total P&L': f"{trades_df['net_pnl'].sum()*100:.2f}%",
            'Avg Size': f"{trades_df['position_size'].mean():.2f}x",
            'Max Size': f"{trades_df['position_size'].max():.0f}x"
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
else:
    print("No trades executed on any timeframe")

print("\n" + "="*80)
print("Strategy Details:")
print("  - Entry: 2+ std devs from polynomial mean")
print("  - Position sizing: 1x per std dev (2 std = 2x, 3 std = 3x, etc.)")
print("  - Exit: When price returns to polynomial mean")
print("  - Polynomial: Quadratic (degree 2)")
print("  - Rolling window: 50% of available data")
print("="*80)

print("\nNext step: Run visualize_mean_reversion.py to see charts")
print("Then: Run multi_timeframe_decision.py to combine signals")
