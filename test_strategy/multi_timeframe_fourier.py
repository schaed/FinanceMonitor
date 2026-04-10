import alpaca_trade_api as alpaca
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import fft
from scipy.signal import find_peaks
from datetime import datetime, timedelta

# Setup API
YOUR_API_SECRET_KEY = os.getenv('ALPACA_PAPER_KEY')
ALPACA_ID = os.getenv('ALPACA_ID')
api = alpaca.REST(ALPACA_ID, YOUR_API_SECRET_KEY, 'https://paper-api.alpaca.markets/v2')

print("="*60)
print("MULTI-TIMEFRAME FOURIER ANALYSIS STRATEGY")
print("="*60)

# ==========================================
# FETCH MULTI-TIMEFRAME DATA
# ==========================================
print("\n1. Fetching multi-timeframe data...")

# Calculate dates
end_date = "2026-04-05"  # Our analysis end date
end_dt = datetime.strptime(end_date, "%Y-%m-%d")

# 26 weeks back
start_weekly = (end_dt - timedelta(weeks=26)).strftime("%Y-%m-%d")
print(f"   Weekly data: {start_weekly} to {end_date}")

# 28 days back
start_daily = (end_dt - timedelta(days=28)).strftime("%Y-%m-%d")
print(f"   Daily data: {start_daily} to {end_date}")

# 7 days back for minute data
start_minute = (end_dt - timedelta(days=7)).strftime("%Y-%m-%d")
print(f"   Minute data: {start_minute} to {end_date}")

# Fetch data
try:
    df_weekly = api.get_bars(symbol="SPY", start=start_weekly, end=end_date, timeframe="1Week").df
    print(f"   Weekly bars: {len(df_weekly)}")
except Exception as e:
    print(f"   Error fetching weekly: {e}")
    df_weekly = pd.DataFrame()

df_daily = api.get_bars(symbol="SPY", start=start_daily, end=end_date, timeframe="1Day").df
print(f"   Daily bars: {len(df_daily)}")

df_minute = api.get_bars(symbol="SPY", start=start_minute, end=end_date, timeframe="1Min").df
print(f"   Minute bars: {len(df_minute)}")

# ==========================================
# FOURIER TRANSFORM ANALYSIS
# ==========================================
print("\n2. Performing Fourier Transform Analysis...")

def analyze_fourier(prices, timeframe_name, sampling_rate=1):
    """
    Analyze price series using FFT to find dominant frequencies

    Args:
        prices: Price series (close prices)
        timeframe_name: Name for display
        sampling_rate: Samples per unit time (e.g., 1 for daily, 390 for minutes in trading day)

    Returns:
        dict with dominant periods and frequencies
    """
    n = len(prices)

    # Convert to numpy array
    prices_array = prices.values if hasattr(prices, 'values') else np.array(prices)

    # Detrend the data (remove linear trend)
    detrended = prices_array - np.linspace(prices_array[0], prices_array[-1], n)

    # Apply FFT
    fft_vals = fft.fft(detrended)
    fft_freq = fft.fftfreq(n, d=1/sampling_rate)

    # Get power spectrum (magnitude)
    power = np.abs(fft_vals)**2

    # Only look at positive frequencies
    positive_freq_idx = fft_freq > 0
    freqs = fft_freq[positive_freq_idx]
    power = power[positive_freq_idx]

    # Find peaks in power spectrum
    peaks, properties = find_peaks(power, prominence=np.max(power)*0.1)

    # Get top 5 dominant frequencies
    if len(peaks) > 0:
        top_peak_idx = peaks[np.argsort(power[peaks])[-5:]][::-1]  # Top 5
        dominant_freqs = freqs[top_peak_idx]
        dominant_powers = power[top_peak_idx]
        dominant_periods = 1 / dominant_freqs  # Period = 1/frequency
    else:
        dominant_freqs = np.array([])
        dominant_powers = np.array([])
        dominant_periods = np.array([])

    print(f"\n   {timeframe_name} Analysis:")
    print(f"   Data points: {n}")

    if len(dominant_periods) > 0:
        print(f"   Top {len(dominant_periods)} dominant cycles:")
        for i, (period, freq, power_val) in enumerate(zip(dominant_periods, dominant_freqs, dominant_powers)):
            print(f"      {i+1}. Period: {period:.2f} bars (freq: {freq:.4f}, power: {power_val:.0f})")
    else:
        print("   No significant cycles detected")

    return {
        'timeframe': timeframe_name,
        'n_samples': n,
        'freqs': freqs,
        'power': power,
        'dominant_freqs': dominant_freqs,
        'dominant_periods': dominant_periods,
        'dominant_powers': dominant_powers
    }

# Analyze each timeframe
fourier_results = {}

if len(df_weekly) > 10:
    fourier_results['weekly'] = analyze_fourier(
        df_weekly['close'],
        'WEEKLY (26 weeks)',
        sampling_rate=1
    )

if len(df_daily) > 10:
    fourier_results['daily'] = analyze_fourier(
        df_daily['close'],
        'DAILY (28 days)',
        sampling_rate=1
    )

if len(df_minute) > 50:
    # For minute data, use hourly sampling (390 mins per trading day, ~6.5 hours)
    fourier_results['minute'] = analyze_fourier(
        df_minute['close'],
        'MINUTE (7 days)',
        sampling_rate=390  # minutes per trading day
    )

# ==========================================
# DETERMINE OPTIMAL HOLDING PERIODS
# ==========================================
print("\n3. Determining optimal holding periods...")

def convert_period_to_minutes(period, timeframe):
    """Convert period in native units to minute bars"""
    if timeframe == 'weekly':
        # 1 week = 5 trading days = 5 * 390 minutes
        return period * 5 * 390
    elif timeframe == 'daily':
        # 1 day = 390 trading minutes
        return period * 390
    else:  # minute
        return period

optimal_periods = {}

for tf_name, results in fourier_results.items():
    if len(results['dominant_periods']) > 0:
        # Get the strongest cycle (highest power)
        strongest_period = results['dominant_periods'][0]

        # Convert to minutes
        period_in_minutes = convert_period_to_minutes(strongest_period, tf_name)

        # For trading, we'll use fractions of the cycle
        # Entry to exit should be roughly 1/4 to 1/2 of the cycle
        hold_time_min = period_in_minutes / 4
        hold_time_max = period_in_minutes / 2

        optimal_periods[tf_name] = {
            'cycle_period_native': strongest_period,
            'cycle_period_minutes': period_in_minutes,
            'suggested_hold_min': hold_time_min,
            'suggested_hold_max': hold_time_max
        }

        print(f"\n   {tf_name.upper()}:")
        print(f"      Dominant cycle: {strongest_period:.2f} {tf_name} bars")
        print(f"      = {period_in_minutes:.0f} minute bars")
        print(f"      Suggested hold time: {hold_time_min:.0f} - {hold_time_max:.0f} minutes")
        print(f"      = {hold_time_min/390:.2f} - {hold_time_max/390:.2f} trading days")

# ==========================================
# SYNTHESIZE MULTI-TIMEFRAME SIGNAL
# ==========================================
print("\n4. Synthesizing multi-timeframe trading rules...")

# Determine consensus holding period
all_hold_mins = [p['suggested_hold_min'] for p in optimal_periods.values()]
all_hold_maxs = [p['suggested_hold_max'] for p in optimal_periods.values()]

if len(all_hold_mins) > 0:
    # Weight by timeframe (weekly most important, then daily, then minute)
    weights = []
    values_min = []
    values_max = []

    if 'weekly' in optimal_periods:
        weights.append(3.0)  # Highest weight
        values_min.append(optimal_periods['weekly']['suggested_hold_min'])
        values_max.append(optimal_periods['weekly']['suggested_hold_max'])

    if 'daily' in optimal_periods:
        weights.append(2.0)  # Medium weight
        values_min.append(optimal_periods['daily']['suggested_hold_min'])
        values_max.append(optimal_periods['daily']['suggested_hold_max'])

    if 'minute' in optimal_periods:
        weights.append(1.0)  # Lowest weight
        values_min.append(optimal_periods['minute']['suggested_hold_min'])
        values_max.append(optimal_periods['minute']['suggested_hold_max'])

    # Weighted average
    consensus_hold_min = np.average(values_min, weights=weights)
    consensus_hold_max = np.average(values_max, weights=weights)

    print(f"\n   CONSENSUS HOLDING PERIOD:")
    print(f"      Min: {consensus_hold_min:.0f} minutes ({consensus_hold_min/390:.2f} days)")
    print(f"      Max: {consensus_hold_max:.0f} minutes ({consensus_hold_max/390:.2f} days)")
else:
    print("\n   WARNING: Could not determine optimal holding period")
    consensus_hold_min = 3  # Default fallback
    consensus_hold_max = 60

# ==========================================
# IMPLEMENT TRADING STRATEGY
# ==========================================
print("\n5. Implementing Fourier-informed trading strategy...")

# Calculate technical indicators on minute data
df = df_minute.copy()

df['returns'] = df['close'].pct_change()
df['price_change'] = df['close'].diff()
df['upward'] = (df['price_change'] > 0).astype(int)
df['downward'] = (df['price_change'] < 0).astype(int)

# Moving averages
for period in [5, 10, 20, 30]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

# Momentum
df['momentum_5'] = df['close'] - df['close'].shift(5)
df['momentum_10'] = df['close'] - df['close'].shift(10)
df['acceleration'] = df['momentum_5'].diff()

# Volume
df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma_20']

# RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi_14'] = calculate_rsi(df['close'], window=14)

# MACD
df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# Consecutive bars
df['consecutive_up'] = 0
df['consecutive_down'] = 0
up_count = 0
down_count = 0

for i in range(len(df)):
    if df['upward'].iloc[i] == 1:
        up_count += 1
        down_count = 0
        df.iloc[i, df.columns.get_loc('consecutive_up')] = up_count
    elif df['downward'].iloc[i] == 1:
        down_count += 1
        up_count = 0
        df.iloc[i, df.columns.get_loc('consecutive_down')] = down_count
    else:
        up_count = 0
        down_count = 0

# Trend alignment
df['bullish_alignment'] = (
    (df['close'] > df['sma_10']).astype(int) +
    (df['close'] > df['sma_20']).astype(int) +
    (df['sma_10'] > df['sma_20']).astype(int) +
    (df['ema_5'] > df['ema_10']).astype(int) +
    (df['momentum_5'] > 0).astype(int) +
    (df['macd'] > df['macd_signal']).astype(int)
)

df['bearish_alignment'] = (
    (df['close'] < df['sma_10']).astype(int) +
    (df['close'] < df['sma_20']).astype(int) +
    (df['sma_10'] < df['sma_20']).astype(int) +
    (df['ema_5'] < df['ema_10']).astype(int) +
    (df['momentum_5'] < 0).astype(int) +
    (df['macd'] < df['macd_signal']).astype(int)
)

# Strategy parameters (informed by Fourier analysis)
TRANSACTION_COST = 0.0001
HOLD_MIN = int(consensus_hold_min)
HOLD_MAX = int(consensus_hold_max)
STOP_LOSS_PCT = 0.002  # 0.2%
TAKE_PROFIT_PCT = 0.03  # 0.4%

print(f"\n   Using Fourier-informed holding period:")
print(f"      Min hold: {HOLD_MIN} bars")
print(f"      Max hold: {HOLD_MAX} bars")
print(f"      Stop loss: {STOP_LOSS_PCT*100:.2f}%")
print(f"      Take profit: {TAKE_PROFIT_PCT*100:.2f}%")

# Generate signals
df['long_signal'] = 0
df['short_signal'] = 0
start_idx = 55

for i in range(start_idx, len(df)):
    current_idx = df.index[i]

    # LONG ENTRY
    long_c1 = df.loc[current_idx, 'consecutive_up'] >= 2
    long_c2 = df.loc[current_idx, 'bullish_alignment'] >= 5
    long_c3 = 45 < df.loc[current_idx, 'rsi_14'] < 65
    long_c4 = df.loc[current_idx, 'momentum_5'] > 0
    long_c5 = df.loc[current_idx, 'volume_ratio'] > 1.0

    if all([long_c1, long_c2, long_c3, long_c4, long_c5]):
        df.loc[current_idx, 'long_signal'] = 1

    # SHORT ENTRY
    short_c1 = df.loc[current_idx, 'consecutive_down'] >= 2
    short_c2 = df.loc[current_idx, 'bearish_alignment'] >= 5
    short_c3 = 35 < df.loc[current_idx, 'rsi_14'] < 55
    short_c4 = df.loc[current_idx, 'momentum_5'] < 0
    short_c5 = df.loc[current_idx, 'volume_ratio'] > 1.0

    if all([short_c1, short_c2, short_c3, short_c4, short_c5]):
        df.loc[current_idx, 'short_signal'] = 1

print(f"   Long signals: {df['long_signal'].sum()}")
print(f"   Short signals: {df['short_signal'].sum()}")

# Execute trades
trades = []
position = None

for i in range(start_idx, len(df)):
    current_idx = df.index[i]
    current_price = df.loc[current_idx, 'close']
    current_high = df.loc[current_idx, 'high']
    current_low = df.loc[current_idx, 'low']

    # Manage position
    if position is not None:
        bars_held = i - position['entry_idx']
        position_type = position['type']

        if position_type == 'long':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        exit_reason = None
        exit_price = current_price

        if position_type == 'long':
            if current_low <= position['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = position['stop_loss']
            elif current_high >= position['take_profit']:
                exit_reason = 'take_profit'
                exit_price = position['take_profit']
            elif bars_held >= HOLD_MAX:
                exit_reason = 'max_time'
            elif bars_held >= HOLD_MIN and df.loc[current_idx, 'downward'] == 1:
                exit_reason = 'trend_reversal'
        else:
            if current_high >= position['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = position['stop_loss']
            elif current_low <= position['take_profit']:
                exit_reason = 'take_profit'
                exit_price = position['take_profit']
            elif bars_held >= HOLD_MAX:
                exit_reason = 'max_time'
            elif bars_held >= HOLD_MIN and df.loc[current_idx, 'upward'] == 1:
                exit_reason = 'trend_reversal'

        if exit_reason:
            if position_type == 'long':
                gross_pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:
                gross_pnl = (position['entry_price'] - exit_price) / position['entry_price']

            net_pnl = gross_pnl - (2 * TRANSACTION_COST)

            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': current_idx,
                'exit_price': exit_price,
                'bars_held': bars_held,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason,
                'position_type': position_type
            })

            position = None

    # Enter position
    if position is None:
        if df.loc[current_idx, 'long_signal'] == 1:
            position = {
                'entry_idx': i,
                'entry_time': current_idx,
                'entry_price': current_price,
                'type': 'long',
                'stop_loss': current_price * (1 - STOP_LOSS_PCT),
                'take_profit': current_price * (1 + TAKE_PROFIT_PCT)
            }
        elif df.loc[current_idx, 'short_signal'] == 1:
            position = {
                'entry_idx': i,
                'entry_time': current_idx,
                'entry_price': current_price,
                'type': 'short',
                'stop_loss': current_price * (1 + STOP_LOSS_PCT),
                'take_profit': current_price * (1 - TAKE_PROFIT_PCT)
            }

# Analyze results
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    print(f"\n{'='*60}")
    print("FOURIER-INFORMED STRATEGY RESULTS")
    print('='*60)

    print(f"\nTotal positions: {len(trades_df)}")
    long_trades = trades_df[trades_df['position_type'] == 'long']
    short_trades = trades_df[trades_df['position_type'] == 'short']
    print(f"  Long: {len(long_trades)}")
    print(f"  Short: {len(short_trades)}")

    winning = (trades_df['net_pnl'] > 0).sum()
    losing = (trades_df['net_pnl'] < 0).sum()
    print(f"\nWinning: {winning} ({winning/len(trades_df)*100:.1f}%)")
    print(f"Losing: {losing} ({losing/len(trades_df)*100:.1f}%)")

    if len(long_trades) > 0:
        print(f"\nLONG:")
        print(f"  Wins: {(long_trades['net_pnl'] > 0).sum()} ({(long_trades['net_pnl'] > 0).sum()/len(long_trades)*100:.1f}%)")
        print(f"  Total P&L: {long_trades['net_pnl'].sum()*100:.4f}%")
        print(f"  Avg hold: {long_trades['bars_held'].mean():.1f} bars")

    if len(short_trades) > 0:
        print(f"\nSHORT:")
        print(f"  Wins: {(short_trades['net_pnl'] > 0).sum()} ({(short_trades['net_pnl'] > 0).sum()/len(short_trades)*100:.1f}%)")
        print(f"  Total P&L: {short_trades['net_pnl'].sum()*100:.4f}%")
        print(f"  Avg hold: {short_trades['bars_held'].mean():.1f} bars")

    total_pnl = trades_df['net_pnl'].sum()
    print(f"\nTOTAL P&L: {total_pnl*100:.4f}%")
    print(f"Avg P&L: {trades_df['net_pnl'].mean()*100:.4f}%")
    print(f"Best: {trades_df['net_pnl'].max()*100:.4f}%")
    print(f"Worst: {trades_df['net_pnl'].min()*100:.4f}%")

    print(f"\nExit Reasons:")
    for reason, count in trades_df['exit_reason'].value_counts().items():
        print(f"  {reason}: {count} ({count/len(trades_df)*100:.1f}%)")

    avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()
    avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean()) if losing > 0 else 0

    if losing > 0:
        print(f"\nProfit Factor: {(avg_win * winning) / (avg_loss * losing):.2f}")
        print(f"Win/Loss Ratio: {avg_win/avg_loss:.2f}")

    # Sharpe
    if trades_df['net_pnl'].std() > 0:
        days_traded = 7
        trades_per_day = len(trades_df) / days_traded
        annual_return = trades_df['net_pnl'].mean() * trades_per_day * 252
        annual_vol = trades_df['net_pnl'].std() * np.sqrt(trades_per_day * 252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        print(f"\nSharpe Ratio: {sharpe:.2f}")

    # Drawdown
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
    cumulative_returns = (1 + trades_df['net_pnl']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    print(f"Max Drawdown: {drawdown.min()*100:.4f}%")

    # Save
    trades_df.to_csv('fourier_trades.csv', index=False)
    print(f"\nSaved: fourier_trades.csv")

else:
    print("\nNo trades executed.")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
