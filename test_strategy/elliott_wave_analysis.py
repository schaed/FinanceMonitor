import alpaca_trade_api as alpaca
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, argrelextrema

# Setup API
YOUR_API_SECRET_KEY = os.getenv('ALPACA_PAPER_KEY')
ALPACA_ID = os.getenv('ALPACA_ID')
api = alpaca.REST(ALPACA_ID, YOUR_API_SECRET_KEY, 'https://paper-api.alpaca.markets/v2')

print("="*70)
print("ELLIOTT WAVE ANALYSIS FOR SPY")
print("="*70)

# ==========================================
# FETCH MULTI-TIMEFRAME DATA
# ==========================================
print("\n1. Fetching multi-timeframe data for Elliott Wave analysis...")

# For Elliott Wave, we need enough data to see wave structures
# Daily: 6 months for higher degree waves
# 15-min: 30 days for intermediate waves
# 1-min: 7 days for minor waves

end_date = "2026-04-05"

# Daily data for higher degree waves
df_daily = api.get_bars(symbol="SPY", start="2025-10-01", end=end_date, timeframe="1Day").df
print(f"   Daily bars: {len(df_daily)}")

# 15-minute data for trading signals
df_15min = api.get_bars(symbol="SPY", start="2026-03-01", end=end_date, timeframe="15Min").df
print(f"   15-min bars: {len(df_15min)}")

# 1-minute data for execution
df_1min = api.get_bars(symbol="SPY", start="2026-03-29", end=end_date, timeframe="1Min").df
print(f"   1-min bars: {len(df_1min)}")

# ==========================================
# ELLIOTT WAVE DETECTION ALGORITHM
# ==========================================
print("\n2. Detecting Elliott Wave structure...")

def find_swing_points(df, order=5):
    """
    Find swing highs and lows (pivot points)

    Args:
        df: DataFrame with OHLC data
        order: Number of bars on each side for pivot detection

    Returns:
        DataFrame with swing_high and swing_low columns
    """
    df = df.copy()

    # Find local maxima (swing highs)
    high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
    df['swing_high'] = np.nan
    df.iloc[high_indices, df.columns.get_loc('swing_high')] = df.iloc[high_indices]['high']

    # Find local minima (swing lows)
    low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]
    df['swing_low'] = np.nan
    df.iloc[low_indices, df.columns.get_loc('swing_low')] = df.iloc[low_indices]['low']

    return df

def identify_elliott_waves(df, swing_type='both'):
    """
    Identify potential Elliott Wave patterns in swing points

    Elliott Wave Rules:
    - Impulse: 5 waves (1-2-3-4-5)
      * Wave 2 never retraces more than 100% of wave 1
      * Wave 3 is never the shortest
      * Wave 4 never enters wave 1 price territory
    - Correction: 3 waves (A-B-C)

    Args:
        df: DataFrame with swing_high and swing_low
        swing_type: 'up' for bullish impulse, 'down' for bearish, 'both'

    Returns:
        List of wave patterns with labels
    """
    # Extract pivot points
    swing_highs = df[df['swing_high'].notna()][['swing_high']].copy()
    swing_highs['price'] = swing_highs['swing_high']
    swing_highs['type'] = 'high'

    swing_lows = df[df['swing_low'].notna()][['swing_low']].copy()
    swing_lows['price'] = swing_lows['swing_low']
    swing_lows['type'] = 'low'

    # Combine and sort by time
    pivots = pd.concat([swing_highs[['price', 'type']], swing_lows[['price', 'type']]])
    pivots = pivots.sort_index()

    print(f"   Found {len(pivots)} pivot points")

    # Analyze wave patterns
    waves = []

    # Look for 5-wave impulse patterns (bullish)
    if swing_type in ['up', 'both']:
        waves.extend(find_impulse_waves(pivots, direction='up'))

    # Look for 5-wave impulse patterns (bearish)
    if swing_type in ['down', 'both']:
        waves.extend(find_impulse_waves(pivots, direction='down'))

    return waves, pivots

def find_impulse_waves(pivots, direction='up'):
    """
    Find 5-wave impulse patterns

    Bullish Impulse: Low-High-Low-High-Low-High (1-2-3-4-5)
    Bearish Impulse: High-Low-High-Low-High-Low
    """
    waves = []

    # Need at least 6 pivots for a 5-wave pattern
    if len(pivots) < 6:
        return waves

    # Scan through pivots looking for 5-wave patterns
    for i in range(len(pivots) - 5):
        pattern = pivots.iloc[i:i+6]

        if direction == 'up':
            # Check for alternating Low-High-Low-High-Low-High
            expected_sequence = ['low', 'high', 'low', 'high', 'low', 'high']
        else:
            # Check for alternating High-Low-High-Low-High-Low
            expected_sequence = ['high', 'low', 'high', 'low', 'high', 'low']

        actual_sequence = pattern['type'].tolist()

        if actual_sequence == expected_sequence:
            # Validate Elliott Wave rules
            prices = pattern['price'].values

            if direction == 'up':
                # Bullish impulse validation
                wave1 = prices[1] - prices[0]  # Wave 1: low to high
                wave2 = prices[2] - prices[1]  # Wave 2: high to low (retracement)
                wave3 = prices[3] - prices[2]  # Wave 3: low to high
                wave4 = prices[4] - prices[3]  # Wave 4: high to low (retracement)
                wave5 = prices[5] - prices[4]  # Wave 5: low to high

                # Rule 1: Wave 2 doesn't retrace more than 100% of wave 1
                if prices[2] <= prices[0]:
                    continue

                # Rule 2: Wave 3 is never the shortest
                if wave3 < wave1 and wave3 < wave5:
                    continue

                # Rule 3: Wave 4 doesn't overlap wave 1
                if prices[4] <= prices[1]:
                    continue

                # Valid bullish impulse
                waves.append({
                    'direction': 'bullish',
                    'start_idx': pattern.index[0],
                    'end_idx': pattern.index[5],
                    'wave_1': (pattern.index[0], pattern.index[1]),
                    'wave_2': (pattern.index[1], pattern.index[2]),
                    'wave_3': (pattern.index[2], pattern.index[3]),
                    'wave_4': (pattern.index[3], pattern.index[4]),
                    'wave_5': (pattern.index[4], pattern.index[5]),
                    'prices': prices,
                    'wave_lengths': [wave1, wave2, wave3, wave4, wave5]
                })

            else:
                # Bearish impulse validation
                wave1 = prices[0] - prices[1]  # Wave 1: high to low
                wave2 = prices[1] - prices[2]  # Wave 2: low to high (retracement)
                wave3 = prices[2] - prices[3]  # Wave 3: high to low
                wave4 = prices[3] - prices[4]  # Wave 4: low to high (retracement)
                wave5 = prices[4] - prices[5]  # Wave 5: high to low

                # Rule 1: Wave 2 doesn't retrace more than 100% of wave 1
                if prices[2] >= prices[0]:
                    continue

                # Rule 2: Wave 3 is never the shortest
                if wave3 < wave1 and wave3 < wave5:
                    continue

                # Rule 3: Wave 4 doesn't overlap wave 1
                if prices[4] >= prices[1]:
                    continue

                # Valid bearish impulse
                waves.append({
                    'direction': 'bearish',
                    'start_idx': pattern.index[0],
                    'end_idx': pattern.index[5],
                    'wave_1': (pattern.index[0], pattern.index[1]),
                    'wave_2': (pattern.index[1], pattern.index[2]),
                    'wave_3': (pattern.index[2], pattern.index[3]),
                    'wave_4': (pattern.index[3], pattern.index[4]),
                    'wave_5': (pattern.index[4], pattern.index[5]),
                    'prices': prices,
                    'wave_lengths': [wave1, wave2, wave3, wave4, wave5]
                })

    return waves

# Analyze each timeframe
print("\n3. Analyzing wave structures across timeframes...")

# Daily timeframe (higher degree)
print("\n   DAILY TIMEFRAME (Higher Degree Waves):")
df_daily_waves = find_swing_points(df_daily, order=5)
daily_waves, daily_pivots = identify_elliott_waves(df_daily_waves)
print(f"   Found {len(daily_waves)} potential Elliott Wave patterns")

if len(daily_waves) > 0:
    # Show most recent wave
    recent_wave = daily_waves[-1]
    print(f"   Most recent: {recent_wave['direction']} impulse")
    print(f"   Dates: {recent_wave['start_idx'].strftime('%Y-%m-%d')} to {recent_wave['end_idx'].strftime('%Y-%m-%d')}")

# 15-minute timeframe (intermediate degree)
print("\n   15-MINUTE TIMEFRAME (Intermediate Waves):")
df_15min_waves = find_swing_points(df_15min, order=3)
waves_15min, pivots_15min = identify_elliott_waves(df_15min_waves)
print(f"   Found {len(waves_15min)} potential Elliott Wave patterns")

if len(waves_15min) > 0:
    recent_wave = waves_15min[-1]
    print(f"   Most recent: {recent_wave['direction']} impulse")

# ==========================================
# DETERMINE CURRENT WAVE POSITION
# ==========================================
print("\n4. Determining current market position in wave structure...")

def get_current_wave_position(df, waves):
    """Determine which wave we're currently in"""
    if len(waves) == 0:
        return None

    # Get most recent complete wave pattern
    latest_wave = waves[-1]
    current_price = df['close'].iloc[-1]

    # Check if we're past the last wave (potential new pattern starting)
    last_wave_price = latest_wave['prices'][-1]

    if latest_wave['direction'] == 'bullish':
        if current_price > last_wave_price:
            position = "Post Wave 5 (potential correction starting)"
        else:
            position = "Within completed pattern"
    else:
        if current_price < last_wave_price:
            position = "Post Wave 5 (potential correction starting)"
        else:
            position = "Within completed pattern"

    return {
        'pattern': latest_wave,
        'position': position,
        'current_price': current_price
    }

# Check daily position
daily_position = get_current_wave_position(df_daily_waves, daily_waves)
if daily_position:
    print(f"\n   DAILY: {daily_position['position']}")
    print(f"   Direction: {daily_position['pattern']['direction']}")

# Check 15-min position
position_15min = get_current_wave_position(df_15min_waves, waves_15min)
if position_15min:
    print(f"\n   15-MIN: {position_15min['position']}")
    print(f"   Direction: {position_15min['pattern']['direction']}")

# ==========================================
# TRADING STRATEGY BASED ON WAVE STRUCTURE
# ==========================================
print("\n5. Implementing Elliott Wave trading strategy...")

# Trading rules based on Elliott Wave:
# - Enter long at Wave 2 retracement (38.2%-61.8% Fibonacci)
# - Enter long at Wave 4 retracement
# - Avoid Wave 5 (exhaustion)
# - Enter correction trades after Wave 5

# Use 15-minute for signal generation, 1-minute for execution
df = df_1min.copy()

# Calculate indicators
df['sma_20'] = df['close'].rolling(window=20).mean()
df['sma_50'] = df['close'].rolling(window=50).mean()

# RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df['close'])

# Determine overall wave bias from higher timeframe
if position_15min and position_15min['pattern']['direction'] == 'bullish':
    wave_bias = 'bullish'
elif position_15min and position_15min['pattern']['direction'] == 'bearish':
    wave_bias = 'bearish'
else:
    wave_bias = 'neutral'

print(f"   Wave bias from 15-min analysis: {wave_bias.upper()}")

# Generate signals based on wave structure
df['long_signal'] = 0
df['short_signal'] = 0

TRANSACTION_COST = 0.000
STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.04
HOLD_MIN = 5
HOLD_MAX = 190

for i in range(50, len(df)):
    current_idx = df.index[i]

    # Long signals (trade in direction of wave bias)
    if wave_bias == 'bullish':
        # Enter on pullbacks in bullish wave structure
        if (df.loc[current_idx, 'close'] > df.loc[current_idx, 'sma_20'] and
            df.loc[current_idx, 'close'] < df.loc[current_idx, 'sma_50'] and
            30 < df.loc[current_idx, 'rsi'] < 50):
            df.loc[current_idx, 'long_signal'] = 1

    # Short signals
    elif wave_bias == 'bearish':
        # Enter on rallies in bearish wave structure
        if (df.loc[current_idx, 'close'] < df.loc[current_idx, 'sma_20'] and
            df.loc[current_idx, 'close'] > df.loc[current_idx, 'sma_50'] and
            50 < df.loc[current_idx, 'rsi'] < 70):
            df.loc[current_idx, 'short_signal'] = 1

print(f"   Long signals: {df['long_signal'].sum()}")
print(f"   Short signals: {df['short_signal'].sum()}")

# Execute trades
trades = []
position = None

for i in range(50, len(df)):
    current_idx = df.index[i]
    current_price = df.loc[current_idx, 'close']
    current_high = df.loc[current_idx, 'high']
    current_low = df.loc[current_idx, 'low']

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
            elif bars_held >= HOLD_MIN and current_price < df.loc[current_idx, 'sma_20']:
                exit_reason = 'trend_break'
        else:
            if current_high >= position['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = position['stop_loss']
            elif current_low <= position['take_profit']:
                exit_reason = 'take_profit'
                exit_price = position['take_profit']
            elif bars_held >= HOLD_MAX:
                exit_reason = 'max_time'
            elif bars_held >= HOLD_MIN and current_price > df.loc[current_idx, 'sma_20']:
                exit_reason = 'trend_break'

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
                'position_type': position_type,
                'wave_bias': wave_bias
            })

            position = None

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
    print(f"\n{'='*70}")
    print("ELLIOTT WAVE STRATEGY RESULTS")
    print('='*70)

    print(f"\nWave Bias: {wave_bias.upper()}")
    print(f"Total positions: {len(trades_df)}")

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

    if len(short_trades) > 0:
        print(f"\nSHORT:")
        print(f"  Wins: {(short_trades['net_pnl'] > 0).sum()} ({(short_trades['net_pnl'] > 0).sum()/len(short_trades)*100:.1f}%)")
        print(f"  Total P&L: {short_trades['net_pnl'].sum()*100:.4f}%")

    total_pnl = trades_df['net_pnl'].sum()
    print(f"\nTOTAL P&L: {total_pnl*100:.4f}%")
    print(f"Avg P&L: {trades_df['net_pnl'].mean()*100:.4f}%")
    print(f"Best: {trades_df['net_pnl'].max()*100:.4f}%")
    print(f"Worst: {trades_df['net_pnl'].min()*100:.4f}%")

    print(f"\nExit Reasons:")
    for reason, count in trades_df['exit_reason'].value_counts().items():
        print(f"  {reason}: {count} ({count/len(trades_df)*100:.1f}%)")

    if losing > 0:
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()
        avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean())
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
    trades_df.to_csv('elliott_wave_trades.csv', index=False)
    print(f"\nSaved: elliott_wave_trades.csv")

else:
    print("\nNo trades executed with current Elliott Wave bias.")

print("\n" + "="*70)
print("ELLIOTT WAVE ANALYSIS COMPLETE")
print("="*70)
