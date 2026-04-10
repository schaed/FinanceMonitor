import alpaca_trade_api as alpaca
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup API
YOUR_API_SECRET_KEY = os.getenv('ALPACA_PAPER_KEY')
ALPACA_ID = os.getenv('ALPACA_ID')
api = alpaca.REST(ALPACA_ID, YOUR_API_SECRET_KEY, 'https://paper-api.alpaca.markets/v2')

# Fetch SPY data
print("Fetching SPY data...")
df = api.get_bars(symbol="SPY", start="2026-03-25", end="2026-04-05", timeframe="1Min").df
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
print("\nCalculating technical indicators...")

# Basic features
df['returns'] = df['close'].pct_change()
df['price_change'] = df['close'].diff()
df['upward'] = (df['price_change'] > 0).astype(int)
df['downward'] = (df['price_change'] < 0).astype(int)

# Moving averages
for period in [5, 10, 15, 20, 30, 50]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

# Momentum indicators
df['momentum_3'] = df['close'] - df['close'].shift(3)
df['momentum_5'] = df['close'] - df['close'].shift(5)
df['momentum_10'] = df['close'] - df['close'].shift(10)
df['momentum_20'] = df['close'] - df['close'].shift(20)

# Acceleration
df['acceleration'] = df['momentum_5'].diff()

# Volatility (ATR)
df['high_low'] = df['high'] - df['low']
df['high_close'] = abs(df['high'] - df['close'].shift())
df['low_close'] = abs(df['low'] - df['close'].shift())
df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr_10'] = df['tr'].rolling(window=10).mean()
df['atr_20'] = df['tr'].rolling(window=20).mean()

# Volume
df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma_20']

# Price channels
df['upper_channel_20'] = df['close'].rolling(window=20).max()
df['lower_channel_20'] = df['close'].rolling(window=20).min()
df['channel_position'] = (df['close'] - df['lower_channel_20']) / (df['upper_channel_20'] - df['lower_channel_20'])

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
df['macd_hist'] = df['macd'] - df['macd_signal']

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

# BULLISH Trend Alignment Score (6 indicators)
df['bullish_alignment'] = (
    (df['close'] > df['sma_10']).astype(int) +
    (df['close'] > df['sma_20']).astype(int) +
    (df['sma_10'] > df['sma_20']).astype(int) +
    (df['ema_5'] > df['ema_10']).astype(int) +
    (df['momentum_5'] > 0).astype(int) +
    (df['macd'] > df['macd_signal']).astype(int)
)

# BEARISH Trend Alignment Score (6 indicators)
df['bearish_alignment'] = (
    (df['close'] < df['sma_10']).astype(int) +
    (df['close'] < df['sma_20']).astype(int) +
    (df['sma_10'] < df['sma_20']).astype(int) +
    (df['ema_5'] < df['ema_10']).astype(int) +
    (df['momentum_5'] < 0).astype(int) +
    (df['macd'] < df['macd_signal']).astype(int)
)

# ==========================================
# BIDIRECTIONAL STRATEGY
# ==========================================
print("\n" + "="*60)
print("BIDIRECTIONAL MEDIUM FREQUENCY TRADING STRATEGY")
print("="*60)

# Strategy parameters
HOLD_BARS_MIN = 2
HOLD_BARS_MAX = 8
STOP_LOSS_PCT = 0.0012  # 0.12%
TRAILING_STOP_PCT = 0.0008  # 0.08%
TAKE_PROFIT_1_PCT = 0.0015  # 0.15%
TAKE_PROFIT_2_PCT = 0.0025  # 0.25%
TRANSACTION_COST = 0.0001

# Generate signals
df['long_signal'] = 0
df['short_signal'] = 0
start_idx = 55

for i in range(start_idx, len(df)):
    current_idx = df.index[i]

    # LONG ENTRY CONDITIONS
    long_c1 = df.loc[current_idx, 'consecutive_up'] >= 2
    long_c2 = df.loc[current_idx, 'bullish_alignment'] >= 5
    long_c3 = 45 < df.loc[current_idx, 'rsi_14'] < 65
    long_c4 = df.loc[current_idx, 'momentum_5'] > 0 and df.loc[current_idx, 'momentum_10'] > 0
    long_c5 = df.loc[current_idx, 'acceleration'] > 0
    long_c6 = df.loc[current_idx, 'volume_ratio'] > 1.0
    long_c7 = df.loc[current_idx, 'channel_position'] < 0.85  # Not near resistance
    long_c8 = df.loc[current_idx, 'macd_hist'] > 0

    if all([long_c1, long_c2, long_c3, long_c4, long_c5, long_c6, long_c7, long_c8]):
        df.loc[current_idx, 'long_signal'] = 1

    # SHORT ENTRY CONDITIONS
    short_c1 = df.loc[current_idx, 'consecutive_down'] >= 2
    short_c2 = df.loc[current_idx, 'bearish_alignment'] >= 5
    short_c3 = 35 < df.loc[current_idx, 'rsi_14'] < 55
    short_c4 = df.loc[current_idx, 'momentum_5'] < 0 and df.loc[current_idx, 'momentum_10'] < 0
    short_c5 = df.loc[current_idx, 'acceleration'] < 0
    short_c6 = df.loc[current_idx, 'volume_ratio'] > 1.0
    short_c7 = df.loc[current_idx, 'channel_position'] > 0.15  # Not near support
    short_c8 = df.loc[current_idx, 'macd_hist'] < 0

    if all([short_c1, short_c2, short_c3, short_c4, short_c5, short_c6, short_c7, short_c8]):
        df.loc[current_idx, 'short_signal'] = 1

print(f"Long signals: {df['long_signal'].sum()}")
print(f"Short signals: {df['short_signal'].sum()}")
print(f"Total signals: {df['long_signal'].sum() + df['short_signal'].sum()}")

# Execute trades
trades = []
position = None

for i in range(start_idx, len(df)):
    current_idx = df.index[i]
    current_price = df.loc[current_idx, 'close']
    current_high = df.loc[current_idx, 'high']
    current_low = df.loc[current_idx, 'low']

    # Manage open position
    if position is not None:
        bars_held = i - position['entry_idx']
        position_type = position['type']  # 'long' or 'short'

        # Calculate P&L based on position type
        if position_type == 'long':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            # Update trailing stop for long
            if current_high > position['highest_price']:
                position['highest_price'] = current_high
                position['trailing_stop'] = position['highest_price'] * (1 - TRAILING_STOP_PCT)
        else:  # short
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            # Update trailing stop for short
            if current_low < position['lowest_price']:
                position['lowest_price'] = current_low
                position['trailing_stop'] = position['lowest_price'] * (1 + TRAILING_STOP_PCT)

        # Exit conditions
        exit_reason = None
        exit_price = current_price
        position_size = position.get('remaining_size', 1.0)

        if position_type == 'long':
            # Long exit conditions
            if current_low <= position['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = position['stop_loss']
            elif pnl_pct > 0 and current_low <= position['trailing_stop']:
                exit_reason = 'trailing_stop'
                exit_price = position['trailing_stop']
            elif current_high >= position['take_profit_1'] and not position.get('scaled_out', False):
                exit_reason = 'partial_profit_1'
                exit_price = position['take_profit_1']
                position['scaled_out'] = True
                position['remaining_size'] = 0.5

                gross_pnl = (exit_price - position['entry_price']) / position['entry_price'] * 0.5
                net_pnl = gross_pnl - TRANSACTION_COST

                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': current_idx,
                    'exit_price': exit_price,
                    'bars_held': bars_held,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'position_size': 0.5,
                    'position_type': 'long'
                })

                position['stop_loss'] = position['entry_price']  # Breakeven
                exit_reason = None
            elif current_high >= position['take_profit_2']:
                exit_reason = 'take_profit_2'
                exit_price = position['take_profit_2']
            elif bars_held >= HOLD_BARS_MAX:
                exit_reason = 'max_time'
            elif bars_held >= HOLD_BARS_MIN and df.loc[current_idx, 'downward'] == 1:
                if df.loc[current_idx, 'momentum_3'] < 0:
                    exit_reason = 'trend_break'

        else:  # short
            # Short exit conditions
            if current_high >= position['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = position['stop_loss']
            elif pnl_pct > 0 and current_high >= position['trailing_stop']:
                exit_reason = 'trailing_stop'
                exit_price = position['trailing_stop']
            elif current_low <= position['take_profit_1'] and not position.get('scaled_out', False):
                exit_reason = 'partial_profit_1'
                exit_price = position['take_profit_1']
                position['scaled_out'] = True
                position['remaining_size'] = 0.5

                gross_pnl = (position['entry_price'] - exit_price) / position['entry_price'] * 0.5
                net_pnl = gross_pnl - TRANSACTION_COST

                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': current_idx,
                    'exit_price': exit_price,
                    'bars_held': bars_held,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'position_size': 0.5,
                    'position_type': 'short'
                })

                position['stop_loss'] = position['entry_price']  # Breakeven
                exit_reason = None
            elif current_low <= position['take_profit_2']:
                exit_reason = 'take_profit_2'
                exit_price = position['take_profit_2']
            elif bars_held >= HOLD_BARS_MAX:
                exit_reason = 'max_time'
            elif bars_held >= HOLD_BARS_MIN and df.loc[current_idx, 'upward'] == 1:
                if df.loc[current_idx, 'momentum_3'] > 0:
                    exit_reason = 'trend_break'

        # Close position
        if exit_reason and exit_reason != 'partial_profit_1':
            if position_type == 'long':
                gross_pnl = (exit_price - position['entry_price']) / position['entry_price'] * position_size
            else:
                gross_pnl = (position['entry_price'] - exit_price) / position['entry_price'] * position_size

            net_pnl = gross_pnl - TRANSACTION_COST

            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': current_idx,
                'exit_price': exit_price,
                'bars_held': bars_held,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason,
                'position_size': position_size,
                'position_type': position_type
            })

            position = None

    # Enter new position (no position overlap)
    if position is None:
        # Check for long entry
        if df.loc[current_idx, 'long_signal'] == 1:
            position = {
                'entry_idx': i,
                'entry_time': current_idx,
                'entry_price': current_price,
                'type': 'long',
                'stop_loss': current_price * (1 - STOP_LOSS_PCT),
                'take_profit_1': current_price * (1 + TAKE_PROFIT_1_PCT),
                'take_profit_2': current_price * (1 + TAKE_PROFIT_2_PCT),
                'highest_price': current_price,
                'trailing_stop': current_price * (1 - TRAILING_STOP_PCT),
                'scaled_out': False,
                'remaining_size': 1.0
            }
        # Check for short entry
        elif df.loc[current_idx, 'short_signal'] == 1:
            position = {
                'entry_idx': i,
                'entry_time': current_idx,
                'entry_price': current_price,
                'type': 'short',
                'stop_loss': current_price * (1 + STOP_LOSS_PCT),
                'take_profit_1': current_price * (1 - TAKE_PROFIT_1_PCT),
                'take_profit_2': current_price * (1 - TAKE_PROFIT_2_PCT),
                'lowest_price': current_price,
                'trailing_stop': current_price * (1 + TRAILING_STOP_PCT),
                'scaled_out': False,
                'remaining_size': 1.0
            }

# Analyze results
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    print(f"\n{'='*60}")
    print("BIDIRECTIONAL STRATEGY PERFORMANCE")
    print('='*60)

    # Aggregate by entry time for position-level stats
    position_stats = trades_df.groupby(['entry_time', 'position_type']).agg({
        'net_pnl': 'sum',
        'bars_held': 'max',
        'exit_reason': lambda x: ', '.join(x.unique())
    }).reset_index()

    print(f"\nTotal positions: {len(position_stats)}")
    print(f"  Long positions: {(position_stats['position_type'] == 'long').sum()}")
    print(f"  Short positions: {(position_stats['position_type'] == 'short').sum()}")

    # Overall stats
    winning_positions = (position_stats['net_pnl'] > 0).sum()
    losing_positions = (position_stats['net_pnl'] < 0).sum()

    print(f"\nWinning positions: {winning_positions} ({winning_positions/len(position_stats)*100:.1f}%)")
    print(f"Losing positions: {losing_positions} ({losing_positions/len(position_stats)*100:.1f}%)")

    # Long vs Short breakdown
    long_stats = position_stats[position_stats['position_type'] == 'long']
    short_stats = position_stats[position_stats['position_type'] == 'short']

    if len(long_stats) > 0:
        print(f"\nLONG POSITIONS:")
        print(f"  Wins: {(long_stats['net_pnl'] > 0).sum()} ({(long_stats['net_pnl'] > 0).sum()/len(long_stats)*100:.1f}%)")
        print(f"  Avg P&L: {long_stats['net_pnl'].mean()*100:.4f}%")
        print(f"  Total P&L: {long_stats['net_pnl'].sum()*100:.4f}%")

    if len(short_stats) > 0:
        print(f"\nSHORT POSITIONS:")
        print(f"  Wins: {(short_stats['net_pnl'] > 0).sum()} ({(short_stats['net_pnl'] > 0).sum()/len(short_stats)*100:.1f}%)")
        print(f"  Avg P&L: {short_stats['net_pnl'].mean()*100:.4f}%")
        print(f"  Total P&L: {short_stats['net_pnl'].sum()*100:.4f}%")

    print(f"\nOVERALL P&L Statistics:")
    print(f"  Average net P&L: {position_stats['net_pnl'].mean()*100:.4f}%")
    print(f"  Median net P&L: {position_stats['net_pnl'].median()*100:.4f}%")
    print(f"  Best position: {position_stats['net_pnl'].max()*100:.4f}%")
    print(f"  Worst position: {position_stats['net_pnl'].min()*100:.4f}%")

    total_pnl = position_stats['net_pnl'].sum()
    print(f"\nTotal Cumulative P&L: {total_pnl*100:.4f}%")

    # Performance metrics
    avg_win = position_stats[position_stats['net_pnl'] > 0]['net_pnl'].mean()
    avg_loss = abs(position_stats[position_stats['net_pnl'] < 0]['net_pnl'].mean()) if losing_positions > 0 else 0
    win_rate = winning_positions / len(position_stats)

    if avg_loss > 0 and losing_positions > 0:
        profit_factor = (avg_win * winning_positions) / (avg_loss * losing_positions)
        print(f"\nProfit Factor: {profit_factor:.2f}")
        print(f"Win/Loss Ratio: {avg_win/avg_loss:.2f}")

    print(f"Average Win: {avg_win*100:.4f}%")
    if avg_loss > 0:
        print(f"Average Loss: {avg_loss*100:.4f}%")

    # Sharpe
    if position_stats['net_pnl'].std() > 0:
        days_traded = (df.index[-1] - df.index[0]).days
        positions_per_day = len(position_stats) / days_traded
        annual_return = position_stats['net_pnl'].mean() * positions_per_day * 252
        annual_vol = position_stats['net_pnl'].std() * np.sqrt(positions_per_day * 252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        print(f"\nAnnualized Sharpe Ratio: {sharpe:.2f}")

    # Drawdown
    position_stats['cumulative_pnl'] = position_stats['net_pnl'].cumsum()
    cumulative_returns = (1 + position_stats['net_pnl']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown*100:.4f}%")

    # Exit reasons
    print("\nExit Reasons (all exits):")
    exit_counts = trades_df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} ({count/len(trades_df)*100:.1f}%)")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            'SPY Price with Long/Short Entries & Exits',
            'Bullish vs Bearish Alignment Scores',
            'RSI Indicator',
            'Per-Position P&L (Long vs Short)',
            'Cumulative P&L',
            'Drawdown'
        ),
        row_heights=[0.30, 0.12, 0.12, 0.15, 0.16, 0.15]
    )

    # Row 1: Price with entries/exits
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='SPY'
        ),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], line=dict(color='orange', width=1), name='SMA 20'), row=1, col=1)

    # Long entries
    long_entries = trades_df[trades_df['position_type'] == 'long'].drop_duplicates('entry_time')
    if len(long_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=long_entries['entry_time'],
                y=long_entries['entry_price'],
                mode='markers',
                marker=dict(color='lime', size=12, symbol='triangle-up'),
                name='Long Entry'
            ),
            row=1, col=1
        )

    # Short entries
    short_entries = trades_df[trades_df['position_type'] == 'short'].drop_duplicates('entry_time')
    if len(short_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=short_entries['entry_time'],
                y=short_entries['entry_price'],
                mode='markers',
                marker=dict(color='red', size=12, symbol='triangle-down'),
                name='Short Entry'
            ),
            row=1, col=1
        )

    # All exits
    long_exits = trades_df[trades_df['position_type'] == 'long']
    if len(long_exits) > 0:
        colors = ['green' if pnl > 0 else 'darkred' for pnl in long_exits['net_pnl']]
        fig.add_trace(
            go.Scatter(
                x=long_exits['exit_time'],
                y=long_exits['exit_price'],
                mode='markers',
                marker=dict(color=colors, size=8, symbol='x'),
                name='Long Exit'
            ),
            row=1, col=1
        )

    short_exits = trades_df[trades_df['position_type'] == 'short']
    if len(short_exits) > 0:
        colors = ['green' if pnl > 0 else 'darkred' for pnl in short_exits['net_pnl']]
        fig.add_trace(
            go.Scatter(
                x=short_exits['exit_time'],
                y=short_exits['exit_price'],
                mode='markers',
                marker=dict(color=colors, size=8, symbol='x'),
                name='Short Exit'
            ),
            row=1, col=1
        )

    # Row 2: Alignment scores
    fig.add_trace(go.Scatter(x=df.index, y=df['bullish_alignment'], line=dict(color='green', width=1), name='Bullish'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bearish_alignment'], line=dict(color='red', width=1), name='Bearish'), row=2, col=1)
    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # Row 3: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], line=dict(color='purple'), name='RSI'), row=3, col=1)
    fig.add_hline(y=65, line_dash="dash", line_color="red", opacity=0.3, row=3, col=1)
    fig.add_hline(y=45, line_dash="dash", line_color="green", opacity=0.3, row=3, col=1)
    fig.add_hline(y=55, line_dash="dash", line_color="orange", opacity=0.3, row=3, col=1)
    fig.add_hline(y=35, line_dash="dash", line_color="orange", opacity=0.3, row=3, col=1)

    # Row 4: P&L per position
    if len(long_stats) > 0:
        fig.add_trace(
            go.Bar(
                x=list(range(len(long_stats))),
                y=long_stats['net_pnl']*100,
                marker_color=['green' if x > 0 else 'red' for x in long_stats['net_pnl']],
                name='Long P&L',
                opacity=0.7
            ),
            row=4, col=1
        )

    if len(short_stats) > 0:
        fig.add_trace(
            go.Bar(
                x=list(range(len(long_stats), len(long_stats) + len(short_stats))),
                y=short_stats['net_pnl']*100,
                marker_color=['green' if x > 0 else 'darkred' for x in short_stats['net_pnl']],
                name='Short P&L',
                opacity=0.7
            ),
            row=4, col=1
        )

    # Row 5: Cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=list(range(len(position_stats))),
            y=position_stats['cumulative_pnl']*100,
            line=dict(color='blue', width=2),
            fill='tozeroy',
            name='Cumulative P&L'
        ),
        row=5, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=5, col=1)

    # Row 6: Drawdown
    fig.add_trace(
        go.Scatter(
            x=list(range(len(position_stats))),
            y=drawdown*100,
            line=dict(color='red'),
            fill='tozeroy',
            name='Drawdown'
        ),
        row=6, col=1
    )

    # Layout
    fig.update_layout(
        title=f'Bidirectional Strategy | P&L: {total_pnl*100:.2f}% | Win: {win_rate*100:.1f}% | Sharpe: {sharpe:.2f} | Long: {len(long_stats)} | Short: {len(short_stats)}',
        height=1600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="P&L %", row=4, col=1)
    fig.update_yaxes(title_text="Cum %", row=5, col=1)
    fig.update_yaxes(title_text="DD %", row=6, col=1)

    fig.write_html('bidirectional_strategy.html')
    fig.show()

    position_stats.to_csv('bidirectional_trades.csv', index=False)

    print("\nFiles saved:")
    print("  - bidirectional_strategy.html")
    print("  - bidirectional_trades.csv")

    # Comparison
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    print("\nLong-Only (Advanced):")
    print("   P&L: -0.19% | Positions: 26")
    print("\nBidirectional:")
    print(f"   P&L: {total_pnl*100:.2f}% | Positions: {len(position_stats)}")
    print(f"   Long: {len(long_stats)} positions, {long_stats['net_pnl'].sum()*100:.2f}% P&L")
    print(f"   Short: {len(short_stats)} positions, {short_stats['net_pnl'].sum()*100:.2f}% P&L")

else:
    print("\nNo trades executed.")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
