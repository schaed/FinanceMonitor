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
# ADVANCED FEATURE ENGINEERING
# ==========================================
print("\nCalculating advanced indicators...")

# Basic features
df['returns'] = df['close'].pct_change()
df['price_change'] = df['close'].diff()
df['upward'] = (df['price_change'] > 0).astype(int)
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

# Multiple timeframe moving averages
for period in [5, 10, 15, 20, 30, 50]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

# Price momentum strength
df['momentum_3'] = df['close'] - df['close'].shift(3)
df['momentum_5'] = df['close'] - df['close'].shift(5)
df['momentum_10'] = df['close'] - df['close'].shift(10)
df['momentum_20'] = df['close'] - df['close'].shift(20)

# Acceleration (momentum of momentum)
df['acceleration'] = df['momentum_5'].diff()

# Volatility measures
df['high_low'] = df['high'] - df['low']
df['high_close'] = abs(df['high'] - df['close'].shift())
df['low_close'] = abs(df['low'] - df['close'].shift())
df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr_10'] = df['tr'].rolling(window=10).mean()
df['atr_20'] = df['tr'].rolling(window=20).mean()

# Normalized volatility
df['volatility_ratio'] = df['atr_10'] / df['atr_20']

# Volume analysis
df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma_20']

# Price channels (support/resistance)
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

# Consecutive bars analysis
df['consecutive_up'] = 0
df['consecutive_down'] = 0
up_count = 0
down_count = 0

for i in range(len(df)):
    if df['upward'].iloc[i] == 1:
        up_count += 1
        down_count = 0
        df.iloc[i, df.columns.get_loc('consecutive_up')] = up_count
    else:
        down_count += 1
        up_count = 0
        df.iloc[i, df.columns.get_loc('consecutive_down')] = down_count

# Trend strength score (composite indicator)
df['trend_alignment'] = (
    (df['close'] > df['sma_10']).astype(int) +
    (df['close'] > df['sma_20']).astype(int) +
    (df['sma_10'] > df['sma_20']).astype(int) +
    (df['ema_5'] > df['ema_10']).astype(int) +
    (df['momentum_5'] > 0).astype(int) +
    (df['macd'] > df['macd_signal']).astype(int)
)

# ==========================================
# ADVANCED STRATEGY WITH BETTER FILTERS
# ==========================================
print("\n" + "="*60)
print("ADVANCED MEDIUM FREQUENCY TREND FOLLOWING STRATEGY")
print("="*60)

# Strategy: More selective entries with stronger trend confirmation
# Entry conditions:
# 1. Strong trend alignment (score >= 5 out of 6)
# 2. 2+ consecutive upward bars
# 3. RSI between 45-65 (momentum but not overextended)
# 4. Positive momentum on multiple timeframes
# 5. Volume confirmation
# 6. Not near resistance (channel position < 0.85)

# More aggressive profit taking and tighter stops
HOLD_BARS_MIN = 2
HOLD_BARS_MAX = 8
STOP_LOSS_PCT = 0.0012  # 0.12% stop loss
TRAILING_STOP_PCT = 0.0008  # 0.08% trailing stop (tighter)
TAKE_PROFIT_1_PCT = 0.0015  # First target: 0.15%
TAKE_PROFIT_2_PCT = 0.0025  # Second target: 0.25%
TRANSACTION_COST = 0.0001

# Generate signals
df['signal'] = 0
start_idx = 55  # Wait for all indicators

for i in range(start_idx, len(df)):
    current_idx = df.index[i]

    # Entry conditions (more selective)
    c1 = df.loc[current_idx, 'consecutive_up'] >= 2
    c2 = df.loc[current_idx, 'trend_alignment'] >= 5  # Strong alignment
    c3 = 45 < df.loc[current_idx, 'rsi_14'] < 65  # Moderate RSI
    c4 = df.loc[current_idx, 'momentum_5'] > 0 and df.loc[current_idx, 'momentum_10'] > 0
    c5 = df.loc[current_idx, 'acceleration'] > 0  # Accelerating momentum
    c6 = df.loc[current_idx, 'volume_ratio'] > 1.0  # Above average volume
    c7 = df.loc[current_idx, 'channel_position'] < 0.85  # Not near resistance
    c8 = df.loc[current_idx, 'macd_hist'] > 0  # MACD histogram positive

    if all([c1, c2, c3, c4, c5, c6, c7, c8]):
        df.loc[current_idx, 'signal'] = 1

print(f"Total entry signals: {df['signal'].sum()}")

# Execute trades with scale-out strategy
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
        pnl_pct = (current_price - position['entry_price']) / position['entry_price']

        # Update trailing stop
        if current_high > position['highest_price']:
            position['highest_price'] = current_high
            position['trailing_stop'] = position['highest_price'] * (1 - TRAILING_STOP_PCT)

        # Exit conditions
        exit_reason = None
        exit_price = current_price
        position_size = position.get('remaining_size', 1.0)

        # Hard stop loss
        if current_low <= position['stop_loss']:
            exit_reason = 'stop_loss'
            exit_price = position['stop_loss']

        # Trailing stop (after profitable)
        elif pnl_pct > 0 and current_low <= position['trailing_stop']:
            exit_reason = 'trailing_stop'
            exit_price = position['trailing_stop']

        # Scale out at first target
        elif current_high >= position['take_profit_1'] and not position.get('scaled_out', False):
            # Take 50% profit at first target
            exit_reason = 'partial_profit_1'
            exit_price = position['take_profit_1']
            position['scaled_out'] = True
            position['remaining_size'] = 0.5

            # Record partial exit
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
                'position_size': 0.5
            })

            # Tighten stop for remaining position
            position['stop_loss'] = position['entry_price']  # Move to breakeven
            exit_reason = None  # Don't close full position

        # Full exit at second target
        elif current_high >= position['take_profit_2']:
            exit_reason = 'take_profit_2'
            exit_price = position['take_profit_2']

        # Time-based exit
        elif bars_held >= HOLD_BARS_MAX:
            exit_reason = 'max_time'

        # Trend break after minimum hold
        elif bars_held >= HOLD_BARS_MIN:
            if df.loc[current_idx, 'upward'] == 0:
                # Check if it's a real break
                if df.loc[current_idx, 'momentum_3'] < 0:
                    exit_reason = 'trend_break'

        # Close full or remaining position
        if exit_reason and exit_reason != 'partial_profit_1':
            gross_pnl = (exit_price - position['entry_price']) / position['entry_price'] * position_size
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
                'position_size': position_size
            })

            position = None

    # Enter new position
    if position is None and df.loc[current_idx, 'signal'] == 1:
        position = {
            'entry_idx': i,
            'entry_time': current_idx,
            'entry_price': current_price,
            'stop_loss': current_price * (1 - STOP_LOSS_PCT),
            'take_profit_1': current_price * (1 + TAKE_PROFIT_1_PCT),
            'take_profit_2': current_price * (1 + TAKE_PROFIT_2_PCT),
            'highest_price': current_price,
            'trailing_stop': current_price * (1 - TRAILING_STOP_PCT),
            'scaled_out': False,
            'remaining_size': 1.0
        }

# Analysis
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    print(f"\n{'='*60}")
    print("ADVANCED STRATEGY PERFORMANCE")
    print('='*60)

    print(f"\nTotal trades/exits: {len(trades_df)}")

    # Aggregate by entry time for position-level stats
    position_stats = trades_df.groupby('entry_time').agg({
        'net_pnl': 'sum',
        'bars_held': 'max',
        'exit_reason': lambda x: ', '.join(x.unique())
    }).reset_index()

    print(f"Total positions: {len(position_stats)}")
    winning_positions = (position_stats['net_pnl'] > 0).sum()
    losing_positions = (position_stats['net_pnl'] < 0).sum()

    print(f"Winning positions: {winning_positions} ({winning_positions/len(position_stats)*100:.1f}%)")
    print(f"Losing positions: {losing_positions} ({losing_positions/len(position_stats)*100:.1f}%)")

    print(f"\nP&L Statistics (per position):")
    print(f"  Average net P&L: {position_stats['net_pnl'].mean()*100:.4f}%")
    print(f"  Median net P&L: {position_stats['net_pnl'].median()*100:.4f}%")
    print(f"  Best position: {position_stats['net_pnl'].max()*100:.4f}%")
    print(f"  Worst position: {position_stats['net_pnl'].min()*100:.4f}%")

    print(f"\nHolding Period:")
    print(f"  Average: {position_stats['bars_held'].mean():.2f} bars")
    print(f"  Median: {position_stats['bars_held'].median():.0f} bars")

    print("\nExit Reasons (all exits):")
    exit_counts = trades_df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} ({count/len(trades_df)*100:.1f}%)")

    # Total P&L
    total_pnl = position_stats['net_pnl'].sum()
    cumulative_pnl = position_stats['net_pnl'].cumsum()
    position_stats['cumulative_pnl'] = cumulative_pnl

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

    # Sharpe ratio
    if position_stats['net_pnl'].std() > 0:
        days_traded = (df.index[-1] - df.index[0]).days
        positions_per_day = len(position_stats) / days_traded
        annual_return = position_stats['net_pnl'].mean() * positions_per_day * 252
        annual_vol = position_stats['net_pnl'].std() * np.sqrt(positions_per_day * 252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        print(f"\nAnnualized Sharpe Ratio: {sharpe:.2f}")

    # Maximum drawdown
    cumulative_returns = (1 + position_stats['net_pnl']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown*100:.4f}%")

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
            'SPY Price & Entries/Exits',
            'Trend Alignment Score',
            'RSI & MACD',
            'Per-Position P&L',
            'Cumulative P&L',
            'Drawdown'
        ),
        row_heights=[0.30, 0.10, 0.15, 0.15, 0.15, 0.15]
    )

    # Row 1: Price
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

    # Entry points
    entry_df = trades_df.drop_duplicates('entry_time')
    fig.add_trace(
        go.Scatter(
            x=entry_df['entry_time'],
            y=entry_df['entry_price'],
            mode='markers',
            marker=dict(color='lime', size=10, symbol='triangle-up'),
            name='Entry'
        ),
        row=1, col=1
    )

    # Exit points
    colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['net_pnl']]
    fig.add_trace(
        go.Scatter(
            x=trades_df['exit_time'],
            y=trades_df['exit_price'],
            mode='markers',
            marker=dict(color=colors, size=8, symbol='triangle-down'),
            name='Exit'
        ),
        row=1, col=1
    )

    # Row 2: Trend Alignment
    fig.add_trace(go.Scatter(x=df.index, y=df['trend_alignment'], line=dict(color='blue'), name='Trend Score'), row=2, col=1)
    fig.add_hline(y=5, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # Row 3: RSI and MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], line=dict(color='purple'), name='RSI'), row=3, col=1)
    fig.add_hline(y=65, line_dash="dash", line_color="red", opacity=0.3, row=3, col=1)
    fig.add_hline(y=45, line_dash="dash", line_color="green", opacity=0.3, row=3, col=1)

    # Row 4: P&L per position
    colors = ['green' if x > 0 else 'red' for x in position_stats['net_pnl']]
    fig.add_trace(
        go.Bar(x=list(range(len(position_stats))), y=position_stats['net_pnl']*100, marker_color=colors, name='P&L'),
        row=4, col=1
    )

    # Row 5: Cumulative
    fig.add_trace(
        go.Scatter(x=list(range(len(position_stats))), y=cumulative_pnl*100,
                  line=dict(color='blue', width=2), fill='tozeroy', name='Cum P&L'),
        row=5, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=5, col=1)

    # Row 6: Drawdown
    fig.add_trace(
        go.Scatter(x=list(range(len(position_stats))), y=drawdown*100,
                  line=dict(color='red'), fill='tozeroy', name='DD'),
        row=6, col=1
    )

    fig.update_layout(
        title=f'Advanced Strategy | Win: {win_rate*100:.1f}% | P&L: {total_pnl*100:.2f}% | Sharpe: {sharpe:.2f}',
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

    fig.write_html('advanced_trend_strategy.html')
    fig.show()

    position_stats.to_csv('advanced_trades_history.csv', index=False)
    print("\nFiles saved:")
    print("  - advanced_trend_strategy.html")
    print("  - advanced_trades_history.csv")

    # Final comparison
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    print("\n1. Simple (2 up bars only):")
    print("   P&L: -14.32% | Win: 18.4% | Sharpe: -39.12 | Trades: 784")
    print("\n2. Improved (multi-factor):")
    print("   P&L: -6.13% | Win: 24.2% | Sharpe: -19.53 | Trades: 244")
    print("\n3. Advanced (selective + scale-out):")
    print(f"   P&L: {total_pnl*100:.2f}% | Win: {win_rate*100:.1f}% | Sharpe: {sharpe:.2f} | Positions: {len(position_stats)}")

else:
    print("\nNo trades executed.")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
