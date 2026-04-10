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
df = api.get_bars(symbol="SPY", start="2026-02-25", end="2026-04-05", timeframe="1Min").df
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
for period in [5, 10, 15, 20, 30]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

# Momentum
df['momentum_3'] = df['close'] - df['close'].shift(3)
df['momentum_5'] = df['close'] - df['close'].shift(5)
df['momentum_10'] = df['close'] - df['close'].shift(10)
df['acceleration'] = df['momentum_5'].diff()

# Volatility
df['high_low'] = df['high'] - df['low']
df['high_close'] = abs(df['high'] - df['close'].shift())
df['low_close'] = abs(df['low'] - df['close'].shift())
df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr_10'] = df['tr'].rolling(window=10).mean()
df['atr_20'] = df['tr'].rolling(window=20).mean()

# Volume
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

# Consecutive bars tracking
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

# Trend alignment scores
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

# ==========================================
# SMA10 EXIT STRATEGY
# ==========================================
print("\n" + "="*60)
print("SMA10 TREND EXIT STRATEGY")
print("="*60)
print("\nExit Logic:")
print("  LONG: Hold until close BELOW SMA10 (trend violation)")
print("  SHORT: Hold until close ABOVE SMA10 (trend violation)")
print("  Optional: Take profit at target levels")

# Strategy parameters
TRANSACTION_COST = 0.000
TAKE_PROFIT_PCT = 0.02  # 0.5% take profit (wider target)
MAX_HOLD_BARS = 180  # Longer max hold since we follow trend

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
    long_c7 = df.loc[current_idx, 'channel_position'] < 0.85
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
    short_c7 = df.loc[current_idx, 'channel_position'] > 0.15
    short_c8 = df.loc[current_idx, 'macd_hist'] < 0

    if all([short_c1, short_c2, short_c3, short_c4, short_c5, short_c6, short_c7, short_c8]):
        df.loc[current_idx, 'short_signal'] = 1

print(f"Long signals: {df['long_signal'].sum()}")
print(f"Short signals: {df['short_signal'].sum()}")

# Execute trades
trades = []
position = None

for i in range(start_idx, len(df)):
    current_idx = df.index[i]
    current_price = df.loc[current_idx, 'close']
    current_high = df.loc[current_idx, 'high']
    current_low = df.loc[current_idx, 'low']
    current_sma10 = df.loc[current_idx, 'sma_10']

    # Manage open position
    if position is not None:
        bars_held = i - position['entry_idx']
        position_type = position['type']

        # Calculate P&L
        if position_type == 'long':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        # Exit conditions
        exit_reason = None
        exit_price = current_price

        if position_type == 'long':
            # LONG exits
            # 1. SMA10 violation: close BELOW SMA10
            if current_price < current_sma10:
                exit_reason = 'sma10_violation'
                exit_price = current_price
            # 2. Take profit (optional)
            elif current_high >= position['take_profit']:
                exit_reason = 'take_profit'
                exit_price = position['take_profit']
            # 3. Max time
            elif bars_held >= MAX_HOLD_BARS:
                exit_reason = 'max_time'

        else:  # short
            # SHORT exits
            # 1. SMA10 violation: close ABOVE SMA10
            if current_price > current_sma10:
                exit_reason = 'sma10_violation'
                exit_price = current_price
            # 2. Take profit (optional)
            elif current_low <= position['take_profit']:
                exit_reason = 'take_profit'
                exit_price = position['take_profit']
            # 3. Max time
            elif bars_held >= MAX_HOLD_BARS:
                exit_reason = 'max_time'

        # Close position
        if exit_reason:
            if position_type == 'long':
                gross_pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:
                gross_pnl = (position['entry_price'] - exit_price) / position['entry_price']

            net_pnl = gross_pnl - (2 * TRANSACTION_COST)

            # Track SMA10 at entry vs exit
            sma10_at_entry = position['sma10_at_entry']
            sma10_at_exit = current_sma10

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
                'sma10_at_entry': sma10_at_entry,
                'sma10_at_exit': sma10_at_exit
            })

            position = None

    # Enter new position
    if position is None:
        # Long entry
        if df.loc[current_idx, 'long_signal'] == 1:
            # Must be above SMA10 to enter
            if current_price > current_sma10:
                position = {
                    'entry_idx': i,
                    'entry_time': current_idx,
                    'entry_price': current_price,
                    'type': 'long',
                    'take_profit': current_price * (1 + TAKE_PROFIT_PCT),
                    'sma10_at_entry': current_sma10
                }

        # Short entry
        elif df.loc[current_idx, 'short_signal'] == 1:
            # Must be below SMA10 to enter
            if current_price < current_sma10:
                position = {
                    'entry_idx': i,
                    'entry_time': current_idx,
                    'entry_price': current_price,
                    'type': 'short',
                    'take_profit': current_price * (1 - TAKE_PROFIT_PCT),
                    'sma10_at_entry': current_sma10
                }

# Analyze results
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    print(f"\n{'='*60}")
    print("SMA10 EXIT STRATEGY PERFORMANCE")
    print('='*60)

    print(f"\nTotal positions: {len(trades_df)}")
    print(f"  Long: {(trades_df['position_type'] == 'long').sum()}")
    print(f"  Short: {(trades_df['position_type'] == 'short').sum()}")

    # Overall performance
    winning = (trades_df['net_pnl'] > 0).sum()
    losing = (trades_df['net_pnl'] < 0).sum()

    print(f"\nWinning: {winning} ({winning/len(trades_df)*100:.1f}%)")
    print(f"Losing: {losing} ({losing/len(trades_df)*100:.1f}%)")

    # Long vs Short
    long_trades = trades_df[trades_df['position_type'] == 'long']
    short_trades = trades_df[trades_df['position_type'] == 'short']

    if len(long_trades) > 0:
        print(f"\nLONG POSITIONS:")
        print(f"  Count: {len(long_trades)}")
        print(f"  Wins: {(long_trades['net_pnl'] > 0).sum()} ({(long_trades['net_pnl'] > 0).sum()/len(long_trades)*100:.1f}%)")
        print(f"  Avg P&L: {long_trades['net_pnl'].mean()*100:.4f}%")
        print(f"  Total P&L: {long_trades['net_pnl'].sum()*100:.4f}%")
        print(f"  Avg hold time: {long_trades['bars_held'].mean():.1f} bars")
        print(f"  Max hold time: {long_trades['bars_held'].max():.0f} bars")

    if len(short_trades) > 0:
        print(f"\nSHORT POSITIONS:")
        print(f"  Count: {len(short_trades)}")
        print(f"  Wins: {(short_trades['net_pnl'] > 0).sum()} ({(short_trades['net_pnl'] > 0).sum()/len(short_trades)*100:.1f}%)")
        print(f"  Avg P&L: {short_trades['net_pnl'].mean()*100:.4f}%")
        print(f"  Total P&L: {short_trades['net_pnl'].sum()*100:.4f}%")
        print(f"  Avg hold time: {short_trades['bars_held'].mean():.1f} bars")
        print(f"  Max hold time: {short_trades['bars_held'].max():.0f} bars")

    total_pnl = trades_df['net_pnl'].sum()
    print(f"\nTOTAL P&L: {total_pnl*100:.4f}%")
    print(f"Average P&L per trade: {trades_df['net_pnl'].mean()*100:.4f}%")
    print(f"Median P&L: {trades_df['net_pnl'].median()*100:.4f}%")
    print(f"Best trade: {trades_df['net_pnl'].max()*100:.4f}%")
    print(f"Worst trade: {trades_df['net_pnl'].min()*100:.4f}%")

    # Exit reasons
    print("\nExit Reasons:")
    exit_counts = trades_df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} ({count/len(trades_df)*100:.1f}%)")

    # Holding time stats
    print(f"\nHolding Time Statistics:")
    print(f"  Average: {trades_df['bars_held'].mean():.1f} bars")
    print(f"  Median: {trades_df['bars_held'].median():.0f} bars")
    print(f"  Min: {trades_df['bars_held'].min():.0f} bars")
    print(f"  Max: {trades_df['bars_held'].max():.0f} bars")

    # Risk metrics
    avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()
    avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean()) if losing > 0 else 0

    if losing > 0:
        profit_factor = (avg_win * winning) / (avg_loss * losing)
        print(f"\nProfit Factor: {profit_factor:.2f}")
        print(f"Win/Loss Ratio: {avg_win/avg_loss:.2f}")

    print(f"Average Win: {avg_win*100:.4f}%")
    if losing > 0:
        print(f"Average Loss: {avg_loss*100:.4f}%")

    # Sharpe
    if trades_df['net_pnl'].std() > 0:
        days_traded = (df.index[-1] - df.index[0]).days
        trades_per_day = len(trades_df) / days_traded
        annual_return = trades_df['net_pnl'].mean() * trades_per_day * 252
        annual_vol = trades_df['net_pnl'].std() * np.sqrt(trades_per_day * 252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        print(f"\nAnnualized Sharpe Ratio: {sharpe:.2f}")

    # Drawdown
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
    cumulative_returns = (1 + trades_df['net_pnl']).cumprod()
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
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            'SPY Price with SMA10 Exit Strategy',
            'Holding Time Distribution',
            'P&L per Trade',
            'Cumulative P&L',
            'Exit Reasons'
        ),
        row_heights=[0.40, 0.15, 0.15, 0.15, 0.15]
    )

    # Row 1: Price chart
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

    # Add SMA 10 (the exit trigger)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['sma_10'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='SMA 10 (Exit Line)'
        ),
        row=1, col=1
    )

    # Add SMA 20 for reference
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['sma_20'],
            mode='lines',
            line=dict(color='orange', width=1.5, dash='dot'),
            name='SMA 20'
        ),
        row=1, col=1
    )

    # Add entries and exits
    for _, trade in trades_df.iterrows():
        color = 'green' if trade['position_type'] == 'long' else 'red'
        symbol = 'triangle-up' if trade['position_type'] == 'long' else 'triangle-down'

        # Entry point
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_time']],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(color=color, size=12, symbol=symbol, line=dict(width=2, color='white')),
                showlegend=False,
                hovertext=f"{trade['position_type'].upper()} entry<br>Held: {trade['bars_held']} bars"
            ),
            row=1, col=1
        )

        # Exit point
        exit_color = 'lime' if trade['net_pnl'] > 0 else 'darkred'
        fig.add_trace(
            go.Scatter(
                x=[trade['exit_time']],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(color=exit_color, size=10, symbol='x', line=dict(width=2)),
                showlegend=False,
                hovertext=f"Exit: {trade['exit_reason']}<br>P&L: {trade['net_pnl']*100:.2f}%<br>Held: {trade['bars_held']} bars"
            ),
            row=1, col=1
        )

        # Draw line connecting entry to exit
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_time'], trade['exit_time']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='lines',
                line=dict(color=exit_color, width=1, dash='dash'),
                showlegend=False,
                opacity=0.3
            ),
            row=1, col=1
        )

    # Row 2: Holding time
    fig.add_trace(
        go.Histogram(
            x=trades_df['bars_held'],
            nbinsx=20,
            marker_color='purple',
            name='Hold Time'
        ),
        row=2, col=1
    )

    # Row 3: P&L per trade
    colors_pnl = ['green' if x > 0 else 'red' for x in trades_df['net_pnl']]
    fig.add_trace(
        go.Bar(
            x=list(range(len(trades_df))),
            y=trades_df['net_pnl']*100,
            marker_color=colors_pnl,
            name='P&L per Trade'
        ),
        row=3, col=1
    )

    # Row 4: Cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=list(range(len(trades_df))),
            y=trades_df['cumulative_pnl']*100,
            mode='lines',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            name='Cumulative P&L'
        ),
        row=4, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=4, col=1)

    # Row 5: Exit reasons
    exit_reason_counts = trades_df['exit_reason'].value_counts()
    fig.add_trace(
        go.Bar(
            x=exit_reason_counts.index,
            y=exit_reason_counts.values,
            marker_color='orange',
            text=exit_reason_counts.values,
            textposition='auto'
        ),
        row=5, col=1
    )

    fig.update_layout(
        title=f'SMA10 Exit Strategy | P&L: {total_pnl*100:.2f}% | Win Rate: {winning/len(trades_df)*100:.1f}% | Trades: {len(trades_df)} | Avg Hold: {trades_df["bars_held"].mean():.1f} bars',
        height=1600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top')
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Bars", row=2, col=1)
    fig.update_yaxes(title_text="P&L %", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative %", row=4, col=1)
    fig.update_yaxes(title_text="Count", row=5, col=1)

    fig.write_html('sma10_exit_strategy.html')
    fig.show()

    trades_df.to_csv('sma10_exit_trades.csv', index=False)

    print("\nFiles saved:")
    print("  - sma10_exit_strategy.html")
    print("  - sma10_exit_trades.csv")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON: Previous Strategies vs SMA10 Exit")
    print("="*60)
    print("\n1. Exit on First Reversal Bar:")
    print("   - P&L: -0.72%")
    print("   - Avg hold: 2.1 bars")
    print("   - Exit: 90.7% on trend reversal")
    print("\n2. Fixed Stop (0.12%):")
    print("   - P&L: -0.57%")
    print("   - Avg hold: 3.4 bars")
    print("   - Exit: 58.9% on trend break")
    print("\n3. SMA10 Exit (Current):")
    print(f"   - P&L: {total_pnl*100:.2f}%")
    print(f"   - Avg hold: {trades_df['bars_held'].mean():.1f} bars")
    print(f"   - Exit: {exit_counts.get('sma10_violation', 0)} ({exit_counts.get('sma10_violation', 0)/len(trades_df)*100:.1f}%) on SMA10 violation")

else:
    print("\nNo trades executed.")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
