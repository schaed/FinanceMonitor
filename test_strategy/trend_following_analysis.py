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

# Calculate returns
df['returns'] = df['close'].pct_change()
df['price_change'] = df['close'].diff()
df['upward'] = (df['price_change'] > 0).astype(int)

# ==========================================
# PART 1: Analyze Consecutive Upward Bars
# ==========================================
print("\n" + "="*60)
print("PART 1: CONSECUTIVE UPWARD BAR ANALYSIS")
print("="*60)

# Count consecutive upward bars
df['consecutive_up'] = 0
count = 0
for i in range(len(df)):
    if df['upward'].iloc[i] == 1:
        count += 1
        df.iloc[i, df.columns.get_loc('consecutive_up')] = count
    else:
        count = 0

# Find all instances where we have 2 consecutive upward bars
two_up_indices = df[df['consecutive_up'] == 2].index

print(f"\nTotal minute bars: {len(df)}")
print(f"Upward bars: {df['upward'].sum()} ({df['upward'].sum()/len(df)*100:.2f}%)")
print(f"Instances of 2 consecutive upward bars: {len(two_up_indices)}")

# Analyze what happens after 2 consecutive upward bars
continuation_analysis = []
for idx in two_up_indices:
    idx_pos = df.index.get_loc(idx)
    # Look ahead up to 10 bars
    for lookahead in range(1, min(11, len(df) - idx_pos)):
        next_idx = df.index[idx_pos + lookahead]
        next_move = df.loc[next_idx, 'upward']
        continuation_analysis.append({
            'lookahead': lookahead,
            'continued_up': next_move
        })

cont_df = pd.DataFrame(continuation_analysis)
continuation_summary = cont_df.groupby('lookahead').agg({
    'continued_up': ['mean', 'count']
}).round(4)

print("\nProbability of continuation after 2 consecutive upward bars:")
print(continuation_summary)

# Calculate expected number of bars that continue upward
expected_bars = 0
for i in range(1, 11):
    if i in continuation_summary.index:
        prob = continuation_summary.loc[i, ('continued_up', 'mean')]
        expected_bars += prob

print(f"\nExpected number of bars (out of next 10) that continue upward: {expected_bars:.2f}")

# More detailed analysis: how many bars in a row continue up?
streak_analysis = []
for idx in two_up_indices:
    idx_pos = df.index.get_loc(idx)
    streak = 0
    for lookahead in range(1, min(20, len(df) - idx_pos)):
        next_idx = df.index[idx_pos + lookahead]
        if df.loc[next_idx, 'upward'] == 1:
            streak += 1
        else:
            break
    streak_analysis.append(streak)

print(f"\nAfter 2 consecutive up bars, average continuation streak: {np.mean(streak_analysis):.2f} bars")
print(f"Median continuation streak: {np.median(streak_analysis):.0f} bars")
print(f"Max continuation streak: {np.max(streak_analysis)} bars")

# Distribution of continuation streaks
streak_counts = pd.Series(streak_analysis).value_counts().sort_index()
print("\nDistribution of continuation streaks:")
for streak, count in streak_counts.items():
    pct = count / len(streak_analysis) * 100
    print(f"  {streak} bars: {count} times ({pct:.1f}%)")

# ==========================================
# PART 2: Medium Frequency Trend Following Strategy
# ==========================================
print("\n" + "="*60)
print("PART 2: MEDIUM FREQUENCY TREND FOLLOWING STRATEGY")
print("="*60)

# Strategy: Enter long after 2 consecutive upward bars, hold for N bars or until stop loss
df['signal'] = 0
df['position'] = 0
df['entry_price'] = np.nan
df['exit_price'] = np.nan
df['trade_pnl'] = 0.0

# Strategy parameters
HOLD_BARS = 5  # Hold for 5 minutes
STOP_LOSS_PCT = 0.001  # 0.1% stop loss
TAKE_PROFIT_PCT = 0.002  # 0.2% take profit
TRANSACTION_COST = 0.0000  # 0.01% transaction cost

trades = []
position = None

for i in range(2, len(df)):
    current_idx = df.index[i]
    current_price = df.loc[current_idx, 'close']

    # Check if we have an open position
    if position is not None:
        bars_held = i - position['entry_idx']
        pnl_pct = (current_price - position['entry_price']) / position['entry_price']

        # Exit conditions
        exit_reason = None
        if pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = 'stop_loss'
        elif pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = 'take_profit'
        elif bars_held >= HOLD_BARS:
            exit_reason = 'time_exit'
        elif df.loc[current_idx, 'upward'] == 0:  # Exit on first down bar
            exit_reason = 'trend_break'

        if exit_reason:
            # Close position
            exit_price = current_price
            gross_pnl = (exit_price - position['entry_price']) / position['entry_price']
            net_pnl = gross_pnl - (2 * TRANSACTION_COST)  # Entry and exit costs

            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': current_idx,
                'exit_price': exit_price,
                'bars_held': bars_held,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason
            })

            df.loc[current_idx, 'exit_price'] = exit_price
            df.loc[current_idx, 'trade_pnl'] = net_pnl
            position = None

    # Entry condition: 2 consecutive upward bars
    if position is None and df.loc[current_idx, 'consecutive_up'] == 2:
        # Enter long position
        position = {
            'entry_idx': i,
            'entry_time': current_idx,
            'entry_price': current_price
        }
        df.loc[current_idx, 'signal'] = 1
        df.loc[current_idx, 'entry_price'] = current_price

# Create trades dataframe
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    print(f"\nTotal trades executed: {len(trades_df)}")
    print(f"Winning trades: {(trades_df['net_pnl'] > 0).sum()} ({(trades_df['net_pnl'] > 0).sum()/len(trades_df)*100:.1f}%)")
    print(f"Losing trades: {(trades_df['net_pnl'] < 0).sum()} ({(trades_df['net_pnl'] < 0).sum()/len(trades_df)*100:.1f}%)")
    print(f"\nAverage net P&L per trade: {trades_df['net_pnl'].mean()*100:.4f}%")
    print(f"Median net P&L per trade: {trades_df['net_pnl'].median()*100:.4f}%")
    print(f"Best trade: {trades_df['net_pnl'].max()*100:.4f}%")
    print(f"Worst trade: {trades_df['net_pnl'].min()*100:.4f}%")
    print(f"\nAverage holding period: {trades_df['bars_held'].mean():.2f} bars")

    print("\nExit reason breakdown:")
    print(trades_df['exit_reason'].value_counts())

    # Calculate cumulative P&L
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()

    print(f"\nTotal cumulative P&L: {trades_df['cumulative_pnl'].iloc[-1]*100:.4f}%")

    # Sharpe ratio (annualized)
    if trades_df['net_pnl'].std() > 0:
        # Assuming 390 trading minutes per day, 252 trading days per year
        trades_per_year = len(trades_df) / ((df.index[-1] - df.index[0]).days / 252)
        sharpe = (trades_df['net_pnl'].mean() * trades_per_year) / (trades_df['net_pnl'].std() * np.sqrt(trades_per_year))
        print(f"Sharpe Ratio: {sharpe:.2f}")

    # Maximum drawdown
    cumulative_returns = (1 + trades_df['net_pnl']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    print(f"Maximum drawdown: {max_drawdown*100:.4f}%")

    # ==========================================
    # PART 3: Visualization
    # ==========================================
    print("\n" + "="*60)
    print("PART 3: GENERATING PLOTS")
    print("="*60)

    # Create comprehensive visualization
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('SPY Price with Entry/Exit Points',
                       'Trade P&L Distribution',
                       'Cumulative P&L',
                       'Drawdown'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # Row 1: Price chart with entry/exit points
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

    # Add entry points
    entry_points = df[df['signal'] == 1]
    fig.add_trace(
        go.Scatter(
            x=entry_points.index,
            y=entry_points['close'],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Entry'
        ),
        row=1, col=1
    )

    # Add exit points
    exit_points = df[df['exit_price'].notna()]
    fig.add_trace(
        go.Scatter(
            x=exit_points.index,
            y=exit_points['exit_price'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Exit'
        ),
        row=1, col=1
    )

    # Row 2: Trade P&L distribution
    fig.add_trace(
        go.Bar(
            x=list(range(len(trades_df))),
            y=trades_df['net_pnl'] * 100,
            marker_color=['green' if x > 0 else 'red' for x in trades_df['net_pnl']],
            name='Trade P&L (%)'
        ),
        row=2, col=1
    )

    # Row 3: Cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=list(range(len(trades_df))),
            y=trades_df['cumulative_pnl'] * 100,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Cumulative P&L (%)'
        ),
        row=3, col=1
    )

    # Row 4: Drawdown
    fig.add_trace(
        go.Scatter(
            x=list(range(len(trades_df))),
            y=drawdown * 100,
            mode='lines',
            fill='tozeroy',
            line=dict(color='red', width=1),
            name='Drawdown (%)'
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        title='Medium Frequency Trend Following Strategy - SPY',
        height=1200,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="P&L (%)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative (%)", row=3, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)
    fig.update_xaxes(title_text="Trade Number", row=4, col=1)

    # Save and show
    fig.write_html('trend_following_strategy.html')
    print("\nPlot saved to: trend_following_strategy.html")
    fig.show()

    # Save trades to CSV
    trades_df.to_csv('trades_history.csv', index=False)
    print("Trade history saved to: trades_history.csv")

else:
    print("\nNo trades were executed with the current strategy parameters.")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
