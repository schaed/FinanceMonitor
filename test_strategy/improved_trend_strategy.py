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

# Moving averages (trend strength)
df['sma_5'] = df['close'].rolling(window=5).mean()
df['sma_10'] = df['close'].rolling(window=10).mean()
df['sma_20'] = df['close'].rolling(window=20).mean()

# Exponential moving average
df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

# Price relative to moving averages
df['price_above_sma10'] = (df['close'] > df['sma_10']).astype(int)
df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)

# Momentum indicators
df['momentum_5'] = df['close'] - df['close'].shift(5)
df['momentum_10'] = df['close'] - df['close'].shift(10)
df['roc_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100  # Rate of change

# Volatility (ATR - Average True Range)
df['high_low'] = df['high'] - df['low']
df['high_close'] = abs(df['high'] - df['close'].shift())
df['low_close'] = abs(df['low'] - df['close'].shift())
df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr_10'] = df['tr'].rolling(window=10).mean()

# Volume analysis
df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma_10']

# RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi_14'] = calculate_rsi(df['close'], window=14)

# Count consecutive bars
df['consecutive_up'] = 0
count = 0
for i in range(len(df)):
    if df['upward'].iloc[i] == 1:
        count += 1
        df.iloc[i, df.columns.get_loc('consecutive_up')] = count
    else:
        count = 0

# Trend strength: EMA crossover
df['ema_crossover'] = (df['ema_5'] > df['ema_10']).astype(int)

# ==========================================
# IMPROVED STRATEGY
# ==========================================
print("\n" + "="*60)
print("IMPROVED MEDIUM FREQUENCY TREND FOLLOWING STRATEGY")
print("="*60)

# Strategy: Enter when multiple conditions align
# 1. 2+ consecutive upward bars
# 2. Price above 20-period SMA (uptrend)
# 3. Strong momentum (5-period ROC > threshold)
# 4. RSI between 40-70 (not overbought/oversold)
# 5. Above average volume

# Strategy parameters
HOLD_BARS_MIN = 3
HOLD_BARS_MAX = 50
STOP_LOSS_PCT = 0.0015  # 0.15% stop loss
TRAILING_STOP_PCT = 0.001  # 0.1% trailing stop
TAKE_PROFIT_PCT = 0.03  # 0.3% take profit
TRANSACTION_COST = 0.0001  # 0.01% per trade

# Signal conditions
df['signal'] = 0

# Wait for indicators to be calculated
start_idx = 25

for i in range(start_idx, len(df)):
    current_idx = df.index[i]

    # Entry conditions
    condition1 = df.loc[current_idx, 'consecutive_up'] >= 2
    condition2 = df.loc[current_idx, 'price_above_sma20'] == 1
    condition3 = df.loc[current_idx, 'momentum_5'] > 0
    condition4 = 40 < df.loc[current_idx, 'rsi_14'] < 70
    condition5 = df.loc[current_idx, 'volume_ratio'] > 0.8
    condition6 = df.loc[current_idx, 'ema_crossover'] == 1  # EMA 5 > EMA 10

    if all([condition1, condition2, condition3, condition4, condition5, condition6]):
        df.loc[current_idx, 'signal'] = 1

print(f"Total entry signals: {df['signal'].sum()}")

# Execute trades
trades = []
position = None

for i in range(start_idx, len(df)):
    current_idx = df.index[i]
    current_price = df.loc[current_idx, 'close']
    current_high = df.loc[current_idx, 'high']
    current_low = df.loc[current_idx, 'low']

    # Check if we have an open position
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

        # Check stop loss or trailing stop
        if current_low <= position['stop_loss']:
            exit_reason = 'stop_loss'
            exit_price = position['stop_loss']
        elif current_low <= position['trailing_stop']:
            exit_reason = 'trailing_stop'
            exit_price = position['trailing_stop']
        elif current_high >= position['take_profit']:
            exit_reason = 'take_profit'
            exit_price = position['take_profit']
        elif bars_held >= HOLD_BARS_MAX:
            exit_reason = 'max_time'
        elif bars_held >= HOLD_BARS_MIN and df.loc[current_idx, 'upward'] == 0:
            exit_reason = 'trend_break'

        if exit_reason:
            # Close position
            gross_pnl = (exit_price - position['entry_price']) / position['entry_price']
            net_pnl = gross_pnl - (2 * TRANSACTION_COST)

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

            position = None

    # Entry condition
    if position is None and df.loc[current_idx, 'signal'] == 1:
        # Enter long position
        position = {
            'entry_idx': i,
            'entry_time': current_idx,
            'entry_price': current_price,
            'stop_loss': current_price * (1 - STOP_LOSS_PCT),
            'take_profit': current_price * (1 + TAKE_PROFIT_PCT),
            'highest_price': current_price,
            'trailing_stop': current_price * (1 - TRAILING_STOP_PCT)
        }

# Create trades dataframe
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    print(f"\n{'='*60}")
    print("STRATEGY PERFORMANCE")
    print('='*60)

    print(f"\nTotal trades executed: {len(trades_df)}")
    winning_trades = (trades_df['net_pnl'] > 0).sum()
    losing_trades = (trades_df['net_pnl'] < 0).sum()
    print(f"Winning trades: {winning_trades} ({winning_trades/len(trades_df)*100:.1f}%)")
    print(f"Losing trades: {losing_trades} ({losing_trades/len(trades_df)*100:.1f}%)")

    print(f"\nP&L Statistics:")
    print(f"  Average net P&L per trade: {trades_df['net_pnl'].mean()*100:.4f}%")
    print(f"  Median net P&L per trade: {trades_df['net_pnl'].median()*100:.4f}%")
    print(f"  Best trade: {trades_df['net_pnl'].max()*100:.4f}%")
    print(f"  Worst trade: {trades_df['net_pnl'].min()*100:.4f}%")
    print(f"  Std dev: {trades_df['net_pnl'].std()*100:.4f}%")

    print(f"\nHolding Period:")
    print(f"  Average: {trades_df['bars_held'].mean():.2f} bars")
    print(f"  Median: {trades_df['bars_held'].median():.0f} bars")

    print("\nExit Reason Breakdown:")
    exit_counts = trades_df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} ({count/len(trades_df)*100:.1f}%)")

    # Calculate cumulative P&L
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()

    print(f"\nTotal Cumulative P&L: {trades_df['cumulative_pnl'].iloc[-1]*100:.4f}%")

    # Win/Loss ratio
    avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()
    avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean())
    win_rate = winning_trades / len(trades_df)

    if avg_loss > 0:
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades)
        print(f"\nProfit Factor: {profit_factor:.2f}")

    print(f"Average Win: {avg_win*100:.4f}%")
    print(f"Average Loss: {avg_loss*100:.4f}%")
    if avg_loss > 0:
        print(f"Win/Loss Ratio: {avg_win/avg_loss:.2f}")

    # Sharpe ratio
    if trades_df['net_pnl'].std() > 0:
        days_traded = (df.index[-1] - df.index[0]).days
        trades_per_day = len(trades_df) / days_traded
        annual_return = trades_df['net_pnl'].mean() * trades_per_day * 252
        annual_vol = trades_df['net_pnl'].std() * np.sqrt(trades_per_day * 252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        print(f"\nAnnualized Sharpe Ratio: {sharpe:.2f}")

    # Maximum drawdown
    cumulative_returns = (1 + trades_df['net_pnl']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown*100:.4f}%")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE PLOTS")
    print("="*60)

    # Create subplot
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            'SPY Price with Entry/Exit Points',
            'RSI Indicator',
            'Trade P&L per Trade',
            'Cumulative P&L',
            'Drawdown'
        ),
        row_heights=[0.35, 0.15, 0.15, 0.20, 0.15]
    )

    # Row 1: Price chart with indicators
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='SPY',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # Add SMAs
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['sma_20'],
            mode='lines',
            line=dict(color='orange', width=1),
            name='SMA 20'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ema_5'],
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='EMA 5'
        ),
        row=1, col=1
    )

    # Add entry points
    entry_times = trades_df['entry_time'].values
    entry_prices = trades_df['entry_price'].values
    fig.add_trace(
        go.Scatter(
            x=entry_times,
            y=entry_prices,
            mode='markers',
            marker=dict(color='green', size=12, symbol='triangle-up', line=dict(width=2, color='darkgreen')),
            name='Entry'
        ),
        row=1, col=1
    )

    # Add exit points
    exit_times = trades_df['exit_time'].values
    exit_prices = trades_df['exit_price'].values
    exit_colors = ['lime' if pnl > 0 else 'red' for pnl in trades_df['net_pnl']]
    fig.add_trace(
        go.Scatter(
            x=exit_times,
            y=exit_prices,
            mode='markers',
            marker=dict(color=exit_colors, size=12, symbol='triangle-down', line=dict(width=2, color='darkred')),
            name='Exit'
        ),
        row=1, col=1
    )

    # Row 2: RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi_14'],
            mode='lines',
            line=dict(color='purple', width=1),
            name='RSI'
        ),
        row=2, col=1
    )

    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=40, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # Row 3: Individual trade P&L
    colors = ['green' if x > 0 else 'red' for x in trades_df['net_pnl']]
    fig.add_trace(
        go.Bar(
            x=list(range(len(trades_df))),
            y=trades_df['net_pnl'] * 100,
            marker_color=colors,
            name='Trade P&L (%)',
            showlegend=False
        ),
        row=3, col=1
    )

    # Row 4: Cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=list(range(len(trades_df))),
            y=trades_df['cumulative_pnl'] * 100,
            mode='lines',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            name='Cumulative P&L (%)',
            showlegend=False
        ),
        row=4, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=4, col=1)

    # Row 5: Drawdown
    fig.add_trace(
        go.Scatter(
            x=list(range(len(trades_df))),
            y=drawdown * 100,
            mode='lines',
            fill='tozeroy',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)',
            name='Drawdown (%)',
            showlegend=False
        ),
        row=5, col=1
    )

    # Update layout
    fig.update_layout(
        title={
            'text': f'Improved Medium Frequency Trend Following Strategy - SPY<br>' +
                    f'<sub>Win Rate: {win_rate*100:.1f}% | Total P&L: {trades_df["cumulative_pnl"].iloc[-1]*100:.2f}% | ' +
                    f'Sharpe: {sharpe:.2f} | Trades: {len(trades_df)}</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=1400,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="P&L (%)", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative (%)", row=4, col=1)
    fig.update_yaxes(title_text="DD (%)", row=5, col=1)
    fig.update_xaxes(title_text="Trade Number", row=5, col=1)

    # Save
    fig.write_html('improved_trend_strategy.html')
    print("\nPlot saved to: improved_trend_strategy.html")
    fig.show()

    # Save trades
    trades_df.to_csv('improved_trades_history.csv', index=False)
    print("Trade history saved to: improved_trades_history.csv")

    # Create comparison summary
    print("\n" + "="*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*60)
    print("\nSimple Strategy (2 consecutive up bars only):")
    print("  - Total P&L: -14.32%")
    print("  - Win Rate: 18.4%")
    print("  - Sharpe: -39.12")
    print("  - Trades: 784")
    print("\nImproved Strategy (multi-factor confirmation):")
    print(f"  - Total P&L: {trades_df['cumulative_pnl'].iloc[-1]*100:.2f}%")
    print(f"  - Win Rate: {win_rate*100:.1f}%")
    print(f"  - Sharpe: {sharpe:.2f}")
    print(f"  - Trades: {len(trades_df)}")

else:
    print("\nNo trades were executed with the current strategy parameters.")
    print("Try relaxing some conditions or adjusting the lookback periods.")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
