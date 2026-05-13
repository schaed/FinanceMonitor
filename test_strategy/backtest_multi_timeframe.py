import alpaca_trade_api as alpaca
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = "SPY"  # Configurable stock symbol
BACKTEST_MONTHS = 6  # Number of months to backtest
TRANSACTION_COST = 0.0001  # 0.01% per trade

# Setup API
YOUR_API_SECRET_KEY = os.getenv('ALPACA_PAPER_KEY')
ALPACA_ID = os.getenv('ALPACA_ID')
api = alpaca.REST(ALPACA_ID, YOUR_API_SECRET_KEY, 'https://paper-api.alpaca.markets/v2')

print("="*80)
print(f"MULTI-TIMEFRAME MEAN REVERSION BACKTEST")
print(f"Symbol: {SYMBOL} | Period: {BACKTEST_MONTHS} months")
print("="*80)

# ==========================================
# CALCULATE POLYNOMIAL BANDS
# ==========================================

def calculate_polynomial_bands(df, window_size, poly_degree=2):
    """Calculate polynomial regression bands"""
    df = df.copy()

    df['poly_mean'] = np.nan
    df['poly_std'] = np.nan
    df['std_distance'] = np.nan
    df['upper_2std'] = np.nan
    df['lower_2std'] = np.nan

    min_points = max(window_size, 7)

    for i in range(min_points, len(df)):
        window = df.iloc[i-window_size:i+1]
        x = np.arange(len(window))
        y = window['close'].values

        try:
            coeffs = np.polyfit(x, y, poly_degree)
            poly_fit = np.polyval(coeffs, x)
            residuals = y - poly_fit
            std_dev = np.std(residuals)

            current_poly_value = poly_fit[-1]
            current_price = y[-1]

            df.iloc[i, df.columns.get_loc('poly_mean')] = current_poly_value
            df.iloc[i, df.columns.get_loc('poly_std')] = std_dev
            df.iloc[i, df.columns.get_loc('upper_2std')] = current_poly_value + 2 * std_dev
            df.iloc[i, df.columns.get_loc('lower_2std')] = current_poly_value - 2 * std_dev

            if std_dev > 0:
                distance = current_price - current_poly_value
                df.iloc[i, df.columns.get_loc('std_distance')] = distance / std_dev
        except:
            continue

    return df

def get_current_signal(df):
    """Get current signal from a timeframe"""
    if len(df) == 0:
        return {'signal': 'neutral', 'strength': 0, 'std_distance': 0}

    latest = df.iloc[-1]

    if pd.isna(latest['std_distance']):
        return {'signal': 'neutral', 'strength': 0, 'std_distance': 0}

    std_dist = latest['std_distance']

    if std_dist <= -2.0:
        strength = min(abs(std_dist), 5)
        return {
            'signal': 'long',
            'strength': strength,
            'std_distance': std_dist,
            'price': latest['close'],
            'poly_mean': latest['poly_mean']
        }
    elif std_dist >= 2.0:
        strength = min(std_dist, 5)
        return {
            'signal': 'short',
            'strength': strength,
            'std_distance': std_dist,
            'price': latest['close'],
            'poly_mean': latest['poly_mean']
        }
    else:
        return {
            'signal': 'neutral',
            'strength': abs(std_dist),
            'std_distance': std_dist,
            'price': latest['close'],
            'poly_mean': latest['poly_mean']
        }

def make_trading_decision(signals):
    """Combine signals from multiple timeframes into a single decision"""
    weights = {
        '1-min': 0.25,
        '15-min': 0.5,
        '1-hour': 3.0,
        'daily': 2.0
    }

    long_score = 0
    short_score = 0

    for tf, sig in signals.items():
        weight = weights[tf]
        if sig['signal'] == 'long':
            long_score += sig['strength'] * weight
        elif sig['signal'] == 'short':
            short_score += sig['strength'] * weight

    long_count = sum(1 for sig in signals.values() if sig['signal'] == 'long')
    short_count = sum(1 for sig in signals.values() if sig['signal'] == 'short')

    decision = {
        'action': 'neutral',
        'confidence': 'none',
        'position_size': 0,
        'timeframes_aligned': 0,
        'weighted_score': 0
    }

    if long_score > short_score and long_score > 0:
        decision['action'] = 'long'
        decision['weighted_score'] = long_score
        decision['timeframes_aligned'] = long_count
        base_size = min(long_score / 10, 5.0)
        decision['position_size'] = base_size

        if long_count >= 3:
            decision['confidence'] = 'high'
        elif long_count >= 2:
            decision['confidence'] = 'medium'
        else:
            decision['confidence'] = 'low'

    elif short_score > long_score and short_score > 0:
        decision['action'] = 'short'
        decision['weighted_score'] = short_score
        decision['timeframes_aligned'] = short_count
        base_size = min(short_score / 10, 5.0)
        decision['position_size'] = base_size

        if short_count >= 3:
            decision['confidence'] = 'high'
        elif short_count >= 2:
            decision['confidence'] = 'medium'
        else:
            decision['confidence'] = 'low'
    else:
        decision['action'] = 'neutral'
        decision['confidence'] = 'none'

    return decision

# ==========================================
# FETCH BACKTEST DATA
# ==========================================
print(f"\n1. Fetching {BACKTEST_MONTHS} months of data for all timeframes...")

end_date = "2026-04-07"
end_dt = datetime.strptime(end_date, "%Y-%m-%d")
start_backtest = (end_dt - timedelta(days=BACKTEST_MONTHS * 30)).strftime("%Y-%m-%d")

# Fetch extended data for each timeframe (need extra for lookback windows)
start_1min = (end_dt - timedelta(days=BACKTEST_MONTHS * 30 + 2)).strftime("%Y-%m-%d")
df_1min_full = api.get_bars(symbol=SYMBOL, start=start_1min, end=end_date, timeframe="1Min").df
print(f"   1-min bars: {len(df_1min_full)}")

start_15min = (end_dt - timedelta(days=BACKTEST_MONTHS * 30 + 7)).strftime("%Y-%m-%d")
df_15min_full = api.get_bars(symbol=SYMBOL, start=start_15min, end=end_date, timeframe="15Min").df
print(f"   15-min bars: {len(df_15min_full)}")

start_1hour = (end_dt - timedelta(days=BACKTEST_MONTHS * 30 + 60)).strftime("%Y-%m-%d")
df_1hour_full = api.get_bars(symbol=SYMBOL, start=start_1hour, end=end_date, timeframe="1Hour").df
print(f"   1-hour bars: {len(df_1hour_full)}")

start_daily = (end_dt - timedelta(days=BACKTEST_MONTHS * 30 + 365)).strftime("%Y-%m-%d")
df_daily_full = api.get_bars(symbol=SYMBOL, start=start_daily, end=end_date, timeframe="1Day").df
print(f"   Daily bars: {len(df_daily_full)}")

# Get hourly bars for backtest timeline (we'll make decisions on hourly basis)
df_backtest = api.get_bars(symbol=SYMBOL, start=start_backtest, end=end_date, timeframe="1Hour").df
print(f"\n   Backtest bars (1-hour): {len(df_backtest)}")

# ==========================================
# RUN BACKTEST
# ==========================================
print(f"\n2. Running backtest...")

trades = []
position = None
equity_curve = []
initial_capital = 100000
current_capital = initial_capital

decision_log = []

for i in range(len(df_backtest)):
    current_time = df_backtest.index[i]
    current_price = df_backtest.loc[current_time, 'close']

    # Get data up to current time for each timeframe
    df_1min_slice = df_1min_full[df_1min_full.index <= current_time]
    df_15min_slice = df_15min_full[df_15min_full.index <= current_time]
    df_1hour_slice = df_1hour_full[df_1hour_full.index <= current_time]
    df_daily_slice = df_daily_full[df_daily_full.index <= current_time]

    # Skip if not enough data for any timeframe
    if len(df_1min_slice) < 100 or len(df_15min_slice) < 50 or \
       len(df_1hour_slice) < 50 or len(df_daily_slice) < 50:
        equity_curve.append({
            'time': current_time,
            'equity': current_capital,
            'price': current_price
        })
        continue

    # Calculate polynomial bands for each timeframe
    df_1min_bands = calculate_polynomial_bands(df_1min_slice.tail(500), len(df_1min_slice.tail(500))//2)
    df_15min_bands = calculate_polynomial_bands(df_15min_slice.tail(500), len(df_15min_slice.tail(500))//2)
    df_1hour_bands = calculate_polynomial_bands(df_1hour_slice.tail(500), len(df_1hour_slice.tail(500))//2)
    df_daily_bands = calculate_polynomial_bands(df_daily_slice.tail(250), len(df_daily_slice.tail(250))//2)

    # Get signals
    signals = {
        '1-min': get_current_signal(df_1min_bands),
        '15-min': get_current_signal(df_15min_bands),
        '1-hour': get_current_signal(df_1hour_bands),
        'daily': get_current_signal(df_daily_bands)
    }

    # Make trading decision
    decision = make_trading_decision(signals)

    # Log decision
    decision_log.append({
        'time': current_time,
        'price': current_price,
        'action': decision['action'],
        'confidence': decision['confidence'],
        'position_size': decision['position_size'],
        'weighted_score': decision['weighted_score'],
        '1min_signal': signals['1-min']['signal'],
        '1min_std': signals['1-min']['std_distance'],
        '15min_signal': signals['15-min']['signal'],
        '15min_std': signals['15-min']['std_distance'],
        '1hour_signal': signals['1-hour']['signal'],
        '1hour_std': signals['1-hour']['std_distance'],
        'daily_signal': signals['daily']['signal'],
        'daily_std': signals['daily']['std_distance']
    })

    # Manage existing position
    if position is not None:
        position_type = position['type']
        entry_price = position['entry_price']
        poly_mean = signals['1-hour']['poly_mean']  # Use 1-hour as reference

        exit_triggered = False
        exit_reason = None

        if position_type == 'long':
            # Exit if price returns to mean or signal flips
            if not pd.isna(poly_mean) and current_price >= poly_mean:
                exit_triggered = True
                exit_reason = 'mean_reversion'
            elif decision['action'] == 'short':
                exit_triggered = True
                exit_reason = 'signal_flip'
        else:  # short
            if not pd.isna(poly_mean) and current_price <= poly_mean:
                exit_triggered = True
                exit_reason = 'mean_reversion'
            elif decision['action'] == 'long':
                exit_triggered = True
                exit_reason = 'signal_flip'

        if exit_triggered:
            # Close position
            if position_type == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            gross_pnl = pnl_pct * position['size']
            net_pnl = gross_pnl - (2 * TRANSACTION_COST * position['size'])

            position_value = current_capital * position['size']
            pnl_dollars = position_value * net_pnl
            current_capital += pnl_dollars

            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': entry_price,
                'exit_time': current_time,
                'exit_price': current_price,
                'position_type': position_type,
                'position_size': position['size'],
                'confidence': position['confidence'],
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'pnl_dollars': pnl_dollars,
                'exit_reason': exit_reason
            })

            position = None

    # Enter new position if no position and strong signal
    if position is None and decision['action'] != 'neutral':
        # Only enter on medium or high confidence
        if decision['confidence'] in ['medium', 'high']:
            position = {
                'entry_time': current_time,
                'entry_price': current_price,
                'type': decision['action'],
                'size': decision['position_size'],
                'confidence': decision['confidence']
            }

    # Track equity
    equity_value = current_capital
    if position is not None:
        position_value = current_capital * position['size']
        if position['type'] == 'long':
            unrealized_pnl = (current_price - position['entry_price']) / position['entry_price']
        else:
            unrealized_pnl = (position['entry_price'] - current_price) / position['entry_price']
        equity_value = current_capital + (position_value * unrealized_pnl)

    equity_curve.append({
        'time': current_time,
        'equity': equity_value,
        'price': current_price,
        'position': position['type'] if position else 'neutral'
    })

# Close final position if exists
if position is not None:
    final_price = df_backtest.iloc[-1]['close']
    final_time = df_backtest.index[-1]

    if position['type'] == 'long':
        pnl_pct = (final_price - position['entry_price']) / position['entry_price']
    else:
        pnl_pct = (position['entry_price'] - final_price) / position['entry_price']

    gross_pnl = pnl_pct * position['size']
    net_pnl = gross_pnl - (2 * TRANSACTION_COST * position['size'])

    position_value = current_capital * position['size']
    pnl_dollars = position_value * net_pnl
    current_capital += pnl_dollars

    trades.append({
        'entry_time': position['entry_time'],
        'entry_price': position['entry_price'],
        'exit_time': final_time,
        'exit_price': final_price,
        'position_type': position['type'],
        'position_size': position['size'],
        'confidence': position['confidence'],
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'pnl_dollars': pnl_dollars,
        'exit_reason': 'backtest_end'
    })

# ==========================================
# RESULTS ANALYSIS
# ==========================================
print("\n" + "="*80)
print("BACKTEST RESULTS")
print("="*80)

trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity_curve)

if len(trades_df) > 0:
    print(f"\nTotal Trades: {len(trades_df)}")

    long_trades = trades_df[trades_df['position_type'] == 'long']
    short_trades = trades_df[trades_df['position_type'] == 'short']

    print(f"  Long: {len(long_trades)} | Short: {len(short_trades)}")

    winning_trades = trades_df[trades_df['net_pnl'] > 0]
    win_rate = len(winning_trades) / len(trades_df) * 100
    print(f"\nWin Rate: {win_rate:.1f}%")
    print(f"  Winning: {len(winning_trades)} | Losing: {len(trades_df) - len(winning_trades)}")

    total_return = (current_capital - initial_capital) / initial_capital * 100
    print(f"\nTotal Return: {total_return:.2f}%")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Final Capital: ${current_capital:,.2f}")
    print(f"  Total P&L: ${current_capital - initial_capital:,.2f}")

    avg_pnl = trades_df['net_pnl'].mean() * 100
    print(f"\nAvg Trade P&L: {avg_pnl:.2f}%")
    print(f"  Best: {trades_df['net_pnl'].max()*100:.2f}%")
    print(f"  Worst: {trades_df['net_pnl'].min()*100:.2f}%")

    print(f"\nPosition Sizing:")
    print(f"  Avg: {trades_df['position_size'].mean():.2f}x")
    print(f"  Max: {trades_df['position_size'].max():.2f}x")

    print(f"\nConfidence Breakdown:")
    for conf in ['high', 'medium', 'low']:
        conf_trades = trades_df[trades_df['confidence'] == conf]
        if len(conf_trades) > 0:
            conf_win_rate = (conf_trades['net_pnl'] > 0).sum() / len(conf_trades) * 100
            conf_return = conf_trades['net_pnl'].sum() * 100
            print(f"  {conf.upper()}: {len(conf_trades)} trades, {conf_win_rate:.1f}% win rate, {conf_return:.2f}% total return")

    # Drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
    max_drawdown = equity_df['drawdown'].min()
    print(f"\nMax Drawdown: {max_drawdown:.2f}%")

    # Sharpe ratio (simplified, assuming 252 trading days)
    returns = trades_df['net_pnl']
    if len(returns) > 1:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / BACKTEST_MONTHS)
        print(f"Sharpe Ratio (annualized): {sharpe:.2f}")

else:
    print("\nNo trades executed during backtest period")

# ==========================================
# SAVE RESULTS
# ==========================================
print(f"\n3. Saving results...")

if len(trades_df) > 0:
    trades_df.to_csv(f'backtest_trades_{SYMBOL}.csv', index=False)
    print(f"   Saved: backtest_trades_{SYMBOL}.csv")

equity_df.to_csv(f'backtest_equity_{SYMBOL}.csv', index=False)
print(f"   Saved: backtest_equity_{SYMBOL}.csv")

decision_log_df = pd.DataFrame(decision_log)
decision_log_df.to_csv(f'backtest_decisions_{SYMBOL}.csv', index=False)
print(f"   Saved: backtest_decisions_{SYMBOL}.csv")

# ==========================================
# VISUALIZATION
# ==========================================
print(f"\n4. Creating backtest visualization...")

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        f'{SYMBOL} Price Chart with Trades',
        'Equity Curve',
        'Drawdown',
        'Trade P&L Distribution'
    ),
    row_heights=[0.35, 0.25, 0.20, 0.20]
)

# Row 1: Price chart with entry/exit points
fig.add_trace(
    go.Scatter(
        x=df_backtest.index,
        y=df_backtest['close'],
        mode='lines',
        line=dict(color='cyan', width=1),
        name=f'{SYMBOL} Price'
    ),
    row=1, col=1
)

if len(trades_df) > 0:
    # Entry points
    for _, trade in trades_df.iterrows():
        color = 'lime' if trade['position_type'] == 'long' else 'red'
        symbol = 'triangle-up' if trade['position_type'] == 'long' else 'triangle-down'

        fig.add_trace(
            go.Scatter(
                x=[trade['entry_time']],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(color=color, size=10, symbol=symbol, line=dict(width=2, color='white')),
                showlegend=False,
                hovertext=f"{trade['position_type'].upper()}<br>Size: {trade['position_size']:.1f}x<br>Conf: {trade['confidence']}"
            ),
            row=1, col=1
        )

        # Exit points
        exit_color = 'green' if trade['net_pnl'] > 0 else 'darkred'
        fig.add_trace(
            go.Scatter(
                x=[trade['exit_time']],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(color=exit_color, size=8, symbol='x', line=dict(width=2)),
                showlegend=False,
                hovertext=f"P&L: {trade['net_pnl']*100:.2f}%"
            ),
            row=1, col=1
        )

# Row 2: Equity curve
fig.add_trace(
    go.Scatter(
        x=equity_df['time'],
        y=equity_df['equity'],
        mode='lines',
        line=dict(color='gold', width=2),
        name='Equity',
        fill='tozeroy'
    ),
    row=2, col=1
)

# Add buy&hold comparison
initial_price = df_backtest.iloc[0]['close']
equity_df['buyhold'] = initial_capital * (df_backtest['close'] / initial_price)
fig.add_trace(
    go.Scatter(
        x=equity_df['time'],
        y=equity_df['buyhold'],
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        name='Buy & Hold'
    ),
    row=2, col=1
)

# Row 3: Drawdown
fig.add_trace(
    go.Scatter(
        x=equity_df['time'],
        y=equity_df['drawdown'],
        mode='lines',
        line=dict(color='red', width=1),
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)'
    ),
    row=3, col=1
)

# Row 4: Trade P&L distribution
if len(trades_df) > 0:
    fig.add_trace(
        go.Bar(
            x=list(range(len(trades_df))),
            y=trades_df['net_pnl'] * 100,
            marker_color=['green' if x > 0 else 'red' for x in trades_df['net_pnl']],
            name='Trade P&L',
            showlegend=False
        ),
        row=4, col=1
    )

# Layout
strategy_return = (current_capital - initial_capital) / initial_capital * 100
buyhold_return = (df_backtest.iloc[-1]['close'] / df_backtest.iloc[0]['close'] - 1) * 100

fig.update_layout(
    title={
        'text': f'Multi-Timeframe Mean Reversion Backtest - {SYMBOL}<br>' +
                f'<sub>{BACKTEST_MONTHS} months | Strategy: {strategy_return:.2f}% | Buy&Hold: {buyhold_return:.2f}% | ' +
                f'Trades: {len(trades_df)} | Win Rate: {win_rate:.1f}%</sub>',
        'x': 0.5,
        'xanchor': 'center'
    },
    height=1400,
    showlegend=True,
    template='plotly_dark',
    hovermode='x unified'
)

fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
fig.update_yaxes(title_text="Trade P&L (%)", row=4, col=1)
fig.update_xaxes(title_text="Date", row=4, col=1)

fig.write_html(f'backtest_results_{SYMBOL}.html')
fig.show()

print(f"   Saved: backtest_results_{SYMBOL}.html")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
