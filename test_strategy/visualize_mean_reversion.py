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
print("VISUALIZING POLYNOMIAL MEAN REVERSION BANDS")
print("="*80)

# Fetch data
end_date = "2026-04-05"
end_dt = datetime.strptime(end_date, "%Y-%m-%d")

# 1-hour: best performer
start_1hour = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
df_1hour = api.get_bars(symbol="SPY", start=start_1hour, end=end_date, timeframe="1Hour").df
print(f"Loaded {len(df_1hour)} 1-hour bars")

# Load trades
trades_1hour = pd.read_csv('mean_reversion_1hour.csv')
trades_1hour['entry_time'] = pd.to_datetime(trades_1hour['entry_time'])
trades_1hour['exit_time'] = pd.to_datetime(trades_1hour['exit_time'])
print(f"Loaded {len(trades_1hour)} trades")

# Recalculate polynomial bands
def calculate_polynomial_bands(df, window_size, poly_degree=2):
    df = df.copy()
    df['poly_mean'] = np.nan
    df['poly_std'] = np.nan
    df['upper_1std'] = np.nan
    df['lower_1std'] = np.nan
    df['upper_2std'] = np.nan
    df['lower_2std'] = np.nan
    df['upper_3std'] = np.nan
    df['lower_3std'] = np.nan

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

            df.iloc[i, df.columns.get_loc('poly_mean')] = current_poly_value
            df.iloc[i, df.columns.get_loc('poly_std')] = std_dev
            df.iloc[i, df.columns.get_loc('upper_1std')] = current_poly_value + std_dev
            df.iloc[i, df.columns.get_loc('lower_1std')] = current_poly_value - std_dev
            df.iloc[i, df.columns.get_loc('upper_2std')] = current_poly_value + 2 * std_dev
            df.iloc[i, df.columns.get_loc('lower_2std')] = current_poly_value - 2 * std_dev
            df.iloc[i, df.columns.get_loc('upper_3std')] = current_poly_value + 3 * std_dev
            df.iloc[i, df.columns.get_loc('lower_3std')] = current_poly_value - 3 * std_dev
        except:
            continue

    return df

window_size = len(df_1hour) // 2
df_1hour = calculate_polynomial_bands(df_1hour, window_size)

# Create visualization
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        'SPY 1-Hour Chart with Polynomial Mean Reversion Bands',
        'Distance from Mean (Standard Deviations)',
        'Trade P&L'
    ),
    row_heights=[0.5, 0.25, 0.25]
)

# Row 1: Price chart with bands
fig.add_trace(
    go.Candlestick(
        x=df_1hour.index,
        open=df_1hour['open'],
        high=df_1hour['high'],
        low=df_1hour['low'],
        close=df_1hour['close'],
        name='SPY'
    ),
    row=1, col=1
)

# Add polynomial mean (quadratic fit)
fig.add_trace(
    go.Scatter(
        x=df_1hour.index,
        y=df_1hour['poly_mean'],
        mode='lines',
        line=dict(color='white', width=2),
        name='Polynomial Mean'
    ),
    row=1, col=1
)

# Add bands
for std_level, color, opacity in [(1, 'yellow', 0.2), (2, 'orange', 0.3), (3, 'red', 0.4)]:
    fig.add_trace(
        go.Scatter(
            x=df_1hour.index,
            y=df_1hour[f'upper_{std_level}std'],
            mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            name=f'+{std_level}σ',
            showlegend=(std_level == 2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_1hour.index,
            y=df_1hour[f'lower_{std_level}std'],
            mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            name=f'-{std_level}σ',
            fill='tonexty' if std_level == 1 else None,
            fillcolor=f'rgba(255, 165, 0, {opacity})' if std_level == 1 else None,
            showlegend=(std_level == 2)
        ),
        row=1, col=1
    )

# Add entry/exit points
for _, trade in trades_1hour.iterrows():
    entry_color = 'lime' if trade['position_type'] == 'long' else 'red'
    exit_color = 'green' if trade['net_pnl'] > 0 else 'darkred'

    # Entry
    fig.add_trace(
        go.Scatter(
            x=[trade['entry_time']],
            y=[trade['entry_price']],
            mode='markers',
            marker=dict(
                color=entry_color,
                size=15,
                symbol='triangle-up' if trade['position_type'] == 'long' else 'triangle-down',
                line=dict(width=2, color='white')
            ),
            name=f"{trade['position_type'].upper()} Entry",
            showlegend=False,
            hovertext=f"{trade['position_type'].upper()}<br>Size: {trade['position_size']:.1f}x<br>{trade['entry_std_distance']:.2f}σ"
        ),
        row=1, col=1
    )

    # Exit
    fig.add_trace(
        go.Scatter(
            x=[trade['exit_time']],
            y=[trade['exit_price']],
            mode='markers',
            marker=dict(
                color=exit_color,
                size=12,
                symbol='x',
                line=dict(width=2)
            ),
            name='Exit',
            showlegend=False,
            hovertext=f"P&L: {trade['net_pnl']*100:.2f}%<br>Held: {trade['bars_held']} bars"
        ),
        row=1, col=1
    )

    # Connect entry to exit
    fig.add_trace(
        go.Scatter(
            x=[trade['entry_time'], trade['exit_time']],
            y=[trade['entry_price'], trade['exit_price']],
            mode='lines',
            line=dict(color=exit_color, width=2, dash='dash'),
            showlegend=False,
            opacity=0.5
        ),
        row=1, col=1
    )

# Row 2: Standard deviation distance
df_1hour['std_distance'] = (df_1hour['close'] - df_1hour['poly_mean']) / df_1hour['poly_std']

fig.add_trace(
    go.Scatter(
        x=df_1hour.index,
        y=df_1hour['std_distance'],
        mode='lines',
        line=dict(color='cyan', width=1),
        name='Std Distance',
        fill='tozeroy'
    ),
    row=2, col=1
)

# Add horizontal lines at entry thresholds
fig.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="Short Entry", row=2, col=1)
fig.add_hline(y=-2, line_dash="dash", line_color="lime", annotation_text="Long Entry", row=2, col=1)
fig.add_hline(y=0, line_dash="solid", line_color="white", annotation_text="Mean", row=2, col=1)

# Row 3: Trade P&L
trades_1hour['cumulative_pnl'] = trades_1hour['net_pnl'].cumsum()

fig.add_trace(
    go.Bar(
        x=list(range(len(trades_1hour))),
        y=trades_1hour['net_pnl'] * 100,
        marker_color=['green' if x > 0 else 'red' for x in trades_1hour['net_pnl']],
        name='Trade P&L',
        showlegend=False
    ),
    row=3, col=1
)

# Add cumulative line
fig.add_trace(
    go.Scatter(
        x=list(range(len(trades_1hour))),
        y=trades_1hour['cumulative_pnl'] * 100,
        mode='lines+markers',
        line=dict(color='yellow', width=2),
        marker=dict(size=8),
        name='Cumulative P&L',
        yaxis='y4'
    ),
    row=3, col=1
)

# Layout
fig.update_layout(
    title={
        'text': 'Polynomial Mean Reversion Strategy - 1-Hour SPY<br>' +
                f'<sub>Quadratic Fit | {len(trades_1hour)} Trades | ' +
                f'Win Rate: {(trades_1hour["net_pnl"]>0).sum()/len(trades_1hour)*100:.1f}% | ' +
                f'Total P&L: {trades_1hour["net_pnl"].sum()*100:.2f}%</sub>',
        'x': 0.5,
        'xanchor': 'center'
    },
    height=1200,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top')
)

fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Std Devs (σ)", row=2, col=1)
fig.update_yaxes(title_text="Trade P&L (%)", row=3, col=1)
fig.update_xaxes(title_text="Trade Number", row=3, col=1)

fig.write_html('mean_reversion_bands.html')
fig.show()

print("\nVisualization saved: mean_reversion_bands.html")
print("\nStrategy Explanation:")
print("  - White line: Polynomial (quadratic) mean")
print("  - Yellow/Orange/Red bands: 1σ, 2σ, 3σ from mean")
print("  - Green ▲: Long entry (price < -2σ)")
print("  - Red ▼: Short entry (price > +2σ)")
print("  - X markers: Exits (when price returns to mean)")
print("  - Position size scales with distance (2σ = 2x, 3σ = 3x)")
