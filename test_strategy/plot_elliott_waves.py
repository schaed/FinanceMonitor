import alpaca_trade_api as alpaca
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

# Setup API
YOUR_API_SECRET_KEY = os.getenv('ALPACA_PAPER_KEY')
ALPACA_ID = os.getenv('ALPACA_ID')
api = alpaca.REST(ALPACA_ID, YOUR_API_SECRET_KEY, 'https://paper-api.alpaca.markets/v2')

print("Fetching data and detecting Elliott Waves...")

end_date = "2026-04-05"

# Get 15-minute data for detailed wave analysis
df_15min = api.get_bars(symbol="SPY", start="2026-03-01", end=end_date, timeframe="15Min").df
print(f"15-min bars: {len(df_15min)}")

# ==========================================
# ELLIOTT WAVE DETECTION
# ==========================================

def find_swing_points(df, order=3):
    """Find swing highs and lows"""
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

def find_impulse_waves(pivots, direction='up'):
    """Find 5-wave impulse patterns"""
    waves = []

    if len(pivots) < 6:
        return waves

    # Scan through pivots looking for 5-wave patterns
    for i in range(len(pivots) - 5):
        pattern = pivots.iloc[i:i+6]

        if direction == 'up':
            expected_sequence = ['low', 'high', 'low', 'high', 'low', 'high']
        else:
            expected_sequence = ['high', 'low', 'high', 'low', 'high', 'low']

        actual_sequence = pattern['type'].tolist()

        if actual_sequence == expected_sequence:
            prices = pattern['price'].values

            if direction == 'up':
                # Bullish impulse validation
                wave1 = prices[1] - prices[0]
                wave2 = prices[2] - prices[1]
                wave3 = prices[3] - prices[2]
                wave4 = prices[4] - prices[3]
                wave5 = prices[5] - prices[4]

                # Elliott Wave rules
                if prices[2] <= prices[0]:  # Wave 2 retraces >100%
                    continue
                if wave3 < wave1 and wave3 < wave5:  # Wave 3 shortest
                    continue
                if prices[4] <= prices[1]:  # Wave 4 overlaps Wave 1
                    continue

                waves.append({
                    'direction': 'bullish',
                    'start_idx': pattern.index[0],
                    'end_idx': pattern.index[5],
                    'pivots': pattern.index.tolist(),
                    'prices': prices.tolist(),
                    'wave_labels': ['0', '1', '2', '3', '4', '5']
                })

            else:
                # Bearish impulse validation
                wave1 = prices[0] - prices[1]
                wave2 = prices[1] - prices[2]
                wave3 = prices[2] - prices[3]
                wave4 = prices[3] - prices[4]
                wave5 = prices[4] - prices[5]

                if prices[2] >= prices[0]:
                    continue
                if wave3 < wave1 and wave3 < wave5:
                    continue
                if prices[4] >= prices[1]:
                    continue

                waves.append({
                    'direction': 'bearish',
                    'start_idx': pattern.index[0],
                    'end_idx': pattern.index[5],
                    'pivots': pattern.index.tolist(),
                    'prices': prices.tolist(),
                    'wave_labels': ['0', '1', '2', '3', '4', '5']
                })

    return waves

# Detect swing points
df_15min_waves = find_swing_points(df_15min, order=3)

# Extract pivot points
swing_highs = df_15min_waves[df_15min_waves['swing_high'].notna()][['swing_high']].copy()
swing_highs['price'] = swing_highs['swing_high']
swing_highs['type'] = 'high'

swing_lows = df_15min_waves[df_15min_waves['swing_low'].notna()][['swing_low']].copy()
swing_lows['price'] = swing_lows['swing_low']
swing_lows['type'] = 'low'

pivots = pd.concat([swing_highs[['price', 'type']], swing_lows[['price', 'type']]])
pivots = pivots.sort_index()

print(f"Found {len(pivots)} pivot points")

# Find Elliott Wave patterns
bullish_waves = find_impulse_waves(pivots, direction='up')
bearish_waves = find_impulse_waves(pivots, direction='down')

all_waves = bullish_waves + bearish_waves

print(f"Found {len(bullish_waves)} bullish impulse waves")
print(f"Found {len(bearish_waves)} bearish impulse waves")

# ==========================================
# VISUALIZATION
# ==========================================
print("\nCreating Elliott Wave visualization...")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        'SPY 15-Minute Chart with Elliott Wave Structure',
        'All Detected Pivot Points'
    ),
    row_heights=[0.7, 0.3]
)

# Row 1: Main chart with candlesticks
fig.add_trace(
    go.Candlestick(
        x=df_15min_waves.index,
        open=df_15min_waves['open'],
        high=df_15min_waves['high'],
        low=df_15min_waves['low'],
        close=df_15min_waves['close'],
        name='SPY 15-Min',
        increasing_line_color='green',
        decreasing_line_color='red'
    ),
    row=1, col=1
)

# Add all swing points (for reference in row 2)
fig.add_trace(
    go.Scatter(
        x=df_15min_waves[df_15min_waves['swing_high'].notna()].index,
        y=df_15min_waves[df_15min_waves['swing_high'].notna()]['swing_high'],
        mode='markers',
        marker=dict(color='red', size=6, symbol='triangle-down'),
        name='Swing Highs',
        showlegend=False
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_15min_waves[df_15min_waves['swing_low'].notna()].index,
        y=df_15min_waves[df_15min_waves['swing_low'].notna()]['swing_low'],
        mode='markers',
        marker=dict(color='green', size=6, symbol='triangle-up'),
        name='Swing Lows',
        showlegend=False
    ),
    row=2, col=1
)

# Plot Elliott Wave patterns
wave_colors = {
    'bullish': 'lime',
    'bearish': 'red'
}

# Focus on most recent waves (last 3 of each type)
recent_bullish = bullish_waves[-3:] if len(bullish_waves) >= 3 else bullish_waves
recent_bearish = bearish_waves[-3:] if len(bearish_waves) >= 3 else bearish_waves

for waves, label_prefix in [(recent_bullish, 'Bullish'), (recent_bearish, 'Bearish')]:
    for idx, wave in enumerate(waves):
        direction = wave['direction']
        pivots_times = wave['pivots']
        prices = wave['prices']
        wave_labels = wave['wave_labels']

        color = wave_colors[direction]

        # Draw lines connecting the wave points
        fig.add_trace(
            go.Scatter(
                x=pivots_times,
                y=prices,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=10, color=color, symbol='circle', line=dict(width=2, color='white')),
                name=f'{label_prefix} Wave {idx+1}',
                showlegend=True,
                hovertext=[f'Wave {label}' for label in wave_labels],
                hoverinfo='text+y'
            ),
            row=1, col=1
        )

        # Add wave labels as annotations
        for i, (time, price, label) in enumerate(zip(pivots_times, prices, wave_labels)):
            if label != '0':  # Don't label the starting point
                fig.add_annotation(
                    x=time,
                    y=price,
                    text=f'<b>{label}</b>',
                    showarrow=False,
                    font=dict(size=14, color='white'),
                    bgcolor=color,
                    bordercolor='white',
                    borderwidth=2,
                    borderpad=4,
                    row=1, col=1
                )

        # Add wave direction label
        mid_time = pivots_times[3]  # Around wave 3
        mid_price = (max(prices) + min(prices)) / 2

        fig.add_annotation(
            x=mid_time,
            y=mid_price,
            text=f'{direction.upper()}<br>IMPULSE',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            ax=50,
            ay=-50,
            font=dict(size=12, color=color),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=color,
            borderwidth=2,
            row=1, col=1
        )

# Add price line to row 2
fig.add_trace(
    go.Scatter(
        x=df_15min_waves.index,
        y=df_15min_waves['close'],
        mode='lines',
        line=dict(color='blue', width=1),
        name='Close Price',
        showlegend=False
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    title={
        'text': f'Elliott Wave Analysis - SPY 15-Minute Chart<br>' +
                f'<sub>Detected {len(bullish_waves)} Bullish and {len(bearish_waves)} Bearish 5-Wave Impulse Patterns</sub>',
        'x': 0.5,
        'xanchor': 'center'
    },
    height=1000,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    template='plotly_dark',
    legend=dict(
        x=0.01,
        y=0.99,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(0, 0, 0, 0.5)'
    )
)

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Price ($)", row=2, col=1)

# Save
fig.write_html('elliott_wave_structure.html')
fig.show()

print(f"\nVisualization saved: elliott_wave_structure.html")

# Print wave details
print("\n" + "="*70)
print("DETECTED ELLIOTT WAVE PATTERNS")
print("="*70)

if len(bullish_waves) > 0:
    print(f"\nBULLISH IMPULSE WAVES: {len(bullish_waves)}")
    for i, wave in enumerate(bullish_waves[-3:]):  # Show last 3
        print(f"\n  Wave {i+1}:")
        print(f"    Start: {wave['start_idx'].strftime('%Y-%m-%d %H:%M')} @ ${wave['prices'][0]:.2f}")
        print(f"    End:   {wave['end_idx'].strftime('%Y-%m-%d %H:%M')} @ ${wave['prices'][-1]:.2f}")
        print(f"    Gain:  {(wave['prices'][-1] - wave['prices'][0]):.2f} ({(wave['prices'][-1]/wave['prices'][0]-1)*100:.2f}%)")

if len(bearish_waves) > 0:
    print(f"\nBEARISH IMPULSE WAVES: {len(bearish_waves)}")
    for i, wave in enumerate(bearish_waves[-3:]):  # Show last 3
        print(f"\n  Wave {i+1}:")
        print(f"    Start: {wave['start_idx'].strftime('%Y-%m-%d %H:%M')} @ ${wave['prices'][0]:.2f}")
        print(f"    End:   {wave['end_idx'].strftime('%Y-%m-%d %H:%M')} @ ${wave['prices'][-1]:.2f}")
        print(f"    Drop:  {(wave['prices'][0] - wave['prices'][-1]):.2f} ({(1-wave['prices'][-1]/wave['prices'][0])*100:.2f}%)")

print("\n" + "="*70)
print("\nElliott Wave Rules Applied:")
print("  ✓ Wave 2 never retraces more than 100% of Wave 1")
print("  ✓ Wave 3 is never the shortest impulse wave")
print("  ✓ Wave 4 never overlaps Wave 1 price territory")
print("\nAll detected patterns meet these strict Elliott Wave criteria.")
print("="*70)
