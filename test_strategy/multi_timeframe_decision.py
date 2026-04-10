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
print("MULTI-TIMEFRAME MEAN REVERSION DECISION SYSTEM")
print("="*80)

# ==========================================
# FETCH DATA FOR ALL TIMEFRAMES
# ==========================================
print("\n1. Fetching multi-timeframe data...")

end_date = "2026-04-08"
end_dt = datetime.strptime(end_date, "%Y-%m-%d")

# Fetch all timeframes
start_1min = (end_dt - timedelta(days=2)).strftime("%Y-%m-%d")
df_1min = api.get_bars(symbol="SPY", start=start_1min, end=end_date, timeframe="1Min").df
print('1min bars: ',len(df_1min))
start_15min = (end_dt - timedelta(days=7)).strftime("%Y-%m-%d")
df_15min = api.get_bars(symbol="SPY", start=start_15min, end=end_date, timeframe="15Min").df

start_1hour = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
df_1hour = api.get_bars(symbol="SPY", start=start_1hour, end=end_date, timeframe="1Hour").df

start_daily = (end_dt - timedelta(days=365)).strftime("%Y-%m-%d")
df_daily = api.get_bars(symbol="SPY", start=start_daily, end=end_date, timeframe="1Day").df

print(f"   1-min: {len(df_1min)} bars")
print(f"   15-min: {len(df_15min)} bars")
print(f"   1-hour: {len(df_1hour)} bars")
print(f"   Daily: {len(df_daily)} bars")

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

print("\n2. Calculating polynomial bands for all timeframes...")

df_1min = calculate_polynomial_bands(df_1min, len(df_1min)//2)
df_15min = calculate_polynomial_bands(df_15min, len(df_15min)//2)
df_1hour = calculate_polynomial_bands(df_1hour, len(df_1hour)//2)
df_daily = calculate_polynomial_bands(df_daily, len(df_daily)//2)

print("   Bands calculated for all timeframes")

# ==========================================
# MULTI-TIMEFRAME SIGNAL AGGREGATION
# ==========================================

def get_current_signal(df):
    """
    Get current signal from a timeframe

    Returns:
        dict with signal type, strength, and details
    """
    if len(df) == 0:
        return {'signal': 'neutral', 'strength': 0, 'std_distance': 0}

    latest = df.iloc[-1]

    if pd.isna(latest['std_distance']):
        return {'signal': 'neutral', 'strength': 0, 'std_distance': 0}

    std_dist = latest['std_distance']

    if std_dist <= -2.0:
        # Long signal
        strength = min(abs(std_dist), 5)  # Cap at 5
        return {
            'signal': 'long',
            'strength': strength,
            'std_distance': std_dist,
            'price': latest['close'],
            'poly_mean': latest['poly_mean']
        }
    elif std_dist >= 2.0:
        # Short signal
        strength = min(std_dist, 5)
        return {
            'signal': 'short',
            'strength': strength,
            'std_distance': std_dist,
            'price': latest['close'],
            'poly_mean': latest['poly_mean']
        }
    else:
        # Neutral (within ±2σ)
        return {
            'signal': 'neutral',
            'strength': abs(std_dist),
            'std_distance': std_dist,
            'price': latest['close'],
            'poly_mean': latest['poly_mean']
        }

print("\n3. Aggregating signals from all timeframes...")

signals = {
    '1-min': get_current_signal(df_1min),
    '15-min': get_current_signal(df_15min),
    '1-hour': get_current_signal(df_1hour),
    'daily': get_current_signal(df_daily)
}

print("\n   Current Signals:")
for tf, sig in signals.items():
    print(f"   {tf:8s}: {sig['signal']:7s} | {sig['std_distance']:+.2f}σ | strength: {sig['strength']:.2f}")

# ==========================================
# MULTI-TIMEFRAME DECISION LOGIC
# ==========================================

def make_trading_decision(signals):
    """
    Combine signals from multiple timeframes into a single decision

    Logic:
    - Weight longer timeframes more heavily
    - Require alignment for strong signals
    - Calculate aggregate position size

    Weights:
    - Daily: 4x (most important)
    - 1-hour: 3x
    - 15-min: 2x
    - 1-min: 1x (least important, execution timing)
    """

    weights = {
        '1-min': 1.0,
        '15-min': 2.0,
        '1-hour': 3.0,
        'daily': 4.0
    }

    # Calculate weighted scores
    long_score = 0
    short_score = 0

    for tf, sig in signals.items():
        weight = weights[tf]
        if sig['signal'] == 'long':
            long_score += sig['strength'] * weight
        elif sig['signal'] == 'short':
            short_score += sig['strength'] * weight

    # Count alignment
    long_count = sum(1 for sig in signals.values() if sig['signal'] == 'long')
    short_count = sum(1 for sig in signals.values() if sig['signal'] == 'short')

    # Decision logic
    decision = {
        'action': 'neutral',
        'confidence': 0,
        'position_size': 0,
        'timeframes_aligned': 0,
        'weighted_score': 0,
        'reasoning': []
    }

    # Determine action
    if long_score > short_score and long_score > 0:
        decision['action'] = 'long'
        decision['weighted_score'] = long_score
        decision['timeframes_aligned'] = long_count

        # Calculate position size (capped at 5x)
        base_size = min(long_score / 10, 5.0)  # Normalize to 0-5x
        decision['position_size'] = base_size

        # Reasoning
        decision['reasoning'].append(f"Long score: {long_score:.1f} vs Short: {short_score:.1f}")
        decision['reasoning'].append(f"{long_count}/4 timeframes signal long")

        # Confidence based on alignment
        if long_count >= 3:
            decision['confidence'] = 'high'
            decision['reasoning'].append("HIGH confidence: 3+ timeframes aligned")
        elif long_count >= 2:
            decision['confidence'] = 'medium'
            decision['reasoning'].append("MEDIUM confidence: 2 timeframes aligned")
        else:
            decision['confidence'] = 'low'
            decision['reasoning'].append("LOW confidence: only 1 timeframe")

    elif short_score > long_score and short_score > 0:
        decision['action'] = 'short'
        decision['weighted_score'] = short_score
        decision['timeframes_aligned'] = short_count

        base_size = min(short_score / 10, 5.0)
        decision['position_size'] = base_size

        decision['reasoning'].append(f"Short score: {short_score:.1f} vs Long: {long_score:.1f}")
        decision['reasoning'].append(f"{short_count}/4 timeframes signal short")

        if short_count >= 3:
            decision['confidence'] = 'high'
            decision['reasoning'].append("HIGH confidence: 3+ timeframes aligned")
        elif short_count >= 2:
            decision['confidence'] = 'medium'
            decision['reasoning'].append("MEDIUM confidence: 2 timeframes aligned")
        else:
            decision['confidence'] = 'low'
            decision['reasoning'].append("LOW confidence: only 1 timeframe")
    else:
        decision['action'] = 'neutral'
        decision['confidence'] = 'none'
        decision['reasoning'].append("No clear signal - stay out")

    return decision

decision = make_trading_decision(signals)

print("\n" + "="*80)
print("TRADING DECISION")
print("="*80)

print(f"\nAction: {decision['action'].upper()}")
print(f"Confidence: {decision['confidence'].upper()}")
print(f"Position Size: {decision['position_size']:.2f}x")
print(f"Weighted Score: {decision['weighted_score']:.1f}")
print(f"Timeframes Aligned: {decision['timeframes_aligned']}/4")

print(f"\nReasoning:")
for reason in decision['reasoning']:
    print(f"  - {reason}")

# ==========================================
# VISUALIZATION
# ==========================================
print("\n4. Creating multi-timeframe dashboard...")

fig = make_subplots(
    rows=5, cols=2,
    subplot_titles=(
        '1-Minute (Execution Timing)',
        'Trading Decision Summary',
        '15-Minute (Short-term)',
        'Timeframe Alignment',
        '1-Hour (Medium-term)',
        'Signal Strength',
        'Daily (Long-term Trend)',
        'Weighted Scores',
        'Multi-Timeframe Signals Overview',
        ''
    ),
    specs=[
        [{"type": "xy"}, {"type": "table"}],
        [{"type": "xy"}, {"type": "bar"}],
        [{"type": "xy"}, {"type": "bar"}],
        [{"type": "xy"}, {"type": "bar"}],
        [{"colspan": 2}, None]
    ],
    row_heights=[0.18, 0.18, 0.18, 0.18, 0.28],
    vertical_spacing=0.08,
    horizontal_spacing=0.12
)

# Helper function to plot std distance
def plot_std_distance(df, row, col, title):
    if 'std_distance' in df.columns and df['std_distance'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df.index[-200:],  # Last 100 bars
                y=df['std_distance'].iloc[-200:],
                mode='lines',
                line=dict(color='cyan', width=1),
                fill='tozeroy',
                name=title
            ),
            row=row, col=col
        )

        fig.add_hline(y=2, line_dash="dash", line_color="red", row=row, col=col)
        fig.add_hline(y=-2, line_dash="dash", line_color="lime", row=row, col=col)
        fig.add_hline(y=0, line_dash="solid", line_color="white", row=row, col=col)

        # Mark current point
        current_std = df['std_distance'].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[current_std],
                mode='markers',
                marker=dict(size=12, color='yellow', symbol='star'),
                name='Current',
                showlegend=False
            ),
            row=row, col=col
        )

# Plot each timeframe
plot_std_distance(df_1min, 1, 1, '1-min')
plot_std_distance(df_15min, 2, 1, '15-min')
plot_std_distance(df_1hour, 3, 1, '1-hour')
plot_std_distance(df_daily, 4, 1, 'Daily')

# Row 1 Col 2: Decision summary table
decision_data = [
    ['Action', decision['action'].upper()],
    ['Confidence', decision['confidence'].upper()],
    ['Position Size', f"{decision['position_size']:.2f}x"],
    ['Aligned TFs', f"{decision['timeframes_aligned']}/4"],
    ['Weighted Score', f"{decision['weighted_score']:.1f}"]
]

fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='darkblue',
            align='left',
            font=dict(size=12, color='white')
        ),
        cells=dict(
            values=list(zip(*decision_data)),
            fill_color=[
                ['lightblue']*5,
                ['yellow' if decision['action'] != 'neutral' else 'gray']*5
            ],
            align='left',
            font=dict(size=11),
            height=25
        )
    ),
    row=1, col=2
)

# Row 2 Col 2: Timeframe alignment
tf_names = list(signals.keys())
tf_signals = [signals[tf]['signal'] for tf in tf_names]
colors_alignment = ['lime' if s == 'long' else 'red' if s == 'short' else 'gray' for s in tf_signals]

fig.add_trace(
    go.Bar(
        y=tf_names,
        x=[1]*len(tf_names),
        orientation='h',
        marker_color=colors_alignment,
        text=[s.upper() for s in tf_signals],
        textposition='inside',
        showlegend=False
    ),
    row=2, col=2
)

# Row 3 Col 2: Signal strength
strengths = [signals[tf]['strength'] for tf in tf_names]
fig.add_trace(
    go.Bar(
        y=tf_names,
        x=strengths,
        orientation='h',
        marker_color='orange',
        text=[f"{s:.2f}" for s in strengths],
        textposition='auto',
        showlegend=False
    ),
    row=3, col=2
)

# Row 4 Col 2: Weighted contribution
weights = {'1-min': 1.0, '15-min': 2.0, '1-hour': 3.0, 'daily': 4.0}
weighted_contributions = []
for tf in tf_names:
    sig = signals[tf]
    weight = weights[tf]
    contrib = sig['strength'] * weight if sig['signal'] != 'neutral' else 0
    weighted_contributions.append(contrib)

fig.add_trace(
    go.Bar(
        y=tf_names,
        x=weighted_contributions,
        orientation='h',
        marker_color='purple',
        text=[f"{w:.1f}" for w in weighted_contributions],
        textposition='auto',
        showlegend=False
    ),
    row=4, col=2
)

# Row 5: Summary of all signals
summary_text = f"""
<b>MULTI-TIMEFRAME ANALYSIS SUMMARY</b>

<b>Current Signals:</b>
  • 1-Minute:  {signals['1-min']['signal']:8s} ({signals['1-min']['std_distance']:+.2f}σ)
  • 15-Minute: {signals['15-min']['signal']:8s} ({signals['15-min']['std_distance']:+.2f}σ)
  • 1-Hour:    {signals['1-hour']['signal']:8s} ({signals['1-hour']['std_distance']:+.2f}σ)
  • Daily:     {signals['daily']['signal']:8s} ({signals['daily']['std_distance']:+.2f}σ)

<b>Decision: {decision['action'].upper()}</b>
Confidence: {decision['confidence'].upper()}
Position Size: {decision['position_size']:.2f}x

<b>Strategy Notes:</b>
  • Entry threshold: ±2σ from polynomial mean
  • Position sizing: Scales with deviation (2σ=2x, 3σ=3x, etc.)
  • Exit trigger: Return to polynomial mean
  • Timeframe weights: Daily(4x) > 1-hour(3x) > 15-min(2x) > 1-min(1x)
"""

fig.add_annotation(
    xref="x domain",
    yref="y domain",
    x=0.5,
    y=0.5,
    text=summary_text,
    showarrow=False,
    font=dict(size=11, family='Courier New'),
    align='left',
    bgcolor='rgba(0, 0, 0, 0.8)',
    bordercolor='cyan',
    borderwidth=2,
    row=5, col=1
)

# Layout
fig.update_layout(
    title={
        'text': f'Multi-Timeframe Mean Reversion Decision System<br>' +
                f'<sub>SPY | Decision: {decision["action"].upper()} @ {decision["position_size"]:.1f}x | ' +
                f'Confidence: {decision["confidence"].upper()}</sub>',
        'x': 0.5,
        'xanchor': 'center'
    },
    height=1600,
    showlegend=False,
    template='plotly_dark'
)

# Update axes
for i in range(1, 5):
    fig.update_yaxes(title_text="Std Devs (σ)", row=i, col=1)

fig.update_xaxes(title_text="Timeframe", row=2, col=2)
fig.update_xaxes(title_text="Strength", row=3, col=2)
fig.update_xaxes(title_text="Weighted Score", row=4, col=2)

fig.write_html('multi_timeframe_decision.html')
fig.show()

print("\n   Saved: multi_timeframe_decision.html")

# ==========================================
# SAVE DECISION LOG
# ==========================================
decision_log = pd.DataFrame([{
    'timestamp': end_date,
    'action': decision['action'],
    'confidence': decision['confidence'],
    'position_size': decision['position_size'],
    'weighted_score': decision['weighted_score'],
    'timeframes_aligned': decision['timeframes_aligned'],
    '1min_signal': signals['1-min']['signal'],
    '1min_std': signals['1-min']['std_distance'],
    '15min_signal': signals['15-min']['signal'],
    '15min_std': signals['15-min']['std_distance'],
    '1hour_signal': signals['1-hour']['signal'],
    '1hour_std': signals['1-hour']['std_distance'],
    'daily_signal': signals['daily']['signal'],
    'daily_std': signals['daily']['std_distance']
}])

decision_log.to_csv('multi_timeframe_decision.csv', index=False)
print("\n   Saved: multi_timeframe_decision.csv")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nFinal Decision: {decision['action'].upper()} with {decision['confidence'].upper()} confidence")
print(f"Suggested position size: {decision['position_size']:.2f}x")
