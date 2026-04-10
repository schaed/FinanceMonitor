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
print("ENHANCED MULTI-TIMEFRAME MEAN REVERSION DECISION SYSTEM")
print("With Volume-Weighted Bars and Polynomial Fit Visualization")
print("="*80)

# ==========================================
# CONFIGURATION
# ==========================================

# Stock symbol to analyze
SYMBOL = "SPY"

# Timeframe weights for signal aggregation
# Lower weights for short timeframes (more noise)
# Higher weights for longer timeframes (cleaner trends)
TIMEFRAME_WEIGHTS = {
    '1-min': 0.25,   # Low weight - high noise, use mainly for entry timing
    '15-min': 0.25,  # Low weight - still noisy from intraday moves
    '1-hour': 3.0,   # High weight - proven best performer (100% win rate)
    'daily': 4.0     # Highest weight - defines overall trend
}

# Volume-weighted bar aggregation settings
# Combine neighboring bars for the smallest volume_percentile% of bars
VOLUME_PERCENTILE = 10  # 10% = bottom decile of volume bars

print(f"\nConfiguration ({SYMBOL}):")
print("  Timeframe Weights:")
for tf, weight in TIMEFRAME_WEIGHTS.items():
    print(f"    {tf:8s}: {weight:.2f}x")
print(f"  Volume Aggregation: Bottom {VOLUME_PERCENTILE}% of bars by volume")
print()

# ==========================================
# VOLUME-WEIGHTED BAR AGGREGATION
# ==========================================

def aggregate_low_volume_bars(df, volume_percentile=VOLUME_PERCENTILE):
    """
    Combine neighboring bars for the smallest volume_percentile% of bars by volume

    Args:
        df: DataFrame with OHLCV data
        volume_percentile: Percentile threshold for low volume (default 10%)

    Returns:
        DataFrame with volume-weighted aggregated bars
    """
    print(f"   Original bars: {len(df)}")

    df = df.copy()
    df['original_volume'] = df['volume']

    # Calculate volume threshold
    volume_threshold = np.percentile(df['volume'], volume_percentile)
    print(f"   Volume threshold ({volume_percentile}th percentile): {volume_threshold:.0f}")

    # Mark low volume bars
    df['low_volume'] = df['volume'] < volume_threshold

    aggregated_bars = []
    i = 0

    while i < len(df):
        current_bar = df.iloc[i]

        if current_bar['low_volume'] and i < len(df) - 1:
            # Find consecutive low volume bars or next normal volume bar
            combine_indices = [i]
            j = i + 1

            # Combine with next bar(s) until we hit a normal volume bar
            while j < len(df) and df.iloc[j]['low_volume']:
                combine_indices.append(j)
                j += 1

            # Also include the next normal volume bar to create a meaningful aggregate
            if j < len(df):
                combine_indices.append(j)

            # Aggregate these bars using volume-weighted average
            bars_to_combine = df.iloc[combine_indices]
            total_volume = bars_to_combine['volume'].sum()

            if total_volume > 0:
                # Volume-weighted close
                vwap = (bars_to_combine['close'] * bars_to_combine['volume']).sum() / total_volume

                aggregated_bar = {
                    'open': bars_to_combine.iloc[0]['open'],
                    'high': bars_to_combine['high'].max(),
                    'low': bars_to_combine['low'].min(),
                    'close': vwap,  # Use VWAP as close
                    'volume': total_volume,
                    'original_volume': bars_to_combine['original_volume'].sum(),
                    'bars_combined': len(combine_indices)
                }
                aggregated_bars.append(aggregated_bar)
                i = j + 1
            else:
                i += 1
        else:
            # Keep normal volume bar as is
            aggregated_bars.append({
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_bar['close'],
                'volume': current_bar['volume'],
                'original_volume': current_bar['original_volume'],
                'bars_combined': 1
            })
            i += 1

    # Create DataFrame with original time index (use first bar's timestamp from each group)
    result_df = pd.DataFrame(aggregated_bars)

    # Use a simple range index since we've aggregated time periods
    result_df.index = df.index[:len(result_df)]

    combined_count = result_df[result_df['bars_combined'] > 1].shape[0]
    print(f"   Aggregated bars: {len(result_df)} (combined {combined_count} groups)")

    return result_df

# ==========================================
# FETCH DATA FOR ALL TIMEFRAMES
# ==========================================
print("\n1. Fetching multi-timeframe data...")

end_date = "2026-04-10"
end_dt = datetime.strptime(end_date, "%Y-%m-%d")

# Fetch all timeframes
start_1min = (end_dt - timedelta(days=2)).strftime("%Y-%m-%d")
df_1min_raw = api.get_bars(symbol=SYMBOL, start=start_1min, end=end_date, timeframe="1Min").df

start_15min = (end_dt - timedelta(days=7)).strftime("%Y-%m-%d")
df_15min_raw = api.get_bars(symbol=SYMBOL, start=start_15min, end=end_date, timeframe="15Min").df

start_1hour = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
df_1hour = api.get_bars(symbol=SYMBOL, start=start_1hour, end=end_date, timeframe="1Hour").df

start_daily = (end_dt - timedelta(days=365)).strftime("%Y-%m-%d")
df_daily = api.get_bars(symbol=SYMBOL, start=start_daily, end=end_date, timeframe="1Day").df

print(f"\n   Raw 1-min bars: {len(df_1min_raw)}")
print(f"   Raw 15-min bars: {len(df_15min_raw)}")
print(f"   1-hour bars: {len(df_1hour)}")
print(f"   Daily bars: {len(df_daily)}")

# Apply volume-weighted aggregation to 1-min and 15-min
print("\n2. Aggregating low-volume bars...")
print("\n   1-minute timeframe:")
df_1min = aggregate_low_volume_bars(df_1min_raw, volume_percentile=10)

print("\n   15-minute timeframe:")
df_15min = aggregate_low_volume_bars(df_15min_raw, volume_percentile=10)

# ==========================================
# CALCULATE POLYNOMIAL BANDS
# ==========================================

def calculate_polynomial_bands(df, window_size, poly_degree=2):
    """Calculate polynomial regression bands with confidence intervals"""
    df = df.copy()

    df['poly_mean'] = np.nan
    df['poly_std'] = np.nan
    df['std_distance'] = np.nan
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
            current_price = y[-1]

            df.iloc[i, df.columns.get_loc('poly_mean')] = current_poly_value
            df.iloc[i, df.columns.get_loc('poly_std')] = std_dev
            df.iloc[i, df.columns.get_loc('upper_1std')] = current_poly_value + std_dev
            df.iloc[i, df.columns.get_loc('lower_1std')] = current_poly_value - std_dev
            df.iloc[i, df.columns.get_loc('upper_2std')] = current_poly_value + 2 * std_dev
            df.iloc[i, df.columns.get_loc('lower_2std')] = current_poly_value - 2 * std_dev
            df.iloc[i, df.columns.get_loc('upper_3std')] = current_poly_value + 3 * std_dev
            df.iloc[i, df.columns.get_loc('lower_3std')] = current_poly_value - 3 * std_dev

            if std_dev > 0:
                distance = current_price - current_poly_value
                df.iloc[i, df.columns.get_loc('std_distance')] = distance / std_dev
        except:
            continue

    return df

print("\n3. Calculating polynomial bands for all timeframes...")

df_1min = calculate_polynomial_bands(df_1min, len(df_1min)//2)
df_15min = calculate_polynomial_bands(df_15min, len(df_15min)//2)
df_1hour = calculate_polynomial_bands(df_1hour, len(df_1hour)//2)
df_daily = calculate_polynomial_bands(df_daily, len(df_daily)//2)

print("   Bands calculated for all timeframes")

# ==========================================
# MULTI-TIMEFRAME SIGNAL AGGREGATION
# ==========================================

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

print("\n4. Aggregating signals from all timeframes...")

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
    """Combine signals from multiple timeframes into a single decision"""
    # Use configured timeframe weights
    weights = TIMEFRAME_WEIGHTS

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
        'weighted_score': 0,
        'reasoning': []
    }

    if long_score > short_score and long_score > 0:
        decision['action'] = 'long'
        decision['weighted_score'] = long_score
        decision['timeframes_aligned'] = long_count
        base_size = min(long_score / 10, 5.0)
        decision['position_size'] = base_size

        decision['reasoning'].append(f"Long score: {long_score:.1f} vs Short: {short_score:.1f}")
        decision['reasoning'].append(f"{long_count}/4 timeframes signal long")

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
# VISUALIZATION: POLYNOMIAL FITS WITH CONFIDENCE INTERVALS
# ==========================================
print("\n5. Creating enhanced visualization with polynomial fits...")

fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        '1-Minute Chart with Polynomial Fit',
        'Trading Decision Summary',
        '15-Minute Chart with Polynomial Fit',
        'Timeframe Alignment',
        '1-Hour Chart with Polynomial Fit',
        'Signal Strength',
        'Daily Chart with Polynomial Fit',
        'Volume Profile (1-min & 15-min)'
    ),
    specs=[
        [{"type": "xy"}, {"type": "table"}],
        [{"type": "xy"}, {"type": "bar"}],
        [{"type": "xy"}, {"type": "bar"}],
        [{"type": "xy"}, {"type": "bar"}]
    ],
    row_heights=[0.25, 0.25, 0.25, 0.25],
    vertical_spacing=0.08,
    horizontal_spacing=0.12
)

# Helper function to plot polynomial fit with confidence intervals
def plot_polynomial_fit(df, row, col, title, lookback=100):
    """Plot price with polynomial mean and confidence bands"""
    df_plot = df.tail(lookback)

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['close'],
            mode='lines',
            line=dict(color='cyan', width=1),
            name=f'{title} Price'
        ),
        row=row, col=col
    )

    # Polynomial mean
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['poly_mean'],
            mode='lines',
            line=dict(color='white', width=2),
            name='Polynomial Mean'
        ),
        row=row, col=col
    )

    # 1 std band
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['upper_1std'],
            mode='lines',
            line=dict(color='yellow', width=1, dash='dot'),
            name='+1σ',
            showlegend=False
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['lower_1std'],
            mode='lines',
            line=dict(color='yellow', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255, 255, 0, 0.1)',
            name='-1σ',
            showlegend=False
        ),
        row=row, col=col
    )

    # 2 std band (entry threshold)
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['upper_2std'],
            mode='lines',
            line=dict(color='orange', width=1.5, dash='dash'),
            name='+2σ (SHORT)',
            showlegend=False
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['lower_2std'],
            mode='lines',
            line=dict(color='lime', width=1.5, dash='dash'),
            name='-2σ (LONG)',
            showlegend=False
        ),
        row=row, col=col
    )

    # 3 std band
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['upper_3std'],
            mode='lines',
            line=dict(color='red', width=1, dash='dot'),
            name='+3σ',
            showlegend=False
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['lower_3std'],
            mode='lines',
            line=dict(color='red', width=1, dash='dot'),
            name='-3σ',
            showlegend=False
        ),
        row=row, col=col
    )

    # Mark current point
    current_price = df_plot.iloc[-1]['close']
    current_mean = df_plot.iloc[-1]['poly_mean']

    fig.add_trace(
        go.Scatter(
            x=[df_plot.index[-1]],
            y=[current_price],
            mode='markers',
            marker=dict(size=12, color='yellow', symbol='star', line=dict(width=2, color='black')),
            name='Current',
            showlegend=False
        ),
        row=row, col=col
    )

# Plot polynomial fits for each timeframe
plot_polynomial_fit(df_1min, 1, 1, '1-min', lookback=150)
plot_polynomial_fit(df_15min, 2, 1, '15-min', lookback=100)
plot_polynomial_fit(df_1hour, 3, 1, '1-hour', lookback=100)
plot_polynomial_fit(df_daily, 4, 1, 'Daily', lookback=100)

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

# Row 4 Col 2: Volume profile comparison
volume_data = {
    '1-min (raw)': df_1min_raw['volume'].mean(),
    '1-min (agg)': df_1min['volume'].mean(),
    '15-min (raw)': df_15min_raw['volume'].mean(),
    '15-min (agg)': df_15min['volume'].mean()
}

fig.add_trace(
    go.Bar(
        y=list(volume_data.keys()),
        x=list(volume_data.values()),
        orientation='h',
        marker_color=['cyan', 'blue', 'orange', 'red'],
        text=[f"{v:.0f}" for v in volume_data.values()],
        textposition='auto',
        showlegend=False
    ),
    row=4, col=2
)

# Layout
fig.update_layout(
    title={
        'text': f'Enhanced Multi-Timeframe Analysis with Polynomial Fits<br>' +
                f'<sub>SPY | Decision: {decision["action"].upper()} @ {decision["position_size"]:.1f}x | ' +
                f'Confidence: {decision["confidence"].upper()} | Volume-Weighted Bars</sub>',
        'x': 0.5,
        'xanchor': 'center'
    },
    height=1600,
    showlegend=False,
    template='plotly_dark'
)

# Update axes labels
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Price ($)", row=2, col=1)
fig.update_yaxes(title_text="Price ($)", row=3, col=1)
fig.update_yaxes(title_text="Price ($)", row=4, col=1)

fig.update_xaxes(title_text="Avg Volume", row=4, col=2)

fig.write_html('multi_timeframe_enhanced.html')
fig.show()

print("\n   Saved: multi_timeframe_enhanced.html")

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
    'daily_std': signals['daily']['std_distance'],
    '1min_bars_orig': len(df_1min_raw),
    '1min_bars_agg': len(df_1min),
    '15min_bars_orig': len(df_15min_raw),
    '15min_bars_agg': len(df_15min)
}])

decision_log.to_csv('multi_timeframe_decision_enhanced.csv', index=False)
print("\n   Saved: multi_timeframe_decision_enhanced.csv")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nFinal Decision: {decision['action'].upper()} with {decision['confidence'].upper()} confidence")
print(f"Suggested position size: {decision['position_size']:.2f}x")
print(f"\nVolume Aggregation Results:")
print(f"  1-min: {len(df_1min_raw)} bars → {len(df_1min)} bars ({100*(1-len(df_1min)/len(df_1min_raw)):.1f}% reduction)")
print(f"  15-min: {len(df_15min_raw)} bars → {len(df_15min)} bars ({100*(1-len(df_15min)/len(df_15min_raw)):.1f}% reduction)")
