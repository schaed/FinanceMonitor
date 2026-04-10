import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Loading Elliott Wave trading results...")

trades_df = pd.read_csv('elliott_wave_trades.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print(f"Loaded {len(trades_df)} trades")

# Create visualization
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        'Elliott Wave Strategy Performance',
        'Win Rate by Position Type',
        'Holding Time Distribution',
        'Exit Reasons',
        'Cumulative P&L',
        'P&L Distribution',
        'Performance Metrics',
        'Strategy Comparison'
    ),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "histogram"}, {"type": "bar"}],
        [{"colspan": 2}, None],
        [{"type": "table"}, {"type": "table"}]
    ],
    row_heights=[0.25, 0.25, 0.25, 0.25],
    vertical_spacing=0.12,
    horizontal_spacing=0.15
)

# Row 1 Col 1: P&L
fig.add_trace(
    go.Bar(
        x=['Elliott Wave\nStrategy'],
        y=[trades_df['net_pnl'].sum() * 100],
        marker_color='red',
        text=[f"{trades_df['net_pnl'].sum()*100:.2f}%"],
        textposition='outside',
        showlegend=False
    ),
    row=1, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)

# Row 1 Col 2: Win rate
winning = (trades_df['net_pnl'] > 0).sum()
fig.add_trace(
    go.Bar(
        x=['Elliott Wave\nStrategy'],
        y=[winning / len(trades_df) * 100],
        marker_color='orange',
        text=[f"{winning/len(trades_df)*100:.1f}%"],
        textposition='outside',
        showlegend=False
    ),
    row=1, col=2
)

# Row 2 Col 1: Holding time
fig.add_trace(
    go.Histogram(
        x=trades_df['bars_held'],
        nbinsx=15,
        marker_color='purple',
        name='Hold Time'
    ),
    row=2, col=1
)

# Row 2 Col 2: Exit reasons
exit_counts = trades_df['exit_reason'].value_counts()
fig.add_trace(
    go.Bar(
        x=exit_counts.index,
        y=exit_counts.values,
        marker_color='teal',
        text=exit_counts.values,
        textposition='auto',
        showlegend=False
    ),
    row=2, col=2
)

# Row 3: Cumulative P&L
trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
fig.add_trace(
    go.Scatter(
        x=list(range(len(trades_df))),
        y=trades_df['cumulative_pnl'] * 100,
        mode='lines',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)',
        name='Cumulative P&L'
    ),
    row=3, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

# Row 4 Col 1: Metrics table
avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()
avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean())

table_data = [
    ['Total Trades', len(trades_df)],
    ['Wave Bias', 'BEARISH'],
    ['Long Positions', '0'],
    ['Short Positions', '70'],
    ['Win Rate', f"{winning/len(trades_df)*100:.1f}%"],
    ['Total P&L', f"{trades_df['net_pnl'].sum()*100:.2f}%"],
    ['Avg Win', f"{avg_win*100:.2f}%"],
    ['Avg Loss', f"{avg_loss*100:.2f}%"],
    ['Profit Factor', f"{(avg_win * winning) / (avg_loss * (len(trades_df)-winning)):.2f}"],
    ['Avg Hold', f"{trades_df['bars_held'].mean():.1f} bars"]
]

fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='lightgray',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=list(zip(*table_data)),
            fill_color='white',
            align='left',
            font=dict(size=11),
            height=25
        )
    ),
    row=4, col=1
)

# Row 4 Col 2: Comparison table
comparison_data = [
    ['Strategy', 'P&L', 'Win%', 'Note'],
    ['Simple (2 up)', '-14.32%', '18.4%', 'Baseline'],
    ['Long-only', '-0.19%', '26.9%', 'Best ✓'],
    ['Fourier', '-1.44%', '31.0%', 'Wrong timeframe'],
    ['Elliott Wave', '-1.57%', '20.0%', 'Bearish bias']
]

fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Strategy</b>', '<b>P&L</b>', '<b>Win%</b>', '<b>Note</b>'],
            fill_color='lightblue',
            align='left',
            font=dict(size=11, color='black')
        ),
        cells=dict(
            values=list(zip(*comparison_data[1:])),
            fill_color='white',
            align='left',
            font=dict(size=10),
            height=23
        )
    ),
    row=4, col=2
)

fig.update_layout(
    title={
        'text': 'Elliott Wave Strategy Results - SPY<br>' +
                '<sub>Detected bearish 5-wave impulse → traded short only → lost -1.57%</sub>',
        'x': 0.5,
        'xanchor': 'center'
    },
    height=1400,
    showlegend=False,
    template='plotly_white'
)

fig.update_yaxes(title_text="P&L %", row=1, col=1)
fig.update_yaxes(title_text="Win Rate %", row=1, col=2)
fig.update_xaxes(title_text="Bars Held", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=2)
fig.update_yaxes(title_text="Cumulative P&L %", row=3, col=1)
fig.update_xaxes(title_text="Trade Number", row=3, col=1)

fig.write_html('elliott_wave_results.html')
print("\nSaved: elliott_wave_results.html")

# Print summary
print("\n" + "="*70)
print("ELLIOTT WAVE STRATEGY SUMMARY")
print("="*70)

print(f"\nWave Analysis:")
print(f"  - Detected: BEARISH 5-wave impulse (15-min timeframe)")
print(f"  - Position: Post Wave 5 (correction expected)")
print(f"  - Trading bias: SHORT only")

print(f"\nTrading Results:")
print(f"  - Total trades: {len(trades_df)}")
print(f"  - All shorts: {len(trades_df)} (0 longs)")
print(f"  - Win rate: {winning/len(trades_df)*100:.1f}%")
print(f"  - Total P&L: {trades_df['net_pnl'].sum()*100:.2f}%")

print(f"\nWhy It Failed:")
print(f"  1. Elliott Wave correctly identified bearish structure")
print(f"  2. BUT SPY has long-term upward bias")
print(f"  3. Shorting into uptrend = fighting the trend")
print(f"  4. 85.7% exits were trend breaks (upward momentum)")

print(f"\nKey Insight:")
print(f"  Elliott Wave is better for TIMING entries within")
print(f"  the dominant trend, not fighting against it.")

print("\n" + "="*70)
