import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Loading results...")

# Load trades
trades_df = pd.read_csv('fourier_trades.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print(f"Loaded {len(trades_df)} trades")

# Create comprehensive visualization
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        'Total P&L: Long vs Short',
        'Win Rate Comparison',
        'Holding Time Distribution',
        'Exit Reasons',
        'Cumulative P&L by Position Type',
        'P&L Distribution',
        'Strategy Summary',
        ''
    ),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "histogram"}, {"type": "bar"}],
        [{"colspan": 2}, None],
        [{"type": "table"}, {"type": "xy"}]
    ],
    row_heights=[0.25, 0.25, 0.25, 0.25],
    vertical_spacing=0.12,
    horizontal_spacing=0.15
)

# Separate long and short
long_trades = trades_df[trades_df['position_type'] == 'long']
short_trades = trades_df[trades_df['position_type'] == 'short']

# Row 1 Col 1: Total P&L
fig.add_trace(
    go.Bar(
        x=['Long', 'Short', 'Total'],
        y=[
            long_trades['net_pnl'].sum() * 100,
            short_trades['net_pnl'].sum() * 100,
            trades_df['net_pnl'].sum() * 100
        ],
        marker_color=['green', 'red', 'blue'],
        text=[
            f"{long_trades['net_pnl'].sum()*100:.2f}%",
            f"{short_trades['net_pnl'].sum()*100:.2f}%",
            f"{trades_df['net_pnl'].sum()*100:.2f}%"
        ],
        textposition='outside',
        showlegend=False
    ),
    row=1, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)

# Row 1 Col 2: Win rates
fig.add_trace(
    go.Bar(
        x=['Long', 'Short', 'Overall'],
        y=[
            (long_trades['net_pnl'] > 0).sum() / len(long_trades) * 100,
            (short_trades['net_pnl'] > 0).sum() / len(short_trades) * 100,
            (trades_df['net_pnl'] > 0).sum() / len(trades_df) * 100
        ],
        marker_color=['green', 'red', 'blue'],
        text=[
            f"{(long_trades['net_pnl'] > 0).sum() / len(long_trades) * 100:.1f}%",
            f"{(short_trades['net_pnl'] > 0).sum() / len(short_trades) * 100:.1f}%",
            f"{(trades_df['net_pnl'] > 0).sum() / len(trades_df) * 100:.1f}%"
        ],
        textposition='outside',
        showlegend=False
    ),
    row=1, col=2
)

# Row 2 Col 1: Holding time distribution
fig.add_trace(
    go.Histogram(
        x=long_trades['bars_held'],
        name='Long',
        marker_color='green',
        opacity=0.7,
        nbinsx=20
    ),
    row=2, col=1
)
fig.add_trace(
    go.Histogram(
        x=short_trades['bars_held'],
        name='Short',
        marker_color='red',
        opacity=0.7,
        nbinsx=20
    ),
    row=2, col=1
)

# Row 2 Col 2: Exit reasons
exit_counts = trades_df['exit_reason'].value_counts()
fig.add_trace(
    go.Bar(
        x=exit_counts.index,
        y=exit_counts.values,
        marker_color='orange',
        text=exit_counts.values,
        textposition='auto',
        showlegend=False
    ),
    row=2, col=2
)

# Row 3: Cumulative P&L
long_cumsum = long_trades.sort_values('entry_time')['net_pnl'].cumsum() * 100
short_cumsum = short_trades.sort_values('entry_time')['net_pnl'].cumsum() * 100
total_cumsum = trades_df.sort_values('entry_time')['net_pnl'].cumsum() * 100

fig.add_trace(
    go.Scatter(
        x=list(range(len(long_trades))),
        y=long_cumsum.values,
        mode='lines',
        name='Long Cumulative',
        line=dict(color='green', width=2)
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=list(range(len(short_trades))),
        y=short_cumsum.values,
        mode='lines',
        name='Short Cumulative',
        line=dict(color='red', width=2)
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=list(range(len(trades_df))),
        y=total_cumsum.values,
        mode='lines',
        name='Total',
        line=dict(color='blue', width=3)
    ),
    row=3, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

# Row 4 Col 1: Summary table
winning = (trades_df['net_pnl'] > 0).sum()
losing = (trades_df['net_pnl'] < 0).sum()
avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()
avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean())

table_data = [
    ['Total Positions', len(trades_df)],
    ['Long Positions', len(long_trades)],
    ['Short Positions', len(short_trades)],
    ['Win Rate', f"{winning/len(trades_df)*100:.1f}%"],
    ['Total P&L', f"{trades_df['net_pnl'].sum()*100:.2f}%"],
    ['Avg Win', f"{avg_win*100:.2f}%"],
    ['Avg Loss', f"{avg_loss*100:.2f}%"],
    ['Profit Factor', f"{(avg_win * winning) / (avg_loss * losing):.2f}"],
    ['Avg Hold (bars)', f"{trades_df['bars_held'].mean():.1f}"],
    ['Fourier Hold Target', '617-1235 bars']
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

# Row 4 Col 2: P&L distribution box plot
fig.add_trace(
    go.Box(
        y=long_trades['net_pnl'] * 100,
        name='Long',
        marker_color='green',
        boxmean='sd'
    ),
    row=4, col=2
)

fig.add_trace(
    go.Box(
        y=short_trades['net_pnl'] * 100,
        name='Short',
        marker_color='red',
        boxmean='sd'
    ),
    row=4, col=2
)

# Layout
fig.update_layout(
    title={
        'text': 'Fourier-Informed Trading Strategy Results<br>' +
                '<sub>Holding Period: 617-1235 minutes (1.6-3.2 days) based on 6.33-day cycle</sub>',
        'x': 0.5,
        'xanchor': 'center'
    },
    height=1400,
    showlegend=True,
    template='plotly_white'
)

fig.update_yaxes(title_text="P&L %", row=1, col=1)
fig.update_yaxes(title_text="Win Rate %", row=1, col=2)
fig.update_xaxes(title_text="Bars Held", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=2)
fig.update_yaxes(title_text="Cumulative P&L %", row=3, col=1)
fig.update_xaxes(title_text="Trade Number", row=3, col=1)
fig.update_yaxes(title_text="P&L %", row=4, col=2)

fig.write_html('fourier_strategy_results.html')
print("\nSaved: fourier_strategy_results.html")

# Print summary
print("\n" + "="*60)
print("FOURIER STRATEGY SUMMARY")
print("="*60)
print(f"\nFourier Analysis Found:")
print(f"  - Daily timeframe: 6.33-day cycle")
print(f"  - Suggested hold: 617-1235 minutes (1.6-3.2 days)")
print(f"\nActual Trading Results:")
print(f"  - Total positions: {len(trades_df)}")
print(f"  - Win rate: {winning/len(trades_df)*100:.1f}%")
print(f"  - Total P&L: {trades_df['net_pnl'].sum()*100:.2f}%")
print(f"  - Avg actual hold: {trades_df['bars_held'].mean():.1f} bars")
print(f"\nKey Issue:")
print(f"  - Fourier suggested holding 617-1235 bars")
print(f"  - Actually held only {trades_df['bars_held'].mean():.1f} bars (avg)")
print(f"  - Most exits (69%) were stop losses")
print(f"  - This means: 1-min trends don't last as long as daily cycles suggest")
print("\nConclusion:")
print("  Fourier analysis identifies multi-day cycles, but these are")
print("  too long for profitable 1-minute bar trading.")
print("  Recommendation: Use Fourier for position sizing or directional")
print("  bias, but exit based on shorter-term price action.")

print("\n" + "="*60)
