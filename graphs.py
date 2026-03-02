import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TensorBoard-style colors (orange, blue, green)
algorithms = ['grpo', 'reinforce', 'maxrl']
colors = ['#ff7043', '#42a5f5', '#66bb6a']  # TensorBoard-like orange, blue, green

# Smoothing factor (0 = no smoothing, 1 = max smoothing) — mirrors TensorBoard's EMA slider
SMOOTHING = 0.8

def ema_smooth(values, weight):
    """Exponential moving average — same algorithm TensorBoard uses."""
    smoothed = []
    last = float(values.iloc[0])
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)

# TensorBoard dark background
BG_COLOR   = "#1a1a2e"   # outer background
PLOT_BG    = "#16213e"   # plot area
GRID_COLOR = "rgba(255,255,255,0.06)"
AXIS_COLOR = "rgba(255,255,255,0.25)"

metric_types = ['acc_val', 'loss_epoch', 'loss_step']

for metric_type in metric_types:
    fig = go.Figure()

    # Collect all y values across algorithms to compute a tight shared y range
    all_y = []
    dataframes = {}
    for algo in algorithms:
        file_path = f'data/cifar100_{algo}_{metric_type}.csv'
        df = pd.read_csv(file_path, header=None, names=['x', 'y'], skiprows=1)
        dataframes[algo] = df
        all_y.extend(df['y'].tolist())

    all_y = np.array(all_y)
    if metric_type == 'loss_step':
        y_lo = np.percentile(all_y, 10)
        y_hi = np.percentile(all_y, 90)
        y_pad = (y_hi - y_lo) * 0.05
        y_range = [y_lo - y_pad, y_hi + y_pad]
    else:
        y_range = None  # let Plotly auto-scale with full data extent

    for i, algo in enumerate(algorithms):
        df = dataframes[algo]

        smoothed_y = ema_smooth(df['y'], SMOOTHING)
        raw_color  = colors[i]
        # semi-transparent version of the same color for raw data
        r, g, b = int(raw_color[1:3], 16), int(raw_color[3:5], 16), int(raw_color[5:7], 16)
        faint_color = f'rgba({r},{g},{b},0.18)'

        # --- Raw data: faint, no markers (TensorBoard "background" trace) ---
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='lines',
            name=f'{algo.upper()} (raw)',
            line=dict(color=faint_color, width=1.2),
            showlegend=False,
            hoverinfo='skip',
        ))

        # --- Smoothed line: bold foreground (TensorBoard "smoothed" trace) ---
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=smoothed_y,
            mode='lines',
            name=algo.upper(),
            line=dict(color=raw_color, width=2.2),
            hovertemplate=(
                f'<b>{algo.upper()}</b><br>'
                'Step: %{x}<br>'
                'Value: %{y:.4f}<extra></extra>'
            ),
        ))

    # Titles
    if metric_type == 'acc_val':
        title, x_title, y_title = "Validation Accuracy", "Epoch", "Accuracy"
    elif metric_type == 'loss_epoch':
        title, x_title, y_title = "Loss / Epoch", "Epoch", "Loss"
    else:
        title, x_title, y_title = "Loss / Step", "Step", "Loss"

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(size=15, color='rgba(255,255,255,0.85)', family='monospace'),
            x=0.0,
            xanchor='left',
            pad=dict(l=10, t=8),
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        height=480,
        width=1100,
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.12)',
            borderwidth=1,
            font=dict(size=12, color='rgba(255,255,255,0.75)', family='monospace'),
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#0f3460',
            font=dict(size=12, color='white', family='monospace'),
            bordercolor='rgba(255,255,255,0.2)',
        ),
        font=dict(family='monospace', color='rgba(255,255,255,0.7)'),
        # Subtle "smoothing = X" annotation mimicking TensorBoard's UI label
        annotations=[dict(
            text=f'smoothing = {SMOOTHING}',
            xref='paper', yref='paper',
            x=1.0, y=-0.12,
            xanchor='right', yanchor='top',
            showarrow=False,
            font=dict(size=10, color='rgba(255,255,255,0.35)', family='monospace'),
        )],
    )

    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(size=12, color='rgba(255,255,255,0.5)'),
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS_COLOR,
        tickfont=dict(size=11, color='rgba(255,255,255,0.5)'),
        showspikes=True,
        spikecolor='rgba(255,255,255,0.15)',
        spikethickness=1,
        spikedash='dot',
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=12, color='rgba(255,255,255,0.5)'),
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS_COLOR,
        tickfont=dict(size=11, color='rgba(255,255,255,0.5)'),
        showspikes=True,
        spikecolor='rgba(255,255,255,0.15)',
        spikethickness=1,
        spikedash='dot',
        range=y_range,
    )

    fig.show()