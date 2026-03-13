from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BG_COLOR = "#05070d"
PLOT_BG = "#0b1020"
GRID_COLOR = "rgba(255,255,255,0.07)"
AXIS_COLOR = "rgba(255,255,255,0.28)"


def ema_smooth(values: pd.Series, weight: float) -> np.ndarray:
    if values.empty:
        return np.array([])
    smoothed = []
    last = float(values.iloc[0])
    for value in values:
        last = last * weight + (1.0 - weight) * float(value)
        smoothed.append(last)
    return np.array(smoothed)


def _rgba_from_hex(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _read_metric(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"Step", "Value"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{csv_path.name}: expected columns {expected}, found {set(df.columns)}")
    return df[["Step", "Value"]].dropna()


def _robust_range(values: pd.Series, lower: float = 1.0, upper: float = 99.0) -> list[float] | None:
    if values.empty:
        return None
    y = values.astype(float).to_numpy()
    lo = np.percentile(y, lower)
    hi = np.percentile(y, upper)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        delta = abs(lo) * 0.05 if lo != 0 else 1.0
        return [lo - delta, hi + delta]
    pad = (hi - lo) * 0.06
    return [lo - pad, hi + pad]


def plot_single_metric(
    metric_name: str,
    title: str,
    color: str,
    y_title: str = "Value",
    smoothing: float = 0.8,
    robust_range: bool = True,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    image_format: str = "png",
    image_scale: float = 2.0,
    show: bool = False,
) -> Path:
    script_dir = Path(__file__).resolve().parent
    resolved_input_dir = input_dir or (script_dir.parent / "csv")
    resolved_output_dir = output_dir or (script_dir / "plots")
    csv_path = resolved_input_dir / f"{metric_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Metric CSV not found: {csv_path}")
    if not (0 <= smoothing < 1):
        raise ValueError("smoothing must be in [0, 1).")
    if image_format not in {"png", "jpg", "jpeg", "webp", "svg", "pdf"}:
        raise ValueError("image_format must be one of: png, jpg, jpeg, webp, svg, pdf")

    df = _read_metric(csv_path)
    smooth_y = ema_smooth(df["Value"], smoothing)
    raw_color = _rgba_from_hex(color, 0.20)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Value"],
            mode="lines",
            name=f"{metric_name} (raw)",
            line=dict(color=raw_color, width=1.2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=smooth_y,
            mode="lines",
            name=metric_name,
            line=dict(color=color, width=2.6),
            hovertemplate="<b>%{fullData.name}</b><br>Step: %{x}<br>Value: %{y:.6g}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=17, color="rgba(255,255,255,0.92)", family="monospace"),
            x=0.0,
            xanchor="left",
            pad=dict(l=10, t=8),
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        height=500,
        width=1150,
        margin=dict(l=70, r=40, t=65, b=70),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(size=12, color="rgba(255,255,255,0.82)", family="monospace"),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111b36",
            font=dict(size=12, color="white", family="monospace"),
            bordercolor="rgba(255,255,255,0.22)",
        ),
        font=dict(family="monospace", color="rgba(255,255,255,0.74)"),
        annotations=[
            dict(
                text=f"smoothing = {smoothing}",
                xref="paper",
                yref="paper",
                x=1.0,
                y=-0.12,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.38)", family="monospace"),
            )
        ],
    )
    fig.update_xaxes(
        title_text="Step",
        title_font=dict(size=12, color="rgba(255,255,255,0.58)"),
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS_COLOR,
        tickfont=dict(size=11, color="rgba(255,255,255,0.56)"),
        showspikes=True,
        spikecolor="rgba(255,255,255,0.18)",
        spikethickness=1,
        spikedash="dot",
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=12, color="rgba(255,255,255,0.58)"),
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS_COLOR,
        tickfont=dict(size=11, color="rgba(255,255,255,0.56)"),
        showspikes=True,
        spikecolor="rgba(255,255,255,0.18)",
        spikethickness=1,
        spikedash="dot",
    )

    if robust_range:
        y_range = _robust_range(df["Value"])
        if y_range is not None:
            fig.update_yaxes(range=y_range)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / f"{metric_name}_tb.{image_format}"
    try:
        fig.write_image(str(output_path), format=image_format, scale=image_scale)
    except Exception as exc:
        raise RuntimeError(
            "Image export failed. Install kaleido with `pip install kaleido` and rerun."
        ) from exc

    if show:
        fig.show()

    return output_path