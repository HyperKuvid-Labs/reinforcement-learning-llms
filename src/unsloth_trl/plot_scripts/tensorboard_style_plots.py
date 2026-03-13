from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BG_COLOR = "#1a1a2e"
PLOT_BG = "#16213e"
GRID_COLOR = "rgba(255,255,255,0.06)"
AXIS_COLOR = "rgba(255,255,255,0.25)"

SMOOTH_COLOR = "#42a5f5"

METRIC_TITLES = {
    "accuracy_r_mean": "Accuracy (r_mean)",
    "clipped_ratio": "Clipped Ratio",
    "completion_length": "Completion Length",
    "forward_r_mean": "Forward Reward (r_mean)",
    "grad_norm": "Gradient Norm",
    "kl": "KL Divergence",
    "loss": "Loss",
    "lr": "Learning Rate",
    "reward": "Reward",
    "reward_std": "Reward Std",
}


def ema_smooth(values: pd.Series, weight: float) -> np.ndarray:
    if values.empty:
        return np.array([])
    smoothed = []
    last = float(values.iloc[0])
    for value in values:
        last = last * weight + (1.0 - weight) * float(value)
        smoothed.append(last)
    return np.array(smoothed)


def read_metric_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_columns = {"Step", "Value"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"{csv_path.name}: expected columns {expected_columns}, got {set(df.columns)}")
    return df[["Step", "Value"]].dropna()


def robust_y_range(values: pd.Series, lower: float, upper: float) -> list[float] | None:
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


def style_figure(fig: go.Figure, title: str, smoothing: float) -> None:
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="rgba(255,255,255,0.85)", family="monospace"),
            x=0.0,
            xanchor="left",
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
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(size=12, color="rgba(255,255,255,0.75)", family="monospace"),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0f3460",
            font=dict(size=12, color="white", family="monospace"),
            bordercolor="rgba(255,255,255,0.2)",
        ),
        font=dict(family="monospace", color="rgba(255,255,255,0.7)"),
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
                font=dict(size=10, color="rgba(255,255,255,0.35)", family="monospace"),
            )
        ],
    )
    fig.update_xaxes(
        title_text="Step",
        title_font=dict(size=12, color="rgba(255,255,255,0.5)"),
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS_COLOR,
        tickfont=dict(size=11, color="rgba(255,255,255,0.5)"),
        showspikes=True,
        spikecolor="rgba(255,255,255,0.15)",
        spikethickness=1,
        spikedash="dot",
    )
    fig.update_yaxes(
        title_text="Value",
        title_font=dict(size=12, color="rgba(255,255,255,0.5)"),
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS_COLOR,
        tickfont=dict(size=11, color="rgba(255,255,255,0.5)"),
        showspikes=True,
        spikecolor="rgba(255,255,255,0.15)",
        spikethickness=1,
        spikedash="dot",
    )


def add_raw_and_smoothed(fig: go.Figure, df: pd.DataFrame, metric_name: str, smoothing: float) -> None:
    smoothed = ema_smooth(df["Value"], smoothing)
    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Value"],
            mode="lines",
            name=f"{metric_name} (raw)",
            line=dict(color="rgba(66,165,245,0.18)", width=1.2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=smoothed,
            mode="lines",
            name=metric_name,
            line=dict(color=SMOOTH_COLOR, width=2.2),
            hovertemplate="<b>%{fullData.name}</b><br>Step: %{x}<br>Value: %{y:.6g}<extra></extra>",
        )
    )


def build_separate_plots(
    input_dir: Path,
    output_dir: Path,
    smoothing: float,
    robust_range: bool,
    lower_percentile: float,
    upper_percentile: float,
    show: bool,
) -> None:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        metric_name = csv_path.stem
        title = METRIC_TITLES.get(metric_name, metric_name.replace("_", " ").title())
        df = read_metric_csv(csv_path)

        fig = go.Figure()
        add_raw_and_smoothed(fig, df, metric_name, smoothing)
        style_figure(fig, title, smoothing)

        if robust_range:
            y_range = robust_y_range(df["Value"], lower_percentile, upper_percentile)
            if y_range is not None:
                fig.update_yaxes(range=y_range)

        out_html = output_dir / f"{metric_name}_tb.html"
        fig.write_html(str(out_html), include_plotlyjs="cdn")

        if show:
            fig.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TensorBoard-style plots from unsloth TRL CSV metrics.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "csv",
        help="Directory containing metric CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Directory where HTML plots are written.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.8,
        help="EMA smoothing factor in [0, 1).",
    )
    parser.add_argument(
        "--no-robust-range",
        action="store_true",
        help="Disable percentile-based y-axis clipping.",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="Lower percentile for robust y-axis range.",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for robust y-axis range.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot windows in addition to writing HTML files.",
    )
    args = parser.parse_args()

    if not (0 <= args.smoothing < 1):
        raise ValueError("--smoothing must be in [0, 1).")
    if not (0 <= args.lower_percentile < args.upper_percentile <= 100):
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    return args


def main() -> None:
    args = parse_args()
    robust_range = not args.no_robust_range
    build_separate_plots(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        smoothing=args.smoothing,
        robust_range=robust_range,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        show=args.show,
    )


if __name__ == "__main__":
    main()