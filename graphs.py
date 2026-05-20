from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_SCALARS = [
    "train/loss",
    "train/reward_mean",
    "eval/accuracy",
    "divergence/topk",
]


def read_scalar_series(run_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TensorBoard runs for GRPO/PPO/DPPO comparisons.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--tag", choices=DEFAULT_SCALARS, default="eval/accuracy")
    args = parser.parse_args()

    fig = go.Figure()
    for run_dir in sorted(path for path in args.runs_root.iterdir() if path.is_dir()):
        try:
            x, y = read_scalar_series(run_dir, args.tag)
        except Exception:
            continue
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=run_dir.name))

    fig.update_layout(
        title=args.tag,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#222222"),
        xaxis_title="Step",
        yaxis_title=args.tag,
    )
    fig.show()


if __name__ == "__main__":
    main()
