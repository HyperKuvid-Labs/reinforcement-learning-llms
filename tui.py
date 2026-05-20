from __future__ import annotations

import argparse
import threading
from pathlib import Path
from queue import Empty, Queue
import traceback

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmrl.auth import ensure_hf_credentials
from llmrl.config import DEFAULT_ALGOS, DEFAULT_MODELS, RunConfig
from llmrl.runtime import Trainer

console = Console()

PALETTE = {
    "bg": "white",
    "fg": "grey93",
    "muted": "grey70",
    "line": "grey50",
    "strong": "grey100",
}


class DashboardState:
    def __init__(self, queue_runs: list[RunConfig]) -> None:
        self.queue_runs = queue_runs
        self.current: dict[str, object] = {}
        self.history: list[dict[str, object]] = []
        self.logs: list[str] = []

    def update(self, payload: dict[str, object]) -> None:
        self.current = payload
        self._append_payload_log(payload)
        if payload.get("phase") == "done":
            self.history.append(payload)

    def add_log(self, message: str) -> None:
        self.logs.append(message)
        if len(self.logs) > 14:
            self.logs = self.logs[-14:]

    def _append_payload_log(self, payload: dict[str, object]) -> None:
        phase = str(payload.get("phase", "unknown"))
        step = payload.get("global_step", 0)
        line = f"[step {step}] {phase}"
        if phase == "running":
            reward = float(payload.get("train/reward_mean", 0.0))
            loss = float(payload.get("train/loss", 0.0))
            line = f"[step {step}] running | reward={reward:.4f} loss={loss:.4f}"
        elif phase == "error":
            error = str(payload.get("error", "unknown error"))
            line = f"[step {step}] error | {error}"
        elif phase == "auth":
            line = "[step 0] auth | checking HF credentials"
        elif phase == "adapter_fallback":
            line = (
                f"[step {step}] adapter fallback | "
                f"{payload.get('requested_method')} -> {payload.get('fallback_method')}"
            )
        elif phase == "uploading":
            line = f"[step {step}] uploading | {payload.get('checkpoint_dir', '')}"
        elif phase == "done":
            line = f"[step {step}] done"
        self.add_log(line)
        if "traceback" in payload:
            for entry in str(payload["traceback"]).strip().splitlines()[-8:]:
                self.add_log(entry)


def make_header(current: dict[str, object]) -> Panel:
    title = Text("AIME 2025 RL Training Console", style="bold white", justify="center")
    subtitle = Text(
        f"{current.get('model_id', 'idle')}  |  {current.get('algo', 'queue')}  |  TensorBoard-first",
        style="grey70",
        justify="center",
    )
    return Panel(Group(title, subtitle), box=box.SQUARE, border_style="grey50")


def make_queue(state: DashboardState) -> Panel:
    table = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold white")
    table.add_column("Model")
    table.add_column("Algo")
    table.add_column("DPPO")
    table.add_column("Status")
    current_slug = state.current.get("run_name")
    finished = {row.get("run_name") for row in state.history}
    for run in state.queue_runs:
        status = "queued"
        if run.slug == current_slug:
            status = str(state.current.get("phase", "running"))
        elif run.slug in finished:
            status = "done"
        table.add_row(run.model_id.split("/")[-1], run.algo, run.dppo_approx if run.algo == "dppo" else "-", status)
    return Panel(table, title="Run Queue", box=box.SQUARE, border_style="grey50")


def make_metrics(current: dict[str, object]) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="grey70", width=18)
    table.add_column(style="white")
    rows = [
        ("phase", current.get("phase", "idle")),
        ("global step", current.get("global_step", 0)),
        ("reward mean", f"{float(current.get('train/reward_mean', 0.0)):.4f}"),
        ("loss", f"{float(current.get('train/loss', 0.0)):.4f}"),
        ("accuracy", f"{float(current.get('eval/accuracy', 0.0)):.4f}"),
        ("tokens/sec", f"{float(current.get('system/tokens_per_sec', 0.0)):.2f}"),
        ("gpu alloc", f"{float(current.get('system/gpu_mem_allocated', 0.0)) / (1024**3):.2f} GB"),
        ("checkpoint upload", f"{float(current.get('system/checkpoint_upload_time', 0.0)):.2f} s"),
    ]
    for key, value in rows:
        table.add_row(key, str(value))
    return Panel(table, title="Live Metrics", box=box.SQUARE, border_style="grey50")


def make_divergence(current: dict[str, object]) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="grey70", width=18)
    table.add_column(style="white")
    table.add_row("binary", f"{float(current.get('divergence/binary', 0.0)):.6f}")
    table.add_row("top-k", f"{float(current.get('divergence/topk', 0.0)):.6f}")
    table.add_row("naive tv", f"{float(current.get('divergence/naive_tv', 0.0)):.6f}")
    table.add_row("naive kl", f"{float(current.get('divergence/naive_kl', 0.0)):.6f}")
    return Panel(table, title="Divergence", box=box.SQUARE, border_style="grey50")


def make_footer(state: DashboardState) -> Panel:
    lines = []
    for item in state.history[-6:]:
        lines.append(
            f"{item.get('model_id', '').split('/')[-1]} | {item.get('algo')} | step {item.get('global_step')} | done"
        )
    body = "\n".join(lines) if lines else "No completed runs yet."
    return Panel(Align.left(body), title="Recent Runs", box=box.SQUARE, border_style="grey50")


def make_logs(state: DashboardState) -> Panel:
    body = "\n".join(state.logs[-12:]) if state.logs else "No logs yet."
    return Panel(Align.left(body), title="Logs", box=box.SQUARE, border_style="grey50")


def build_layout(state: DashboardState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="bottom", size=10),
    )
    layout["main"].split_row(
        Layout(name="queue", ratio=2),
        Layout(name="metrics", ratio=2),
        Layout(name="divergence", ratio=1),
    )
    layout["bottom"].split_row(
        Layout(name="logs", ratio=2),
        Layout(name="footer", ratio=2),
    )
    layout["header"].update(make_header(state.current))
    layout["queue"].update(make_queue(state))
    layout["metrics"].update(make_metrics(state.current))
    layout["divergence"].update(make_divergence(state.current))
    layout["logs"].update(make_logs(state))
    layout["footer"].update(make_footer(state))
    return layout


def run_training(queue_runs: list[RunConfig], state: DashboardState, events: Queue) -> None:
    for run in queue_runs:
        try:
            trainer = Trainer(run, callback=events.put)
            trainer.train()
        except Exception as exc:
            events.put(
                {
                    "phase": "error",
                    "global_step": 0,
                    "run_name": run.slug,
                    "model_id": run.model_id,
                    "algo": run.algo,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            break


def run_single_training(run: RunConfig, state: DashboardState, events: Queue) -> None:
    try:
        trainer = Trainer(run, callback=events.put)
        trainer.train()
    except Exception as exc:
        events.put(
            {
                "phase": "error",
                "global_step": 0,
                "run_name": run.slug,
                "model_id": run.model_id,
                "algo": run.algo,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_config_with_tui(run: RunConfig) -> None:
    ensure_hf_credentials(prompt=True)
    state = DashboardState([run])
    events: Queue = Queue()
    worker = threading.Thread(target=run_single_training, args=(run, state, events), daemon=True)
    worker.start()

    with Live(build_layout(state), console=console, screen=True, auto_refresh=False) as live:
        while worker.is_alive() or not events.empty():
            updated = False
            try:
                payload = events.get(timeout=0.5)
                state.update(payload)
                updated = True
            except Empty:
                pass
            while not events.empty():
                state.update(events.get())
                updated = True
            if updated:
                live.update(build_layout(state), refresh=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grey/white TUI for queued training runs.")
    parser.add_argument("--status-dir", type=Path, default=Path(".runtime"))
    args = parser.parse_args()
    ensure_hf_credentials(prompt=True)

    queue_runs = []
    for model_id in DEFAULT_MODELS:
        for algo in DEFAULT_ALGOS:
            queue_runs.append(
                RunConfig(
                    model_id=model_id,
                    algo=algo,
                    dppo_approx="topk",
                    status_file=args.status_dir / f"{model_id.split('/')[-1]}-{algo}.json",
                    cpu_offload=True,
                )
            )

    state = DashboardState(queue_runs)
    events: Queue = Queue()
    worker = threading.Thread(target=run_training, args=(queue_runs, state, events), daemon=True)
    worker.start()

    with Live(build_layout(state), console=console, screen=True, auto_refresh=False) as live:
        while worker.is_alive() or not events.empty():
            updated = False
            try:
                payload = events.get(timeout=0.5)
                state.update(payload)
                updated = True
            except Empty:
                pass
            while not events.empty():
                state.update(events.get())
                updated = True
            if updated:
                live.update(build_layout(state), refresh=True)


if __name__ == "__main__":
    main()
