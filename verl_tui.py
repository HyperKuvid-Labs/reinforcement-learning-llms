#!/usr/bin/env python3
"""
verl_tui.py — Rich TUI wrapper for verl GRPO training.

Streams the training subprocess, parses metrics in real-time,
and renders live sparkline charts + a final accuracy summary.

Usage:
    python verl_tui.py            # runs training with defaults
    python verl_tui.py --dry-run  # simulate with fake data (no GPU needed)
"""

import argparse
import re
import subprocess
import sys
import time
import threading
from collections import deque
from datetime import timedelta
from typing import Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ─── Training command (mirrors verl_train.sh) ────────────────────────────────

VERL_CMD = [
    "python", "-m", "verl.trainer.main_ppo",
    "algorithm.adv_estimator=grpo",
    "trainer.val_before_train=False",
    "data.train_files=~/data/gsm8k/train.parquet",
    "data.val_files=~/data/gsm8k/test.parquet",
    "data.train_batch_size=128",
    "data.max_prompt_length=512",
    "data.max_response_length=1024",
    "data.filter_overlong_prompts=True",
    "data.truncation=error",
    "data.shuffle=True",
    "actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct",
    "actor_rollout_ref.model.lora_rank=96",
    "actor_rollout_ref.model.lora_alpha=32",
    "actor_rollout_ref.actor.optim.lr=5e-6",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.actor.ppo_mini_batch_size=32",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
    "actor_rollout_ref.actor.use_kl_loss=True",
    "actor_rollout_ref.actor.kl_loss_coef=0.001",
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
    "actor_rollout_ref.actor.entropy_coeff=0.0",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=24",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.88",
    "actor_rollout_ref.rollout.n=8",
    "actor_rollout_ref.rollout.load_format=safetensors",
    "actor_rollout_ref.rollout.layered_summon=True",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=24",
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    "algorithm.use_kl_in_reward=False",
    "trainer.critic_warmup=0",
    'trainer.logger=["console"]',
    "trainer.project_name=verl_grpo_gsm8k",
    "trainer.experiment_name=qwen2.5-7b_grpo_single_a100",
    "trainer.n_gpus_per_node=1",
    "trainer.nnodes=1",
    "trainer.save_freq=50",
    "trainer.test_freq=10",
    "trainer.total_epochs=4",
]

TOTAL_EPOCHS   = 4
STEPS_PER_EPOCH = 250   # approximate; updates live once the actual value is parsed


# ─── Sparkline chart ──────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"

class SparklineChart:
    """Fixed-width sparkline backed by a rolling deque."""

    def __init__(self, width: int = 48, maxlen: int = 200) -> None:
        self.width  = width
        self.values: deque[float] = deque(maxlen=maxlen)

    def push(self, val: float) -> None:
        self.values.append(val)

    @property
    def latest(self) -> Optional[float]:
        return self.values[-1] if self.values else None

    @property
    def minimum(self) -> Optional[float]:
        return min(self.values) if self.values else None

    def render(self, color: str = "cyan") -> Text:
        if len(self.values) < 2:
            return Text("─" * self.width, style="dim white")
        pts   = list(self.values)[-self.width:]
        lo, hi = min(pts), max(pts)
        span   = (hi - lo) or 1e-9
        chars  = [
            _SPARK_CHARS[int((v - lo) / span * (len(_SPARK_CHARS) - 1))]
            for v in pts
        ]
        pad  = self.width - len(chars)
        return Text(" " * pad + "".join(chars), style=color)


# ─── State shared between the reader thread and the renderer ─────────────────

class TrainState:
    def __init__(self) -> None:
        self.step: int             = 0
        self.epoch: float          = 0.0
        self.total_steps: int      = STEPS_PER_EPOCH * TOTAL_EPOCHS

        self.actor_loss            = SparklineChart()
        self.pg_loss               = SparklineChart()
        self.kl                    = SparklineChart()
        self.reward                = SparklineChart()
        self.entropy               = SparklineChart()

        self.val_scores: list[tuple[int, float]] = []   # (step, score)

        self.log_tail: deque[str]  = deque(maxlen=10)
        self.status: str           = "initialising…"
        self.done: bool             = False
        self.exit_code: int         = 0
        self.elapsed: float         = 0.0
        self.first_step_time: float = 0.0   # absolute monotonic time when step 1 appears
        self.lock                   = threading.Lock()


# ─── Log-line parser ──────────────────────────────────────────────────────────

def _kv(line: str, key: str) -> Optional[float]:
    """Extract a numeric value from key=value or key: value style logs."""
    m = re.search(
        rf"(?:^|[\s,|])(?:{re.escape(key)})[=:\s]+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        line, re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def parse_line(line: str, state: TrainState) -> None:
    """Parse one stdout line and update TrainState in-place."""
    with state.lock:
        state.log_tail.append(line.rstrip())

        # Step / epoch
        if (s := _kv(line, r"global[_/]?step")) is not None:
            new_step = int(s)
            if new_step > 0 and state.first_step_time == 0.0:
                state.first_step_time = time.monotonic()
            state.step = new_step
        elif (s := _kv(line, "step")) is not None and state.step < int(s):
            new_step = int(s)
            if new_step > 0 and state.first_step_time == 0.0:
                state.first_step_time = time.monotonic()
            state.step = new_step

        if (e := _kv(line, "epoch")) is not None:
            state.epoch = e

        # Losses
        for attr, keys in [
            ("actor_loss",  [r"actor/loss",     r"actor_loss",    r"loss/actor"]),
            ("pg_loss",     [r"actor/pg_loss",   r"pg_loss",       r"policy_loss"]),
            ("kl",          [r"actor/kl",        r"kl_loss",       r"kl"]),
            ("reward",      [r"critic/rewards/mean", r"reward_mean", r"mean_reward", r"reward"]),
            ("entropy",     [r"actor/entropy",   r"entropy"]),
        ]:
            for key in keys:
                if (v := _kv(line, key)) is not None:
                    getattr(state, attr).push(v)
                    break

        # Validation accuracy (test_score / val_score)
        for key in [r"val/test_score", r"test_score", r"val_score", r"accuracy"]:
            if (v := _kv(line, key)) is not None:
                state.val_scores.append((state.step, v))
                break

        # Total-step hint from verl logs
        if (t := _kv(line, r"total[_/]?steps")) is not None:
            state.total_steps = int(t)

        # Status hints
        if "training" in line.lower() and "start" in line.lower():
            state.status = "training"
        elif "saving" in line.lower() or "checkpoint" in line.lower():
            state.status = "saving checkpoint…"
        elif "validation" in line.lower() or "evaluating" in line.lower():
            state.status = "evaluating…"
        elif state.step > 0:
            state.status = "training"


# ─── Layout builder ───────────────────────────────────────────────────────────

def _fmt(val: Optional[float], decimals: int = 4) -> str:
    return f"{val:.{decimals}f}" if val is not None else "—"


def build_metrics_table(state: TrainState) -> Table:
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold bright_white",
                expand=True, pad_edge=False)
    tbl.add_column("Metric",  style="dim white",       no_wrap=True, ratio=3)
    tbl.add_column("Current", style="bright_cyan",     no_wrap=True, ratio=2, justify="right")
    tbl.add_column("Min",     style="bright_yellow",   no_wrap=True, ratio=2, justify="right")
    tbl.add_column("Sparkline",                        no_wrap=True, ratio=6)

    rows = [
        ("Actor Loss",  state.actor_loss, "bright_red"),
        ("PG Loss",     state.pg_loss,    "red"),
        ("KL Div",      state.kl,         "magenta"),
        ("Reward Mean", state.reward,     "bright_green"),
        ("Entropy",     state.entropy,    "yellow"),
    ]
    for label, chart, color in rows:
        tbl.add_row(
            label,
            _fmt(chart.latest),
            _fmt(chart.minimum),
            chart.render(color),
        )
    return tbl


def build_val_table(state: TrainState) -> Table:
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold bright_white",
                expand=True, pad_edge=False)
    tbl.add_column("Step",     style="dim white",    no_wrap=True, justify="right")
    tbl.add_column("Val Accuracy", style="bright_green", justify="center")
    tbl.add_column("Bar",      no_wrap=False)

    for step, score in state.val_scores[-8:]:
        filled = int(score * 20)
        bar    = Text("█" * filled + "░" * (20 - filled),
                      style="bright_green" if score >= 0.7 else "yellow" if score >= 0.4 else "red")
        tbl.add_row(str(step), f"{score*100:.1f}%", bar)

    if not state.val_scores:
        tbl.add_row("—", "awaiting first eval…", Text(""))
    return tbl


def build_log_panel(state: TrainState) -> Panel:
    lines = list(state.log_tail)
    text  = Text()
    for ln in lines:
        if any(k in ln for k in ("error", "Error", "ERROR", "Traceback")):
            text.append(ln + "\n", style="bright_red")
        elif any(k in ln for k in ("warn", "Warn", "WARN")):
            text.append(ln + "\n", style="yellow")
        else:
            text.append(ln + "\n", style="dim white")
    return Panel(text, title="[bold]stdout tail[/bold]", border_style="dim white",
                 padding=(0, 1))


def build_layout(state: TrainState, progress: Progress) -> Layout:
    with state.lock:
        step, total = state.step, state.total_steps
        pct = (step / total * 100) if total else 0.0

        # ── header ──────────────────────────────────────────────────────────
        epoch_str  = f"{state.epoch:.2f} / {TOTAL_EPOCHS}"
        elapsed    = str(timedelta(seconds=int(state.elapsed)))
        status_col = "bright_cyan" if state.status == "training" else "yellow"
        header_tbl = Table(box=None, expand=True, show_header=False, pad_edge=False)
        header_tbl.add_column(ratio=3)
        header_tbl.add_column(ratio=2)
        header_tbl.add_column(ratio=2)
        header_tbl.add_column(ratio=2)
        header_tbl.add_row(
            Text(f"  Qwen2.5-7B · GRPO · GSM8K", style="bold bright_white"),
            Text(f"step  {step:>6} / {total}", style="cyan"),
            Text(f"epoch  {epoch_str}", style="cyan"),
            Text(f"  {state.status}", style=status_col),
        )
        sps_str = ""
        if state.first_step_time > 0 and state.step > 0:
            dt = time.monotonic() - state.first_step_time
            if dt > 0.1:
                sps_str = f"  ·  {state.step / dt:.2f} steps/sec"
        header_panel = Panel(header_tbl, border_style="bright_blue",
                             title="[bold bright_blue]verl GRPO Training[/bold bright_blue]",
                             subtitle=f"[dim]elapsed {elapsed}{sps_str}[/dim]")

        # ── metrics ─────────────────────────────────────────────────────────
        metrics_panel = Panel(
            build_metrics_table(state),
            title="[bold]live metrics[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

        # ── validation ──────────────────────────────────────────────────────
        best = max((s for _, s in state.val_scores), default=None)
        val_title = "[bold]validation accuracy[/bold]"
        if best is not None:
            val_title += f"  [bright_green]best {best*100:.1f}%[/bright_green]"
        val_panel = Panel(
            build_val_table(state),
            title=val_title, border_style="green", padding=(0, 1),
        )

        # ── progress bar ────────────────────────────────────────────────────
        prog_panel = Panel(progress, border_style="dim", padding=(0, 1))

        # ── log tail ────────────────────────────────────────────────────────
        log_panel = build_log_panel(state)

    layout = Layout()
    layout.split_column(
        Layout(header_panel, name="header",   size=5),
        Layout(name="body", ratio=1),
        Layout(prog_panel,  name="progress",  size=4),
        Layout(log_panel,   name="log",       size=12),
    )
    layout["body"].split_row(
        Layout(metrics_panel, name="metrics", ratio=3),
        Layout(val_panel,     name="val",     ratio=2),
    )
    return layout


# ─── Dry-run simulator ────────────────────────────────────────────────────────

def dry_run_generator():
    """Yields fake verl log lines to demo the TUI without a GPU."""
    import random, math
    total = 40
    yield "Training starts\n"
    for step in range(1, total + 1):
        t       = step / total
        al      = 0.8 * math.exp(-3 * t) + 0.05 + random.gauss(0, 0.01)
        pg      = al * 0.9 + random.gauss(0, 0.005)
        kl      = 0.02 + 0.01 * random.random()
        reward  = 0.1 + 0.6 * t + random.gauss(0, 0.02)
        ent     = 1.2 - 0.5 * t + random.gauss(0, 0.02)
        yield (
            f"global_step={step} epoch={step/10:.2f} "
            f"actor/loss={al:.4f} actor/pg_loss={pg:.4f} "
            f"actor/kl={kl:.4f} critic/rewards/mean={reward:.4f} "
            f"actor/entropy={ent:.4f}\n"
        )
        if step % 10 == 0:
            acc = 0.2 + 0.5 * t + random.gauss(0, 0.03)
            yield f"val/test_score={acc:.4f} step={step}\n"
        time.sleep(0.12)
    yield "Training finished\n"


# ─── Main run loop ────────────────────────────────────────────────────────────

def run(dry: bool = False) -> None:
    state   = TrainState()
    console = Console()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )
    task = progress.add_task("training", total=state.total_steps)

    start = time.monotonic()

    def reader(proc_iter):
        """Read lines from subprocess (or generator) and parse them."""
        for raw in proc_iter:
            line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
            parse_line(line, state)
        with state.lock:
            state.done = True

    if dry:
        proc_iter: object = dry_run_generator()
        t = threading.Thread(target=reader, args=(proc_iter,), daemon=True)
    else:
        proc = subprocess.Popen(
            VERL_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        t = threading.Thread(target=reader, args=(proc.stdout,), daemon=True)

    t.start()

    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            state.elapsed = time.monotonic() - start
            with state.lock:
                done  = state.done
                step  = state.step
                total = state.total_steps

            progress.update(task, completed=step, total=total,
                            description=f"step {step}/{total}")
            live.update(build_layout(state, progress))

            if done:
                time.sleep(0.5)   # final render flush
                live.update(build_layout(state, progress))
                break
            time.sleep(0.25)

    t.join(timeout=5)

    if not dry:
        rc = proc.wait()
        state.exit_code = rc

    # ── Final summary ──────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold bright_blue]Training Complete[/bold bright_blue]"))
    elapsed = str(timedelta(seconds=int(state.elapsed)))

    summary = Table(box=box.ROUNDED, show_header=False, expand=False,
                    border_style="blue", padding=(0, 2))
    summary.add_column(style="dim white",      no_wrap=True)
    summary.add_column(style="bright_white",   no_wrap=True)

    summary.add_row("Total steps",  str(state.step))

    # ── timing breakdown ──────────────────────────────────────────────────
    total_sec = state.elapsed
    summary.add_row("", "")
    if state.first_step_time > 0:
        setup_sec = state.first_step_time - start
        train_sec = max(total_sec - setup_sec, 0)
        summary.add_row("Setup (model load)",  str(timedelta(seconds=int(setup_sec))))
        summary.add_row("Training loop",       str(timedelta(seconds=int(train_sec))))
    summary.add_row("Total wall time",         str(timedelta(seconds=int(total_sec))))
    if state.first_step_time > 0 and state.step > 0:
        train_sec = max(total_sec - (state.first_step_time - start), 0.1)
        sps = state.step / train_sec
        summary.add_row("Avg steps/sec",       f"{sps:.3f}")
        summary.add_row("Avg sec/step",        f"{1/sps:.2f}s")

    # ── metrics ───────────────────────────────────────────────────────────
    summary.add_row("", "")
    if state.actor_loss.minimum is not None:
        summary.add_row("Best actor loss",    _fmt(state.actor_loss.minimum))
    if state.pg_loss.minimum is not None:
        summary.add_row("Best PG loss",       _fmt(state.pg_loss.minimum))
    if state.reward.latest is not None:
        summary.add_row("Final reward mean",  _fmt(state.reward.latest))

    if state.val_scores:
        best_step, best_acc = max(state.val_scores, key=lambda x: x[1])
        last_step, last_acc = state.val_scores[-1]
        summary.add_row("Best val accuracy",  f"{best_acc*100:.2f}%  (step {best_step})")
        summary.add_row("Final val accuracy", f"{last_acc*100:.2f}%  (step {last_step})")
    else:
        summary.add_row("Val accuracy", "no evaluations recorded")

    if not dry:
        color = "bright_green" if state.exit_code == 0 else "bright_red"
        summary.add_row("Exit code", Text(str(state.exit_code), style=color))

    console.print(Align.center(summary))
    console.print()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rich TUI for verl GRPO training")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate training output without a GPU")
    args = parser.parse_args()
    run(dry=args.dry_run)
