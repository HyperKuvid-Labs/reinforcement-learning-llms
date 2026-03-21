import threading
import time
from collections import deque
from datetime import timedelta
from typing import Optional
import sys
import re
import numbers

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import GRPOTrainer, GRPOConfig
from torch.utils.tensorboard import SummaryWriter

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

# enabling tf32 for faster matmul on a100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model_name = "Qwen/Qwen3.5-4B"

SYSTEM_PROMPT = """\
<system>
    <role>You are a mathematical reasoning assistant.</role>
    <instructions>
        <step>Think through the problem carefully inside &lt;think&gt;...&lt;/think&gt; tags.</step>
        <step>Show your full step-by-step reasoning inside the think block.</step>
        <step>Provide your final numeric answer inside \\boxed{{...}}.</step>
    </instructions>
    <format>
        <think>step-by-step reasoning here</think>
        \\boxed{{final answer}}
    </format>
</system>"""


# ─── Sparkline ────────────────────────────────────────────────────────────────

_SPARK = "▁▂▃▄▅▆▇█"


class SparklineChart:
    def __init__(self, width: int = 46, maxlen: int = 300) -> None:
        self.width = width
        self.values: deque[float] = deque(maxlen=maxlen)

    def push(self, v: float) -> None:
        self.values.append(v)

    @property
    def latest(self) -> Optional[float]:
        return self.values[-1] if self.values else None

    @property
    def minimum(self) -> Optional[float]:
        return min(self.values) if self.values else None

    def render(self, color: str = "cyan") -> Text:
        if len(self.values) < 2:
            return Text("─" * self.width, style="dim white")
        pts = list(self.values)[-self.width :]
        lo, hi = min(pts), max(pts)
        span = (hi - lo) or 1e-9
        chars = [_SPARK[int((v - lo) / span * (len(_SPARK) - 1))] for v in pts]
        pad = self.width - len(chars)
        return Text(" " * pad + "".join(chars), style=color)


# ─── Shared training state ────────────────────────────────────────────────────


class TrainState:
    def __init__(self) -> None:
        self.step: int = 0
        self.max_steps: int = 0
        self.epoch: float = 0.0
        self.total_epochs: int = 1

        self.loss = SparklineChart()
        self.pg_loss = SparklineChart()
        self.kl = SparklineChart()
        self.reward = SparklineChart()
        self.entropy = SparklineChart()

        # (step, accuracy) pairs from reasoning_accuracy_reward logs
        self.accuracy_history: list[tuple[int, float]] = []

        self.status: str = "initialising…"
        self.done: bool = False
        self.elapsed: float = 0.0
        self.train_start: float = 0.0  # wall time when trainer.train() begins
        self.train_end: float = 0.0  # wall time when trainer.train() finishes
        self.stdout_lines: deque[str] = deque(maxlen=400)
        self.lock = threading.Lock()


class _TeeStream:
    def __init__(self, stream, ts: TrainState) -> None:
        self._stream = stream
        self._state = ts
        self._partial = ""

    def write(self, data: str):
        self._stream.write(data)
        if not data:
            return
        text = self._partial + data
        lines = text.splitlines(keepends=True)
        complete: list[str] = []
        partial = ""
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                stripped = line.strip()
                if stripped:
                    complete.append(stripped)
            else:
                partial += line
        self._partial = partial
        if complete:
            with self._state.lock:
                self._state.stdout_lines.extend(complete)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def fileno(self):
        return self._stream.fileno()


# ─── Trainer callback ─────────────────────────────────────────────────────────


class RichGRPOCallback(TrainerCallback):
    def __init__(self, ts: TrainState) -> None:
        self._s = ts
        self._tb_writer: Optional[SummaryWriter] = None

    @staticmethod
    def _first_scalar(logs, keys: tuple[str, ...]) -> Optional[float]:
        for key in keys:
            if key in logs and isinstance(logs[key], numbers.Real):
                return float(logs[key])
        return None

    def _write_tb_aliases(self, logs, step: int) -> None:
        if self._tb_writer is None:
            return

        reward_keys = tuple(k for k in logs if "reward" in k.lower())
        reward_mean = self._first_scalar(
            logs, tuple(k for k in reward_keys if "mean" in k) or reward_keys
        )
        accuracy = self._first_scalar(
            logs,
            (
                "rewards/accuracy_reward_func",
                "accuracy_reward",
                "reward/accuracy",
                "accuracy",
            ),
        )
        format_reward = self._first_scalar(
            logs,
            (
                "rewards/format_reward_func",
                "format_reward",
                "reward/format",
            ),
        )

        aliases: dict[str, Optional[float]] = {
            "loss": self._first_scalar(logs, ("loss", "train_loss")),
            "kl": self._first_scalar(logs, ("kl", "kl_loss", "actor/kl")),
            "lr": self._first_scalar(logs, ("learning_rate", "lr")),
            "grad_norm": self._first_scalar(logs, ("grad_norm", "gradient_norm")),
            "clipped_ratio": self._first_scalar(
                logs,
                (
                    "clip_ratio",
                    "clipped_ratio",
                    "policy/clipped_ratio",
                    "actor/clipped_ratio",
                ),
            ),
            "completion_length": self._first_scalar(
                logs,
                (
                    "completion_length",
                    "completions/mean_length",
                    "mean_completion_length",
                ),
            ),
            "reward": reward_mean,
            "reward_std": self._first_scalar(
                logs, ("reward_std", "rewards/std", "std_reward")
            ),
            "accuracy_r_mean": accuracy,
            "format_r_mean": format_reward,
            "forward_r_mean": format_reward,
        }

        for tag, value in aliases.items():
            if value is not None:
                self._tb_writer.add_scalar(tag, value, step)
        self._tb_writer.flush()

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
        with self._s.lock:
            self._s.max_steps = state.max_steps or 0
            self._s.total_epochs = int(args.num_train_epochs)
            self._s.status = "training"
            self._s.train_start = time.monotonic()
        self._tb_writer = SummaryWriter(log_dir=args.logging_dir)

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kw
    ):
        if not logs:
            return
        s = self._s
        with s.lock:
            s.step = state.global_step
            s.epoch = state.epoch

            for key in ("loss", "train_loss"):
                if key in logs:
                    s.loss.push(float(logs[key]))
                    break

            for key in ("pg_loss", "policy_loss", "actor/pg_loss"):
                if key in logs:
                    s.pg_loss.push(float(logs[key]))
                    break

            for key in ("kl", "kl_loss", "actor/kl"):
                if key in logs:
                    s.kl.push(float(logs[key]))
                    break

            for key in ("entropy", "actor/entropy"):
                if key in logs:
                    s.entropy.push(float(logs[key]))
                    break

            # reward: prefer mean key, fall back to first reward key found
            reward_keys = [k for k in logs if "reward" in k.lower()]
            if reward_keys:
                key = next((k for k in reward_keys if "mean" in k), reward_keys[0])
                s.reward.push(float(logs[key]))

            # accuracy from reasoning_accuracy_reward
            for key in (
                "rewards/accuracy_reward_func",
                "accuracy_reward",
                "rewards/reasoning_accuracy_reward",
                "reasoning_accuracy_reward",
                "reward/accuracy",
                "accuracy",
            ):
                if key in logs:
                    s.accuracy_history.append((state.global_step, float(logs[key])))
                    break

        self._write_tb_aliases(logs, state.global_step)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kw):
        with self._s.lock:
            self._s.status = "saving checkpoint…"

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kw):
        with self._s.lock:
            self._s.status = "evaluating…"

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        with self._s.lock:
            self._s.train_end = time.monotonic()
            self._s.done = True
            self._s.status = "done"
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None


# ─── TUI renderer ─────────────────────────────────────────────────────────────


def _fmt(v: Optional[float], d: int = 5) -> str:
    return f"{v:.{d}f}" if v is not None else "—"


def _metrics_table(s: TrainState) -> Table:
    tbl = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold bright_white",
        expand=True,
        pad_edge=False,
    )
    tbl.add_column("Metric", style="dim white", no_wrap=True, ratio=3)
    tbl.add_column(
        "Current", style="bright_cyan", no_wrap=True, ratio=2, justify="right"
    )
    tbl.add_column("Min", style="bright_yellow", no_wrap=True, ratio=2, justify="right")
    tbl.add_column("Sparkline", no_wrap=True, ratio=6)
    for label, chart, color in [
        ("Loss", s.loss, "bright_red"),
        ("PG Loss", s.pg_loss, "red"),
        ("KL", s.kl, "magenta"),
        ("Reward Mean", s.reward, "bright_green"),
        ("Entropy", s.entropy, "yellow"),
    ]:
        tbl.add_row(label, _fmt(chart.latest), _fmt(chart.minimum), chart.render(color))
    return tbl


def _accuracy_table(s: TrainState) -> Table:
    tbl = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold bright_white",
        expand=True,
        pad_edge=False,
    )
    tbl.add_column("Step", style="dim white", no_wrap=True, justify="right")
    tbl.add_column("Accuracy", style="bright_green", no_wrap=True, justify="center")
    tbl.add_column("Bar", no_wrap=False)
    for step, acc in s.accuracy_history[-8:]:
        color = "bright_green" if acc >= 0.7 else "yellow" if acc >= 0.4 else "red"
        filled = int(acc * 20)
        tbl.add_row(
            str(step),
            f"{acc * 100:.1f}%",
            Text("█" * filled + "░" * (20 - filled), style=color),
        )
    if not s.accuracy_history:
        tbl.add_row("—", "awaiting logs…", Text(""))
    return tbl


def _stdout_text(s: TrainState) -> Text:
    if not s.stdout_lines:
        return Text("Awaiting printed completions…", style="dim white")
    out = Text()
    for idx, line in enumerate(list(s.stdout_lines)[-8:]):
        out.append(line)
        if idx < 7:
            out.append("\n")
    return out


def _build_ui(s: TrainState, prog: Progress) -> Layout:
    with s.lock:
        step, total = s.step, s.max_steps
        elapsed = str(timedelta(seconds=int(s.elapsed)))
        epoch_str = f"{s.epoch:.2f} / {s.total_epochs}"
        status_col = "bright_cyan" if s.status == "training" else "yellow"
        best = max((a for _, a in s.accuracy_history), default=None)
        sps_str = ""
        if s.train_start > 0 and s.step > 0:
            dt = time.monotonic() - s.train_start
            if dt > 0.1:
                sps_str = f"  ·  {s.step / dt:.2f} steps/sec"

        hdr = Table(box=None, expand=True, show_header=False, pad_edge=False)
        hdr.add_column(ratio=4)
        hdr.add_column(ratio=2)
        hdr.add_column(ratio=2)
        hdr.add_column(ratio=2)
        hdr.add_row(
            Text(f"  {model_name}", style="bold bright_white"),
            Text(f"step  {step:>5} / {total or '?'}", style="cyan"),
            Text(f"epoch  {epoch_str}", style="cyan"),
            Text(f"  {s.status}", style=status_col),
        )
        header = Panel(
            hdr,
            border_style="bright_blue",
            title="[bold bright_blue]TRL GRPO · GSM8K · vLLM[/bold bright_blue]",
            subtitle=f"[dim]elapsed {elapsed}{sps_str}[/dim]",
        )

        acc_title = "[bold]accuracy reward[/bold]"
        if best is not None:
            acc_title += f"  [bright_green]best {best * 100:.1f}%[/bright_green]"

        metrics_panel = Panel(
            _metrics_table(s),
            title="[bold]live metrics[/bold]",
            border_style="blue",
            padding=(0, 1),
        )
        accuracy_panel = Panel(
            _accuracy_table(s), title=acc_title, border_style="green", padding=(0, 1)
        )
        print_panel = Panel(
            _stdout_text(s),
            title="[bold]printed samples[/bold]",
            border_style="yellow",
            padding=(0, 1),
        )
        prog_panel = Panel(prog, border_style="dim", padding=(0, 1))

    layout = Layout()
    layout.split_column(
        Layout(header, name="header", size=5),
        Layout(name="body", ratio=1),
        Layout(prog_panel, name="progress", size=4),
    )
    layout["body"].split_row(
        Layout(metrics_panel, name="metrics", ratio=3),
        Layout(name="right", ratio=3),
    )
    layout["right"].split_column(
        Layout(accuracy_panel, name="accuracy", ratio=2),
        Layout(print_panel, name="prints", ratio=1),
    )
    return layout


def _live_loop(s: TrainState, prog: Progress, task_id, console: Console) -> None:
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            with s.lock:
                step, total, done = s.step, s.max_steps, s.done
            prog.update(
                task_id,
                completed=step,
                total=total or 1,
                description=f"step {step}/{total or '?'}",
            )
            live.update(_build_ui(s, prog))
            if done:
                live.update(_build_ui(s, prog))
                break
            time.sleep(0.25)


def _print_summary(
    s: TrainState,
    console: Console,
    t0: float,
    t_model: float,
    t_data: float,
    t_save: float,
) -> None:
    def _dur(a: float, b: float) -> str:
        return str(timedelta(seconds=int(b - a))) if b > a else "—"

    console.print()
    console.print(Rule("[bold bright_blue]Training Complete[/bold bright_blue]"))
    tbl = Table(
        box=box.ROUNDED,
        show_header=False,
        expand=False,
        border_style="blue",
        padding=(0, 2),
    )
    tbl.add_column(style="dim white", no_wrap=True)
    tbl.add_column(style="bright_white", no_wrap=True)

    tbl.add_row("Model", model_name)
    tbl.add_row("Total steps", str(s.step))

    # ── timing breakdown ──────────────────────────────────────────────────
    tbl.add_row("", "")
    tbl.add_row("Model load + LoRA", _dur(t0, t_model))
    tbl.add_row("Dataset prep", _dur(t_model, t_data))
    # tbl.add_row("vLLM init",          _dur(t_data,  t_vllm))
    if s.train_start > 0:
        t_end = s.train_end if s.train_end > s.train_start else t_save
        tbl.add_row("Training loop", _dur(s.train_start, t_end))
    tbl.add_row("Total wall time", _dur(t0, t_save))
    if s.train_start > 0 and s.train_end > s.train_start and s.step > 0:
        sps = s.step / (s.train_end - s.train_start)
        tbl.add_row("Avg steps/sec", f"{sps:.3f}")
        tbl.add_row("Avg sec/step", f"{1 / sps:.2f}s")

    # ── metrics ───────────────────────────────────────────────────────────
    tbl.add_row("", "")
    if s.loss.minimum is not None:
        tbl.add_row("Best loss", _fmt(s.loss.minimum))
    if s.reward.latest is not None:
        tbl.add_row("Final reward", _fmt(s.reward.latest))
    if s.accuracy_history:
        best_step, best_acc = max(s.accuracy_history, key=lambda x: x[1])
        last_step, last_acc = s.accuracy_history[-1]
        tbl.add_row("Best accuracy", f"{best_acc * 100:.2f}%  (step {best_step})")
        tbl.add_row("Final accuracy", f"{last_acc * 100:.2f}%  (step {last_step})")
    else:
        tbl.add_row("Accuracy", "no accuracy logs recorded")
    console.print(Align.center(tbl))
    console.print()


# ─── Dataset ──────────────────────────────────────────────────────────────────

_t0 = time.monotonic()  # script start — for cross-run comparison

ds = load_dataset("gsm8k", "main", split="train")


def process(example):
    ground_truth = example["answer"].split("####")[-1].strip()
    return {
        "prompt": f"{SYSTEM_PROMPT}\n\nQuestion: {example['question']}",
        "ground_truth": ground_truth,
    }


dataset = ds.map(process)
_t_data = time.monotonic()  # dataset ready


def extract_boxed_answer(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{(.*?)\}", text)
    return m.group(1).strip() if m else None


def accuracy_reward_func(completions, ground_truth, **kwargs):
    return [
        1.0 if extract_boxed_answer(c) == gt else 0.0
        for c, gt in zip(completions, ground_truth)
    ]


def format_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        has_think = bool(re.search(r"<think>.*?</think>", c, re.DOTALL))
        has_boxed = bool(re.search(r"\\\\boxed\{.*?\}", c, re.DOTALL))
        if has_think and has_boxed:
            rewards.append(1.0)
        elif has_boxed:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


reward_funcs = [accuracy_reward_func, format_reward_func]


# ─── vLLM + LoRA + config ─────────────────────────────────────────────────────

# vllm_model, vllm_tokenizer = setup_vllm(
#     model_name,
#     dtype="bfloat16",
#     tensor_parallel_size=1,
# )
# _t_vllm = time.monotonic()   # vLLM engine ready

# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

training_args = GRPOConfig(
    output_dir="./grpo_gsm8k",
    logging_dir="./grpo_gsm8k/tb_logs",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="adamw_8bit",
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=True,
    num_generations=8,
    generation_kwargs={
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    },
    logging_steps=5,
    save_steps=200,
    report_to="tensorboard",
    use_vllm=True,
    vllm_mode="colocate",
    temperature=0.7,
    max_prompt_length=256,
    num_completions_to_print=8,
    top_p=0.95,
)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Fix for trl/peft compatibility: GRPOTrainer expects warnings_issued attribute
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

_t_model = time.monotonic()  # model + tokenizer loaded


# ─── Wire up TUI and train ────────────────────────────────────────────────────

_console = Console()
_tui_state = TrainState()

_progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.description}"),
    BarColumn(bar_width=None),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    console=_console,
    expand=True,
)
_task_id = _progress.add_task("training", total=1)
_rich_cb = RichGRPOCallback(_tui_state)
_start = time.monotonic()


def _tick():
    while not _tui_state.done:
        _tui_state.elapsed = time.monotonic() - _start
        time.sleep(0.1)
    _tui_state.elapsed = time.monotonic() - _start


threading.Thread(target=_tick, daemon=True).start()

_live_t = threading.Thread(
    target=_live_loop,
    args=(_tui_state, _progress, _task_id, _console),
    daemon=True,
)
_live_t.start()

_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
sys.stdout = _TeeStream(_orig_stdout, _tui_state)
sys.stderr = _TeeStream(_orig_stderr, _tui_state)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_funcs,
    callbacks=[_rich_cb],
)

try:
    trainer.train()
finally:
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
_t_saved = time.monotonic()  # (trl saves inside train(); capture wall time here)

_live_t.join(timeout=3)
_print_summary(_tui_state, _console, _t0, _t_model, _t_data, _t_saved)
