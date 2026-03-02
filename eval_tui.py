from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import re
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box
from rich.align import Align
from rich.style import Style
from rich.padding import Padding

# ── config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert mathematician. "
    "Think step by step and put the final answer in \\boxed{}."
)

console = Console()


# ── helpers ──────────────────────────────────────────────────────────────────
def normalize_answer(text: str) -> str:
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def extract_boxed(s: str) -> str:
    match = re.search(r'\\boxed\{(.*)\}', s, re.DOTALL)
    return match.group(1).strip() if match else ""


def truncate(text: str, max_len: int = 42) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ── banner ───────────────────────────────────────────────────────────────────
def print_banner() -> None:
    banner = Text(justify="center")
    banner.append("  ██████╗ ██╗      ██╗      ███╗   ███╗\n", style="bold bright_cyan")
    banner.append("  ██╔══██╗██║      ██║      ████╗ ████║\n", style="bold bright_cyan")
    banner.append("  ██████╔╝██║      ██║      ██╔████╔██║\n", style="bold cyan")
    banner.append("  ██╔══██╗██║      ██║      ██║╚██╔╝██║\n", style="bold cyan")
    banner.append("  ██║  ██║███████╗ ███████╗ ██║ ╚═╝ ██║\n", style="bold blue")
    banner.append("  ╚═╝  ╚═╝╚══════╝ ╚══════╝ ╚═╝     ╚═╝\n", style="bold blue")
    banner.append("  Reinforcement Learning · LLM Evaluator", style="dim italic")

    console.print(
        Panel(
            Align(banner, align="center"),
            border_style="bright_cyan",
            padding=(1, 4),
        )
    )

    meta = Table.grid(padding=(0, 3))
    meta.add_column(style="bold dim")
    meta.add_column(style="bright_white")
    meta.add_row("Model",   MODEL_NAME)
    meta.add_row("Dataset", "HuggingFaceH4/MATH-500")
    meta.add_row("Split",   "test")
    console.print(Align(meta, align="center"))
    console.print()


# ── results table ────────────────────────────────────────────────────────────
def build_table() -> Table:
    table = Table(
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold bright_white on grey23",
        show_lines=False,
        expand=True,
        highlight=True,
    )
    table.add_column("#",        style="dim",          width=5,  justify="right")
    table.add_column("",         width=3,              justify="center")           # ✓ / ✗
    table.add_column("Predicted",                      min_width=20, max_width=45)
    table.add_column("Ground Truth",                   min_width=20, max_width=45)
    table.add_column("Correct?",  justify="center",    width=10)
    return table


# ── score panel ───────────────────────────────────────────────────────────────
def score_panel(correct: int, total: int) -> Panel:
    pct = correct / total if total else 0.0
    bar_width = 32
    filled = int(bar_width * pct)
    bar = (
        "[bold bright_green]" + "█" * filled + "[/]"
        + "[dim]" + "░" * (bar_width - filled) + "[/]"
    )

    if pct >= 0.8:
        colour = "bright_green"
    elif pct >= 0.5:
        colour = "yellow"
    else:
        colour = "red"

    grid = Table.grid(padding=(0, 2))
    grid.add_column()
    grid.add_column()
    grid.add_row(
        f"[bold {colour}]{pct:.1%}[/]",
        f"[dim]({correct} / {total})[/]",
    )
    grid.add_row(bar, "")

    return Panel(
        Align(grid, align="center"),
        title="[bold]Live Score[/bold]",
        border_style=colour,
        padding=(0, 2),
    )


# ── main eval loop ────────────────────────────────────────────────────────────
def run_eval() -> None:
    print_banner()

    # ── load model ───────────────────────────────────────────────────────────
    console.print(Rule("[bold bright_cyan]Loading[/]", style="bright_cyan"))
    with console.status("[bold cyan]Loading tokenizer…[/]", spinner="dots"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with console.status("[bold cyan]Loading model weights…[/]", spinner="bouncingBall"):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    console.print("[bold green]✓[/] Model loaded\n")

    # ── load dataset ──────────────────────────────────────────────────────────
    with console.status("[bold cyan]Fetching dataset…[/]", spinner="dots"):
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    total = len(dataset)
    console.print(f"[bold green]✓[/] Dataset loaded — [bright_white]{total}[/] problems\n")

    console.print(Rule("[bold bright_cyan]Evaluation[/]", style="bright_cyan"))

    # ── live evaluation ───────────────────────────────────────────────────────
    table   = build_table()
    correct = 0
    start   = time.time()

    progress = Progress(
        SpinnerColumn(spinner_name="dots", style="bright_cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None, complete_style="bright_cyan", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("[dim]|"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta"),
        TimeRemainingColumn(),
        expand=True,
    )
    task_id = progress.add_task("Evaluating problems", total=total)

    layout = Layout()
    layout.split_column(
        Layout(name="top",    ratio=3),
        Layout(name="bottom", ratio=1),
    )

    with Live(layout, console=console, refresh_per_second=8, vertical_overflow="visible") as live:
        for i, example in enumerate(dataset):
            progress.update(task_id, description=f"Problem [bold]{i+1}/{total}[/]")

            question    = example["question"]
            true_answer = normalize_answer(example["answer"])

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ]

            text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=768,
                    do_sample=False,
                )

            response  = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            pred      = extract_boxed(response)
            pred_norm = normalize_answer(pred)
            ok        = pred_norm == true_answer
            if ok:
                correct += 1

            # row styling
            icon      = "[bold green]✓[/]" if ok else "[bold red]✗[/]"
            row_style = "on grey11" if i % 2 == 0 else ""
            pred_text = Text(truncate(pred_norm or "[dim italic](none)[/dim italic]"), style="green" if ok else "red")
            true_text = Text(truncate(true_answer))
            result    = Text("CORRECT", style="bold green") if ok else Text("wrong", style="bold red")

            table.add_row(
                str(i + 1),
                icon,
                pred_text,
                true_text,
                result,
                style=row_style,
            )

            progress.advance(task_id)

            layout["top"].update(
                Panel(
                    table,
                    title=f"[bold bright_white]Results[/]",
                    border_style="bright_black",
                    padding=(0, 1),
                )
            )
            layout["bottom"].update(
                Columns(
                    [score_panel(correct, i + 1), progress],
                    equal=False,
                    expand=True,
                )
            )

    # ── final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start
    pct     = correct / total

    if pct >= 0.8:
        grade_colour = "bright_green"
        grade_label  = "Excellent"
    elif pct >= 0.6:
        grade_colour = "yellow"
        grade_label  = "Good"
    elif pct >= 0.4:
        grade_colour = "bright_yellow"
        grade_label  = "Fair"
    else:
        grade_colour = "red"
        grade_label  = "Needs Work"

    console.print()
    console.print(Rule("[bold bright_cyan]Summary[/]", style="bright_cyan"))

    summary = Table.grid(padding=(0, 4))
    summary.add_column(style="bold dim", justify="right")
    summary.add_column(style="bright_white")
    summary.add_row("Model",    MODEL_NAME)
    summary.add_row("Dataset",  "MATH-500 (test)")
    summary.add_row("Problems", str(total))
    summary.add_row("Correct",  f"[bold bright_green]{correct}[/]")
    summary.add_row("Wrong",    f"[bold red]{total - correct}[/]")
    summary.add_row("Accuracy", f"[bold {grade_colour}]{pct:.2%}[/]  [dim]({grade_label})[/dim]")
    summary.add_row("Elapsed",  f"{elapsed:.1f}s  [dim]({elapsed/total:.1f}s/problem)[/dim]")

    console.print(
        Panel(
            Align(summary, align="center"),
            border_style=grade_colour,
            title=f"[bold {grade_colour}]{grade_label}[/bold {grade_colour}]",
            padding=(1, 4),
        )
    )


if __name__ == "__main__":
    run_eval()
