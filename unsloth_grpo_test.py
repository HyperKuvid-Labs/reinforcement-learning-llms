import argparse
import json
import re
import threading
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from rich import box
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
from rich.table import Table
from rich.text import Text
from unsloth import FastLanguageModel, is_bfloat16_supported


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Unsloth GRPO adapter on GSM8K."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="gsm8k_grpo_qwen3b_final_unlsoth_trl",
        help="Path to the saved adapter directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="GSM8K split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Number of samples to evaluate (<=0 means full split).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Model max sequence length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for sampling when temperature > 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output path for detailed predictions JSON.",
    )
    return parser.parse_args()


def extract_boxed_answer(text: str) -> Optional[str]:
    match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_prediction_answer(text: str) -> Optional[str]:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    number_matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if number_matches:
        return number_matches[-1]
    return None


def extract_ground_truth(answer: str) -> str:
    return answer.split("####")[-1].strip()


def normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    normalized = answer.strip()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\s+", "", normalized)
    if normalized.endswith("."):
        normalized = normalized[:-1]
    return normalized


def build_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}"


class EvalState:
    def __init__(self, total: int) -> None:
        self.total = total
        self.step = 0
        self.correct = 0
        self.status = "starting"
        self.done = False
        self.started = time.monotonic()
        self.elapsed = 0.0
        self.current_index = -1
        self.current_question = ""
        self.current_ground_truth = ""
        self.current_rollout = ""
        self.current_prediction: Optional[str] = None
        self.current_correct: Optional[bool] = None
        self.recent: deque[dict] = deque(maxlen=8)
        self.lock = threading.Lock()


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _build_recent_table(state: EvalState) -> Table:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold bright_white",
        expand=True,
        pad_edge=False,
    )
    table.add_column("Idx", justify="right", style="dim")
    table.add_column("GT", style="bright_white")
    table.add_column("Pred", style="cyan")
    table.add_column("Correct", justify="center")
    for row in state.recent:
        ok_style = "bright_green" if row["correct"] else "red"
        table.add_row(
            str(row["index"]),
            row["ground_truth"],
            row["prediction"] if row["prediction"] is not None else "—",
            Text("✓" if row["correct"] else "✗", style=ok_style),
        )
    if not state.recent:
        table.add_row("—", "awaiting", "awaiting", "—")
    return table


def _build_rollout_panel(state: EvalState) -> Panel:
    with state.lock:
        idx = state.current_index
        question = state.current_question
        rollout = state.current_rollout

    question_text = Text()
    question_text.append("Q: ", style="bold bright_white")
    question_text.append(_clip(question, 260) if question else "awaiting sample...", style="white")

    rollout_text = Text()
    rollout_text.append("\n\nRollout:\n", style="bold bright_cyan")
    rollout_text.append(rollout if rollout else "waiting for generation...", style="cyan")

    body = Text.assemble(question_text, rollout_text)
    title = f"[bold]current rollout[/bold]  [dim]sample {idx if idx >= 0 else '—'}[/dim]"
    return Panel(body, title=title, border_style="bright_blue", padding=(1, 1))


def _build_prediction_panel(state: EvalState) -> Panel:
    with state.lock:
        gt = state.current_ground_truth
        pred = state.current_prediction
        correct = state.current_correct
        status = state.status

    verdict = "—"
    verdict_style = "yellow"
    if correct is True:
        verdict = "✓ Correct"
        verdict_style = "bright_green"
    elif correct is False:
        verdict = "✗ Incorrect"
        verdict_style = "bright_red"

    tbl = Table(box=box.SIMPLE, show_header=False, expand=True, pad_edge=False)
    tbl.add_column(style="dim", ratio=2)
    tbl.add_column(style="white", ratio=5)
    tbl.add_row("Status", Text(status, style="bright_yellow"))
    tbl.add_row("GT", Text(gt if gt else "—", style="bright_white"))
    tbl.add_row("Pred", Text(pred if pred else "—", style="bright_cyan"))
    tbl.add_row("Result", Text(verdict, style=verdict_style))

    return Panel(tbl, title="[bold]prediction[/bold]", border_style="green", padding=(1, 1))


def _build_ui(state: EvalState, progress: Progress) -> Layout:
    with state.lock:
        step = state.step
        total = state.total
        correct = state.correct
        elapsed = timedelta(seconds=int(state.elapsed))
        status = state.status

    accuracy = (correct / step) if step > 0 else 0.0
    speed = (step / state.elapsed) if state.elapsed > 0 else 0.0

    summary = Table(box=None, show_header=False, expand=True, pad_edge=False)
    summary.add_column(ratio=3)
    summary.add_column(ratio=2)
    summary.add_column(ratio=2)
    summary.add_column(ratio=2)
    summary.add_row(
        Text("GSM8K Evaluation · Unsloth GRPO", style="bold bright_white"),
        Text(f"sample {step}/{total}", style="cyan"),
        Text(f"acc {accuracy * 100:.2f}%", style="bright_green"),
        Text(f"speed {speed:.2f}/s", style="yellow"),
    )
    summary.add_row(
        Text("", style="white"),
        Text(f"status {status}", style="bright_yellow"),
        Text(f"correct {correct}", style="bright_green"),
        Text(f"wrong {max(0, step - correct)}", style="bright_red"),
    )

    header = Panel(
        summary,
        border_style="bright_blue",
        subtitle=f"[dim]elapsed {elapsed}[/dim]",
    )

    rollout_panel = _build_rollout_panel(state)
    prediction_panel = _build_prediction_panel(state)
    recent_panel = Panel(
        _build_recent_table(state),
        title="[bold]recent predictions[/bold]",
        border_style="magenta",
        padding=(0, 1),
    )

    progress_panel = Panel(progress, border_style="dim", padding=(0, 1))

    layout = Layout()
    layout.split_column(
        Layout(header, size=5),
        Layout(name="body", ratio=3),
        Layout(recent_panel, ratio=2),
        Layout(progress_panel, size=4),
    )
    layout["body"].split_row(
        Layout(rollout_panel, ratio=3),
        Layout(prediction_panel, ratio=2),
    )
    return layout


def _live_loop(state: EvalState, progress: Progress, task_id: int, console: Console) -> None:
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            with state.lock:
                state.elapsed = time.monotonic() - state.started
                step = state.step
                total = state.total
                done = state.done

            progress.update(
                task_id,
                completed=step,
                total=max(1, total),
                description=f"evaluating {step}/{total}",
            )
            live.update(_build_ui(state, progress))

            if done:
                live.update(_build_ui(state, progress))
                break
            time.sleep(0.25)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    total = len(dataset)
    exact_correct = 0

    rows = []

    console = Console()
    state = EvalState(total=total)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )
    task_id = progress.add_task("evaluating", total=max(1, total))

    live_thread = threading.Thread(
        target=_live_loop,
        args=(state, progress, task_id, console),
        daemon=True,
    )
    live_thread.start()

    for idx, sample in enumerate(dataset):
        question = sample["question"]
        ground_truth = extract_ground_truth(sample["answer"])

        with state.lock:
            state.current_index = idx
            state.current_question = question
            state.current_ground_truth = ground_truth
            state.current_rollout = ""
            state.current_prediction = None
            state.current_correct = None
            state.status = "generating"

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = args.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        pred_answer = extract_prediction_answer(completion)
        gt_norm = normalize_answer(ground_truth)
        pred_norm = normalize_answer(pred_answer)

        is_correct = pred_norm == gt_norm and pred_norm != ""
        if is_correct:
            exact_correct += 1

        rows.append(
            {
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": pred_answer,
                "prediction_text": completion,
                "correct": is_correct,
            }
        )

        with state.lock:
            state.step = idx + 1
            state.correct = exact_correct
            state.current_rollout = completion
            state.current_prediction = pred_answer
            state.current_correct = is_correct
            state.status = "evaluating"
            state.recent.append(
                {
                    "index": idx,
                    "ground_truth": ground_truth,
                    "prediction": pred_answer,
                    "correct": is_correct,
                }
            )

    exact_acc = exact_correct / total if total else 0.0

    with state.lock:
        state.status = "done"
        state.done = True

    live_thread.join(timeout=3)

    print("\n=== Evaluation Summary ===")
    print(f"Samples:         {total}")
    print(f"Exact Match:     {exact_acc:.4f} ({exact_correct}/{total})")

    print("\n=== Example Predictions (first 5) ===")
    for row in rows[:5]:
        print(f"\n[{row['index']}] Q: {row['question']}")
        print(f"GT:   {row['ground_truth']}")
        print(f"PRED: {row['prediction']}")
        print(f"OK:   {row['correct']}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": args.model_path,
            "split": args.split,
            "samples": total,
            "exact_match": exact_acc,
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved detailed results to: {out_path}")


if __name__ == "__main__":
    main()