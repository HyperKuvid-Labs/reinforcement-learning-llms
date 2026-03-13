import argparse
import json
import re
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
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
    format_ok = 0
    boxed_found = 0

    rows = []

    print(f"Evaluating {total} samples from GSM8K/{args.split}")
    print(f"Model path: {args.model_path}")

    for idx, sample in enumerate(dataset):
        question = sample["question"]
        ground_truth = extract_ground_truth(sample["answer"])

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = args.temperature > 0.0
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        pred_boxed = extract_boxed_answer(completion)
        gt_norm = normalize_answer(ground_truth)
        pred_norm = normalize_answer(pred_boxed)

        has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        has_boxed = pred_boxed is not None

        if has_boxed:
            boxed_found += 1
        if has_think and has_boxed:
            format_ok += 1

        is_correct = pred_norm == gt_norm and pred_norm != ""
        if is_correct:
            exact_correct += 1

        rows.append(
            {
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "prediction_boxed": pred_boxed,
                "prediction_text": completion,
                "correct": is_correct,
                "has_think": has_think,
                "has_boxed": has_boxed,
            }
        )

        if (idx + 1) % 20 == 0 or (idx + 1) == total:
            running_acc = exact_correct / (idx + 1)
            print(
                f"[{idx + 1:>5}/{total}] exact={running_acc:.4f} "
                f"boxed={boxed_found/(idx + 1):.4f} format={format_ok/(idx + 1):.4f}"
            )

    exact_acc = exact_correct / total if total else 0.0
    boxed_rate = boxed_found / total if total else 0.0
    format_rate = format_ok / total if total else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Samples:         {total}")
    print(f"Exact Match:     {exact_acc:.4f} ({exact_correct}/{total})")
    print(f"Boxed Found:     {boxed_rate:.4f} ({boxed_found}/{total})")
    print(f"Format Success:  {format_rate:.4f} ({format_ok}/{total})")

    print("\n=== Example Predictions (first 5) ===")
    for row in rows[:5]:
        print(f"\n[{row['index']}] Q: {row['question']}")
        print(f"GT:   {row['ground_truth']}")
        print(f"PRED: {row['prediction_boxed']}")
        print(f"OK:   {row['correct']}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": args.model_path,
            "split": args.split,
            "samples": total,
            "exact_match": exact_acc,
            "boxed_rate": boxed_rate,
            "format_success_rate": format_rate,
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved detailed results to: {out_path}")


if __name__ == "__main__":
    main()