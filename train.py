from __future__ import annotations

import argparse
import os
from pathlib import Path
import traceback

from llmrl.config import DEFAULT_ALGOS, DEFAULT_DATASET, DEFAULT_MODELS, RunConfig
from llmrl.runtime import Trainer
from tui import run_config_with_tui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GRPO/PPO/DPPO on AIME 2025.")
    parser.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    parser.add_argument("--algo", choices=DEFAULT_ALGOS, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--rollout-group-size", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--dppo-delta", type=float, default=0.03)
    parser.add_argument("--dppo-approx", choices=("binary", "topk"), default="topk")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--train-examples-limit", type=int, default=None)
    parser.add_argument("--eval-examples-limit", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--delete-local-checkpoints", action="store_true")
    parser.add_argument("--resume", default="off")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--trainer-backend", choices=("unsloth", "transformers"), default="unsloth")
    parser.add_argument("--finetune-method", choices=("full", "lora", "qlora"), default="qlora")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--reward-debug-every", type=int, default=1)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--offload-root", type=Path, default=Path(".offload"))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--tui", action="store_true")
    return parser


def apply_smoke_preset(args) -> None:
    args.finetune_method = "qlora"
    args.cpu_offload = True
    args.rollout_group_size = 1
    args.max_prompt_tokens = min(args.max_prompt_tokens, 384)
    args.max_new_tokens = min(args.max_new_tokens, 32)
    args.ppo_epochs = 1
    args.topk = min(args.topk, 4)
    args.train_examples_limit = 2 if args.train_examples_limit is None else min(args.train_examples_limit, 2)
    args.eval_examples_limit = 2 if args.eval_examples_limit is None else min(args.eval_examples_limit, 2)
    args.save_every = min(args.save_every, 2)
    args.reward_debug_every = 1


def make_cli_callback():
    def _callback(payload: dict[str, object]) -> None:
        phase = str(payload.get("phase", "unknown"))
        step = int(payload.get("global_step", 0))
        if phase == "auth":
            print("[auth] checking Hugging Face credentials", flush=True)
        elif phase == "load_start":
            print(
                f"[load] starting | backend={payload.get('trainer_backend')} finetune={payload.get('finetune_method')} "
                f"offload={payload.get('cpu_offload')} split={payload.get('dataset_split')}",
                flush=True,
            )
        elif phase == "tokenizer_loaded":
            print("[load] tokenizer ready", flush=True)
        elif phase == "model_loaded":
            print(
                f"[load] model ready | cuda_available={payload.get('cuda_available')} "
                f"device={payload.get('model_device')}",
                flush=True,
            )
        elif phase == "adapter_ready":
            print(
                f"[load] adapters ready | method={payload.get('finetune_method')} "
                f"trainable_params={payload.get('trainable_params')}",
                flush=True,
            )
        elif phase == "hparam_adjustment":
            print(
                f"[hparam] {payload.get('field')} {payload.get('old_value')} -> {payload.get('new_value')} "
                f"| {payload.get('reason')}",
                flush=True,
            )
        elif phase == "dataset_loading":
            print(f"[data] loading {payload.get('dataset')} [{payload.get('split')}]", flush=True)
        elif phase == "dataset_loaded":
            print(f"[data] loaded {payload.get('examples')} examples", flush=True)
        elif phase == "optimizer_ready":
            print(f"[optim] ready | lr={payload.get('learning_rate')}", flush=True)
        elif phase == "resumed":
            print(f"[resume] from {payload.get('resume_path')} @ step {payload.get('global_step')}", flush=True)
        elif phase == "ready":
            print(f"[run] ready | logdir={payload.get('run_dir')}", flush=True)
        elif phase == "sampling":
            print(f"[step {step}] sampling", flush=True)
        elif phase == "rollout_ready":
            print(
                f"[step {step}] rollout ready | tokens={payload.get('completion_tokens')} "
                f"group={payload.get('group_size')}",
                flush=True,
            )
        elif phase == "sample_debug":
            print(
                f"[step {step}] sample | reward={payload.get('reward')} "
                f"pred={payload.get('pred_answer')} gold={payload.get('gold_answer')} "
                f"text={payload.get('completion_preview')}",
                flush=True,
            )
        elif phase == "backward":
            print(f"[step {step}] backward | ppo_epoch={payload.get('ppo_epoch')}", flush=True)
        elif phase == "stats_ready":
            print(
                f"[step {step}] stats ready | seq_steps={payload.get('sequence_steps')} "
                f"micro={payload.get('micro_batch_size', '-')}x{payload.get('micro_batches', '-')}",
                flush=True,
            )
        elif phase == "saving":
            print(f"[step {step}] saving checkpoint", flush=True)
        elif phase == "uploading":
            print(f"[step {step}] uploading checkpoint", flush=True)
        elif phase == "adapter_fallback":
            print(
                f"[adapter] fallback {payload.get('requested_method')} -> {payload.get('fallback_method')} "
                f"| {payload.get('reason')}",
                flush=True,
            )
        elif phase == "running":
            print(
                f"[step {step}] reward={float(payload.get('train/reward_mean', 0.0)):.4f} "
                f"loss={float(payload.get('train/loss', 0.0)):.4f} "
                f"grad={float(payload.get('train/grad_norm', 0.0)):.4f} "
                f"tps={float(payload.get('system/tokens_per_sec', 0.0)):.2f}",
                flush=True,
            )
        elif phase == "done":
            print(f"[done] total_steps={payload.get('total_steps', step)}", flush=True)
        elif phase == "error":
            print(f"[error] {payload.get('error')}", flush=True)
            tb = str(payload.get("traceback", "")).strip()
            if tb:
                print(tb, flush=True)
    return _callback


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.smoke_test:
        apply_smoke_preset(args)
    config = RunConfig(
        model_id=args.model,
        algo=args.algo,
        dataset_id=args.dataset,
        dataset_split=args.dataset_split,
        rollout_group_size=args.rollout_group_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        ppo_clip_eps=args.ppo_clip_eps,
        ppo_epochs=args.ppo_epochs,
        dppo_delta=args.dppo_delta,
        dppo_approx=args.dppo_approx,
        dppo_topk=args.topk,
        train_examples_limit=args.train_examples_limit,
        eval_examples_limit=args.eval_examples_limit,
        save_every=args.save_every,
        log_every=args.log_every,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
        delete_local_checkpoints=args.delete_local_checkpoints,
        resume=args.resume,
        cpu_offload=args.cpu_offload,
        trainer_backend=args.trainer_backend,
        finetune_method=args.finetune_method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        smoke_test=args.smoke_test,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        reward_debug_every=args.reward_debug_every,
        status_file=args.status_file,
        output_root=args.output_root,
        checkpoints_root=args.checkpoints_root,
        offload_root=args.offload_root,
        eval_only=args.eval_only,
    )
    if args.eval_only:
        trainer = Trainer(config, callback=make_cli_callback())
        try:
            trainer.compare_divergence()
        except Exception:
            print(traceback.format_exc(), flush=True)
            raise
    elif args.tui:
        run_config_with_tui(config)
    else:
        trainer = Trainer(config, callback=make_cli_callback())
        try:
            trainer.train()
        except Exception:
            print(traceback.format_exc(), flush=True)
            raise


if __name__ == "__main__":
    main()
