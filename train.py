from __future__ import annotations

import argparse
from pathlib import Path

from llmrl.config import DEFAULT_ALGOS, DEFAULT_DATASET, DEFAULT_MODELS, RunConfig
from llmrl.runtime import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GRPO/PPO/DPPO on AIME 2025.")
    parser.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    parser.add_argument("--algo", choices=DEFAULT_ALGOS, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--rollout-group-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--dppo-delta", type=float, default=0.03)
    parser.add_argument("--dppo-approx", choices=("binary", "topk"), default="topk")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--train-examples-limit", type=int, default=None)
    parser.add_argument("--eval-examples-limit", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--delete-local-checkpoints", action="store_true")
    parser.add_argument("--resume", default="off")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--eval-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        status_file=args.status_file,
        output_root=args.output_root,
        checkpoints_root=args.checkpoints_root,
        eval_only=args.eval_only,
    )
    trainer = Trainer(config)
    if args.eval_only:
        trainer.compare_divergence()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
