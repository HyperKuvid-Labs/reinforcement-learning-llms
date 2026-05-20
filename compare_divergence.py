from __future__ import annotations

import argparse

from llmrl.config import DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, DEFAULT_MODELS, RunConfig
from llmrl.runtime import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive, binary, and top-k divergence approximations.")
    parser.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default="auto")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--approx", choices=("all", "binary", "topk", "naive"), default="all")
    parser.add_argument("--topk", type=int, default=16)
    args = parser.parse_args()
    dataset_config = args.dataset_config.strip()
    if dataset_config.lower() == "auto":
        dataset_config = DEFAULT_DATASET_CONFIG if args.dataset == DEFAULT_DATASET else None
    elif dataset_config.lower() in {"", "none", "null"}:
        dataset_config = None

    config = RunConfig(
        model_id=args.model,
        algo="dppo",
        dataset_id=args.dataset,
        dataset_config=dataset_config,
        dataset_split=args.dataset_split,
        dppo_approx="topk",
        dppo_topk=args.topk,
        eval_only=True,
    )
    Trainer(config).compare_divergence()


if __name__ == "__main__":
    main()
