from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re


DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_MODELS = (
    "LiquidAI/LFM2.5-1.2B-Thinking",
    "sapientinc/HRM-Text-1B",
    "Qwen/Qwen3.5-4B-Base",
)
DEFAULT_ALGOS = ("grpo", "ppo", "dppo")


def _slugify(value: str) -> str:
    slug = value.lower().replace("_", "-").replace(".", "-").replace("/", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    gradient_checkpointing: bool = True


@dataclass(slots=True)
class RunConfig:
    model_id: str
    algo: str
    dataset_id: str = DEFAULT_DATASET
    dataset_config: str | None = DEFAULT_DATASET_CONFIG
    dataset_split: str = "train"
    output_root: Path = Path("runs")
    checkpoints_root: Path = Path("checkpoints")
    offload_root: Path = Path(".offload")
    seed: int = 7
    rollout_group_size: int = 2
    max_prompt_tokens: int = 512
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    ppo_clip_eps: float = 0.2
    dppo_delta: float = 0.03
    dppo_approx: str = "topk"
    dppo_topk: int = 8
    ppo_epochs: int = 1
    train_examples_limit: int | None = None
    eval_examples_limit: int | None = 16
    save_every: int = 20
    log_every: int = 1
    push_to_hub: bool = True
    hub_repo: str | None = None
    delete_local_checkpoints: bool = False
    resume: str = "off"
    cpu_offload: bool = False
    trainer_backend: str = "auto"
    finetune_method: str = "qlora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    smoke_test: bool = False
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    reward_debug_every: int = 1
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    status_file: Path | None = None
    eval_only: bool = False

    @property
    def dataset_slug(self) -> str:
        parts = [self.dataset_id.split("/")[-1], self.dataset_config]
        return _slugify("-".join(part for part in parts if part))

    @property
    def model_slug(self) -> str:
        return _slugify(self.model_id.split("/")[-1])

    @property
    def algo_suffix(self) -> str:
        parts = [self.algo, f"rk{self.rollout_group_size}", self.finetune_method]
        if self.algo == "dppo":
            parts.extend(
                [
                    self.dppo_approx,
                    f"topk{self.dppo_topk}" if self.dppo_approx == "topk" else None,
                    f"delta{str(self.dppo_delta).replace('.', 'p')}",
                ]
            )
        elif self.algo == "ppo":
            parts.append(f"clip{str(self.ppo_clip_eps).replace('.', 'p')}")
        return "-".join(part for part in parts if part)

    @property
    def slug(self) -> str:
        return f"{self.dataset_slug}-{self.model_slug}-{self.algo_suffix}-{self.run_name}"

    def default_hub_repo(self, hf_username: str) -> str:
        return f"{hf_username}/{self.dataset_slug}-{self.model_slug}-{self.algo_suffix}"

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.slug

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoints_root / self.slug
