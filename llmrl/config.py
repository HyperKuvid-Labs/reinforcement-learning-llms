from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


DEFAULT_DATASET = "test-time-compute/aime_2025"
DEFAULT_MODELS = (
    "sapientinc/HRM-Text-1B",
    "LiquidAI/LFM2.5-1.2B-Thinking",
)
DEFAULT_ALGOS = ("grpo", "ppo", "dppo")


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
    dataset_split: str = "test"
    output_root: Path = Path("runs")
    checkpoints_root: Path = Path("checkpoints")
    seed: int = 7
    rollout_group_size: int = 4
    max_prompt_tokens: int = 1024
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    ppo_clip_eps: float = 0.2
    dppo_delta: float = 0.03
    dppo_approx: str = "topk"
    dppo_topk: int = 16
    ppo_epochs: int = 2
    train_examples_limit: int | None = None
    eval_examples_limit: int | None = 16
    save_every: int = 20
    log_every: int = 1
    push_to_hub: bool = False
    hub_repo: str | None = None
    delete_local_checkpoints: bool = False
    resume: str = "off"
    cpu_offload: bool = True
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    status_file: Path | None = None
    eval_only: bool = False

    @property
    def slug(self) -> str:
        model_slug = self.model_id.split("/")[-1].lower().replace(".", "-")
        if self.algo == "dppo":
            return f"{model_slug}-{self.algo}-{self.dppo_approx}-{self.run_name}"
        return f"{model_slug}-{self.algo}-{self.run_name}"

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.slug

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoints_root / self.slug
