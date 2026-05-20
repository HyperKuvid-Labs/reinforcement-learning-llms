from __future__ import annotations

import json
from pathlib import Path


class TensorBoardLogger:
    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.log_dir = log_dir
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir))

    def add_scalars(self, step: int, metrics: dict[str, float]) -> None:
        if not self.writer:
            return
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if self.writer:
            self.writer.add_text(tag, text, step)

    def add_hparams_blob(self, config: dict[str, object]) -> None:
        self.add_text("run/config", json.dumps(config, indent=2, default=str))

    def flush(self) -> None:
        if self.writer:
            self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.close()
