#!/usr/bin/env bash
set -euo pipefail

uv venv
. .venv/bin/activate
uv pip install --upgrade pip
uv pip install \
  "torch>=2.4.0" \
  datasets \
  accelerate \
  bitsandbytes \
  huggingface_hub \
  peft \
  tensorboard \
  rich \
  plotly \
  pandas
uv pip install --upgrade "git+https://github.com/huggingface/transformers.git@main"
