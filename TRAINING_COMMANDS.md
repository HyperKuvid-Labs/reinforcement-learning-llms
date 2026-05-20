# Training Commands

This is the copy-paste sheet for the runs that actually make sense here.

## Setup

```bash
bash install.sh
source .venv/bin/activate
tensorboard --logdir runs
```

## Recommended Models

Primary model:
- `Qwen/Qwen3.5-4B-Base`

Secondary models:
- `nvidia/AceReason-Nemotron-1.1-7B`
- `Skywork/Skywork-OR1-Math-7B`

Optional third model:
- `nvidia/AceMath-RL-Nemotron-7B`

The Qwen 4B base model is the first thing I want to train now. The 7B models stay here as comparison points.

## Stable Finetune Defaults

If you want the highest chance of a clean run:
- `Qwen/Qwen3.5-4B-Base` -> `--finetune-method qlora`
- `nvidia/AceReason-Nemotron-1.1-7B` -> `--finetune-method qlora`
- `Skywork/Skywork-OR1-Math-7B` -> `--finetune-method qlora`
- `nvidia/AceMath-RL-Nemotron-7B` -> `--finetune-method qlora`

If you are on a strong `48 GB` GPU and want to push harder, you can try `--finetune-method full`, but `qlora` is still the safer baseline.

## Smoke Test

Use this first before any real run:

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --smoke-test
```

TUI version:

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --smoke-test \
  --tui
```

## Main Runs: Qwen3.5-4B-Base

### GRPO

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo grpo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128
```

### PPO

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo ppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1
```

### DPPO top-k

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16
```

### DPPO binary

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx binary
```

### DPPO top-k with TUI

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16 \
  --tui
```

## Main Runs: AceReason-Nemotron-1.1-7B

### GRPO

```bash
.venv/bin/python train.py \
  --model nvidia/AceReason-Nemotron-1.1-7B \
  --algo grpo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128
```

### PPO

```bash
.venv/bin/python train.py \
  --model nvidia/AceReason-Nemotron-1.1-7B \
  --algo ppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1
```

### DPPO top-k

```bash
.venv/bin/python train.py \
  --model nvidia/AceReason-Nemotron-1.1-7B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16
```

### DPPO binary

```bash
.venv/bin/python train.py \
  --model nvidia/AceReason-Nemotron-1.1-7B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx binary
```

## Main Runs: Skywork-OR1-Math-7B

### GRPO

```bash
.venv/bin/python train.py \
  --model Skywork/Skywork-OR1-Math-7B \
  --algo grpo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128
```

### PPO

```bash
.venv/bin/python train.py \
  --model Skywork/Skywork-OR1-Math-7B \
  --algo ppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1
```

### DPPO top-k

```bash
.venv/bin/python train.py \
  --model Skywork/Skywork-OR1-Math-7B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16
```

### DPPO binary

```bash
.venv/bin/python train.py \
  --model Skywork/Skywork-OR1-Math-7B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx binary
```

## Optional Third Model: AceMath-RL-Nemotron-7B

### DPPO top-k

```bash
.venv/bin/python train.py \
  --model nvidia/AceMath-RL-Nemotron-7B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16
```

## 48 GB GPU Variants

These are the commands I would use on an `L40S`-class `48 GB` GPU if I want a stronger run than the laptop-safe defaults.

### Qwen DPPO top-k, QLoRA

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16
```

### Qwen DPPO top-k, full finetune

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method full \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16
```

### AceReason DPPO top-k, full finetune

```bash
.venv/bin/python train.py \
  --model nvidia/AceReason-Nemotron-1.1-7B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method full \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16
```

### Skywork DPPO top-k, full finetune

```bash
.venv/bin/python train.py \
  --model Skywork/Skywork-OR1-Math-7B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method full \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16
```

If full finetune starts getting unstable, go straight back to `qlora`.

## Hugging Face Push

### Qwen DPPO top-k with push

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16 \
  --save-every 20 \
  --delete-local-checkpoints
```

### Explicit repo name

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 4 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 16 \
  --hub-repo yourname/aime-2025-qwen-qwen3-5-4b-base-dppo-rk4-qlora-topk16
```

## Divergence Eval

### Qwen

```bash
.venv/bin/python compare_divergence.py \
  --model Qwen/Qwen3.5-4B-Base \
  --approx all \
  --topk 16
```

### AceReason

```bash
.venv/bin/python compare_divergence.py \
  --model nvidia/AceReason-Nemotron-1.1-7B \
  --approx all \
  --topk 16
```

### Skywork

```bash
.venv/bin/python compare_divergence.py \
  --model Skywork/Skywork-OR1-Math-7B \
  --approx all \
  --topk 16
```

## Notes

- `train.py --tui` runs the exact chosen config inside the TUI.
- `train.py` without `--tui` is the better path for debugging crashes.
- The CLI will ask for `HF_USERNAME` and `HF_TOKEN` if they are missing, then save them into `.env`.
- TensorBoard is the real source of truth. The TUI is just the live operator surface.
