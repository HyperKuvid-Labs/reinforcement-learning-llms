# Training Commands

This file is just the copy-paste version of the commands.

Use the repo venv:

```bash
bash install.sh
source .venv/bin/activate
tensorboard --logdir runs
```

## HRM

For `sapientinc/HRM-Text-1B`, use `lora` for now.

### HRM smoke test with TUI

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --finetune-method lora \
  --cpu-offload \
  --smoke-test \
  --tui
```

### HRM smoke test without TUI

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --finetune-method lora \
  --cpu-offload \
  --smoke-test
```

### HRM GRPO

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo grpo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96
```

### HRM GRPO with TUI

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo grpo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --tui
```

### HRM PPO

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo ppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1
```

### HRM PPO with TUI

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo ppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1 \
  --tui
```

### HRM DPPO top-k

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8
```

### HRM DPPO top-k with TUI

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8 \
  --tui
```

### HRM DPPO binary

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1 \
  --dppo-approx binary
```

### HRM tiny survival test

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --finetune-method lora \
  --cpu-offload \
  --rollout-group-size 1 \
  --max-new-tokens 16 \
  --train-examples-limit 1 \
  --ppo-epochs 1 \
  --topk 4 \
  --save-every 1
```

## LFM

For `LiquidAI/LFM2.5-1.2B-Thinking`, start with `qlora`.

### LFM smoke test with TUI

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --finetune-method qlora \
  --cpu-offload \
  --smoke-test \
  --tui
```

### LFM GRPO

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo grpo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96
```

### LFM PPO

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo ppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1
```

### LFM DPPO top-k

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8
```

### LFM DPPO top-k with TUI

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8 \
  --tui
```

### LFM DPPO binary

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --cpu-offload \
  --rollout-group-size 2 \
  --max-new-tokens 96 \
  --ppo-epochs 1 \
  --dppo-approx binary
```

## HF Push Variants

### HRM DPPO top-k with HF push

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --dppo-approx topk \
  --topk 8 \
  --save-every 20 \
  --push-to-hub \
  --delete-local-checkpoints
```

### HRM PPO with explicit HF repo name

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo ppo \
  --dataset-split train \
  --finetune-method lora \
  --cpu-offload \
  --push-to-hub \
  --hub-repo Pradheep1647/aime-2025-hrm-text-1b-ppo-rk2-lora-clip0p2
```

### LFM DPPO top-k with HF push

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset-split train \
  --finetune-method qlora \
  --cpu-offload \
  --dppo-approx topk \
  --topk 8 \
  --save-every 20 \
  --push-to-hub \
  --delete-local-checkpoints
```

## Eval / Divergence

### HRM divergence comparison

```bash
.venv/bin/python compare_divergence.py \
  --model sapientinc/HRM-Text-1B \
  --approx all \
  --topk 8
```

### LFM divergence comparison

```bash
.venv/bin/python compare_divergence.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --approx all \
  --topk 8
```

## Old queued TUI mode

This is the separate queue-style TUI, not the exact single-run `train.py --tui` path:

```bash
.venv/bin/python tui.py
```
