# Training Commands

This is the copy-paste sheet for the laptop-first runs.

The current default dataset is `openai/gsm8k` with config `main`. That is intentional: AIME was too sparse for 1B-class models, so GSM8K is the easier reward surface for debugging the RL loop.

## Setup

```bash
bash install.sh
source .venv/bin/activate
tensorboard --logdir runs
```

## Model Choices

Primary laptop models:
- `LiquidAI/LFM2.5-1.2B-Thinking`
- `sapientinc/HRM-Text-1B`

Optional stronger baseline:
- `Qwen/Qwen3.5-4B-Base`

Stable finetune choices:
- LFM -> `--finetune-method qlora`
- HRM -> `--finetune-method lora`
- Qwen 4B -> `--finetune-method qlora`

Backend default:
- `--trainer-backend auto`
- auto uses Transformers for HRM/LFM
- auto uses Unsloth for Qwen

Important: do not use `--rollout-group-size 1` for real training. The group advantage becomes zero. Use `1` only for smoke tests.

## Smoke Tests

LFM smoke test:

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method qlora \
  --smoke-test \
  --no-push-to-hub
```

HRM smoke test:

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method lora \
  --smoke-test \
  --no-push-to-hub
```

## LFM2.5-1.2B-Thinking

### GRPO

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo grpo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

### PPO

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo ppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --ppo-epochs 1 \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

### DPPO top-k

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8 \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

### DPPO binary

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --ppo-epochs 1 \
  --dppo-approx binary \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

## HRM-Text-1B

### GRPO

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo grpo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method lora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

### PPO

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo ppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method lora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --ppo-epochs 1 \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

### DPPO top-k

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method lora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8 \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

### DPPO binary

```bash
.venv/bin/python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method lora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --ppo-epochs 1 \
  --dppo-approx binary \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

## TUI For Exact Run

Add `--tui` to any exact training command.

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8 \
  --micro-batch-size 1 \
  --train-examples-limit 200 \
  --tui
```

## Optional Qwen 4B Baseline

Use this only if the laptop has enough headroom, or move it to a 24 GB+ GPU.

```bash
.venv/bin/python train.py \
  --model Qwen/Qwen3.5-4B-Base \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 2 \
  --max-prompt-tokens 384 \
  --max-new-tokens 64 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8 \
  --micro-batch-size 1 \
  --train-examples-limit 200
```

## Divergence Eval

LFM:

```bash
.venv/bin/python compare_divergence.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split test \
  --approx all \
  --topk 8
```

HRM:

```bash
.venv/bin/python compare_divergence.py \
  --model sapientinc/HRM-Text-1B \
  --dataset openai/gsm8k \
  --dataset-config main \
  --dataset-split test \
  --approx all \
  --topk 8
```

## Hugging Face Push

Training pushes checkpoints by default.

Default repo naming:

```text
<HF_USERNAME>/<dataset-slug>-<model-slug>-<algo-suffix>
```

Examples:

```text
yourname/gsm8k-main-lfm2-5-1-2b-thinking-grpo-rk2-qlora
yourname/gsm8k-main-hrm-text-1b-ppo-rk2-lora-clip0p2
yourname/gsm8k-main-lfm2-5-1-2b-thinking-dppo-rk2-qlora-topk-topk8-delta0p03
```

Use an explicit repo if needed:

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --finetune-method qlora \
  --hub-repo yourname/gsm8k-lfm-dppo-topk8
```

## AIME Later

When we want to go back to AIME, use:

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-config none \
  --dataset-split train \
  --finetune-method qlora \
  --rollout-group-size 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 128 \
  --ppo-epochs 1 \
  --dppo-approx topk \
  --topk 8 \
  --micro-batch-size 1
```

## Notes

- `train.py --tui` runs the exact chosen config inside the TUI.
- `train.py` without `--tui` is the better path for debugging crashes.
- The CLI asks for `HF_USERNAME` and `HF_TOKEN` if missing, then saves them into `.env`.
- TensorBoard is the real source of truth. The TUI is just the live operator surface.
