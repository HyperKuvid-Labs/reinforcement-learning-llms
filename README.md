# Laptop RL Comparison on Easier Math

This repo is me trying to compare **GRPO**, **PPO**, and **DPPO** without making the setup heavier than the laptop can handle.

The first-principles setup is simple:
- take a small reasoning model
- ask it an easier math question
- sample a small group of rollouts
- parse the final answer
- give reward `1` if the answer matches, else `0`
- compare how the policy update rule behaves under that same verifier signal

So this is not a full RLHF stack. There is no learned reward model and no value critic here. The point is to keep the reward side fixed and compare the update behavior cleanly.

Default dataset:
- `openai/gsm8k`
- config: `main`
- split: `train`

Default models:
- `LiquidAI/LFM2.5-1.2B-Thinking`
- `sapientinc/HRM-Text-1B`
- `Qwen/Qwen3.5-4B-Base`

> This repo compares GRPO, PPO-style clipping, and DPPO-style divergence gating under the same deterministic verifier reward, without a learned reward model or value critic, so the comparison is intentionally about policy update behavior rather than full actor-critic RLHF PPO.

> The current laptop-first setup uses GSM8K instead of AIME because the 1B-class models need an easier reward surface before the RL loop gives useful signal.

## What Is Compared

Main comparison:
- `GRPO`
- `PPO`
- `DPPO` with `top-k` divergence approximation by default

Extra divergence study:
- `naive` full-vocab divergence, eval-only
- `binary` approximation
- `top-k` approximation

Reward is binary final-answer reward:
- extract the model final answer
- extract the gold answer, including GSM8K `####` answers
- normalize both
- reward = `1` if they match, else `0`

## Why These Models

The small-model comparison is intentional.

`LiquidAI/LFM2.5-1.2B-Thinking` is the diffusion-style small reasoning model. `sapientinc/HRM-Text-1B` is the hierarchical reasoning model. I want to see whether the same RL update behaves differently across these two model families before scaling the experiment back up.

`Qwen/Qwen3.5-4B-Base` stays as the optional stronger baseline, but it is no longer the laptop-first default path.

> HRM and LFM are not just two random 1B models here: HRM is a hierarchical reasoning model and LFM is a diffusion-based model, so the comparison is also about how different model families respond to the same verifier RL loop.

## Backends

`--trainer-backend auto` is the default.

In auto mode:
- HRM and LFM use the regular Transformers/PEFT path
- Qwen uses Unsloth, because that is the path that makes more sense for the optional 4B run

Stable finetune choices:
- LFM: `--finetune-method qlora`
- HRM: `--finetune-method lora`
- Qwen 4B: `--finetune-method qlora`

## Logging

Everything that matters should go to **TensorBoard**.

The TUI is only the live operator surface. TensorBoard is the actual source of truth for:
- training metrics
- eval metrics
- divergence stats
- checkpoint events
- upload events
- run config
- system stats

Run it with:

```bash
tensorboard --logdir runs
```

Representative tags:
- `train/loss`
- `train/reward_mean`
- `train/reward_std`
- `train/reward_nonzero_fraction`
- `train/advantage_mean`
- `train/advantage_std`
- `train/clip_fraction`
- `train/dppo_mask_fraction`
- `eval/accuracy`
- `eval/reward_mean`
- `eval/completion_length`
- `divergence/binary`
- `divergence/topk`
- `divergence/naive_tv`
- `divergence/naive_kl`
- `system/tokens_per_sec`
- `system/step_time`
- `system/gpu_mem_allocated`
- `system/gpu_mem_reserved`
- `system/checkpoint_upload_time`
- `system/checkpoint_prune_status`

## Setup

Bootstrap from scratch:

```bash
bash install.sh
```

Before training or divergence eval starts, the CLI asks for:
- `HF_USERNAME`
- `HF_TOKEN`

These get saved into a repo-local `.env` and reused later.

## Training

Laptop-first DPPO run:

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

HRM version:

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

`train.py --tui` launches the TUI for that exact run:

```bash
.venv/bin/python train.py \
  --model LiquidAI/LFM2.5-1.2B-Thinking \
  --algo grpo \
  --dataset openai/gsm8k \
  --dataset-config main \
  --finetune-method qlora \
  --rollout-group-size 2 \
  --max-new-tokens 64 \
  --train-examples-limit 200 \
  --tui
```

If `--hub-repo` is omitted, the trainer falls back to:

```text
<HF_USERNAME>/<dataset-slug>-<model-slug>-<algo-suffix>
```

Examples:

```text
yourname/gsm8k-main-lfm2-5-1-2b-thinking-grpo-rk2-qlora
yourname/gsm8k-main-hrm-text-1b-ppo-rk2-lora-clip0p2
yourname/gsm8k-main-lfm2-5-1-2b-thinking-dppo-rk2-qlora-topk-topk8-delta0p03
```

Checkpoints are pushed by default. Pass `--no-push-to-hub` only for local smoke/debug runs.

Copy-paste commands live in [TRAINING_COMMANDS.md](/home/pradheep/reinforcement-learning-llms/TRAINING_COMMANDS.md).

## AIME Later

AIME is still supported, but it is no longer the default because it was too sparse for the small-model laptop loop.

For AIME, pass:

```bash
--dataset test-time-compute/aime_2025 --dataset-config none
```

## Repo Layout

- `train.py`: main training CLI
- `compare_divergence.py`: eval-only divergence comparison
- `tui.py`: grey/white operator console
- `graphs.py`: plot TensorBoard traces
- `llmrl/`: training, auth, divergence, reward, and logging code
