# AIME 2025 RL Comparison for 1B LLMs

This repo is basically me trying to compare **GRPO**, **PPO**, and **DPPO** on the same math setup, without hiding the core assumption.

The setup is simple from first principles:
- take a reasoning model
- ask it an AIME 2025 question
- sample rollouts
- check whether the final answer is actually correct
- give reward `1` if correct, else `0`
- then compare how different policy update rules behave under that same signal

So the point here is not “best possible full RLHF stack”. The point is to hold the reward side fixed and compare the optimization behavior cleanly.

This repo uses:
- `nvidia/AceReason-Nemotron-1.1-7B`
- `Skywork/Skywork-OR1-Math-7B`
- `nvidia/AceMath-RL-Nemotron-7B`

The dataset is:
- `test-time-compute/aime_2025`

> This repo compares GRPO, PPO-style clipping, and DPPO-style divergence gating under the same deterministic verifier reward, without a learned reward model or value critic, so the comparison is intentionally about policy update behavior rather than full actor-critic RLHF PPO.

## What Is Actually Being Compared

Main comparison:
- `GRPO`
- `PPO`
- `DPPO` with `top-k` divergence approximation by default

Extra divergence study:
- `naive` exact divergence, eval-only
- `binary` approximation
- `top-k` approximation

Reward is binary-answer reward on completions:
- extract the final answer
- normalize it
- compare against the gold AIME answer
- reward = `1` if it matches, else `0`

So all three methods are seeing the same reward function. That is the whole point of this repo.

## Why These Two Models

- `nvidia/AceReason-Nemotron-1.1-7B`
- `Skywork/Skywork-OR1-Math-7B`
- `nvidia/AceMath-RL-Nemotron-7B`

This pair is intentional.

`AceReason-Nemotron-1.1-7B` is the primary model here because it is a strong 7B reasoning model. `Skywork-OR1-Math-7B` is the second comparison point, and `AceMath-RL-Nemotron-7B` is the optional third check. So I’m comparing algorithms, but I’m also comparing how the same RL-style update rules behave across different reasoning-model families on the same AIME setup.

> The model set is intentional: AceReason-Nemotron-1.1-7B is the primary reasoning model, Skywork-OR1-Math-7B is the second comparison point, and AceMath-RL-Nemotron-7B is the optional third check, so the repo compares both algorithm behavior and cross-model training behavior on the same AIME setup.

Notes from upstream model cards:
- `nvidia/AceReason-Nemotron-1.1-7B` is a 7B reasoning model that should work with the normal Transformers-style text generation flow.
- `Skywork/Skywork-OR1-Math-7B` is another math reasoning model in the same size band.
- `nvidia/AceMath-RL-Nemotron-7B` is the optional third comparison model.

Sources:
- https://huggingface.co/nvidia/AceReason-Nemotron-1.1-7B
- https://huggingface.co/Skywork/Skywork-OR1-Math-7B
- https://huggingface.co/nvidia/AceMath-RL-Nemotron-7B
- https://huggingface.co/datasets/test-time-compute/aime_2025

## Logging

Everything that matters should go to **TensorBoard**.

The TUI is just the live operator surface. TensorBoard is the actual source of truth for:
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

The local environment here previously had a broken CUDA-linked PyTorch install, so if `torch` still fails to import because of missing CUDA shared libraries, reinstall PyTorch for your exact CUDA version.

Before training or divergence eval starts, the CLI asks for:
- `HF_USERNAME`
- `HF_TOKEN`

These get saved into a repo-local `.env` and reused later.

## Training

Single run example:

```bash
python train.py \
  --model nvidia/AceReason-Nemotron-1.1-7B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dataset-split train \
  --finetune-method qlora \
  --dppo-approx topk \
  --topk 8 \
  --save-every 20 \
  --hub-repo your-user/your-repo \
  --delete-local-checkpoints \
  --cpu-offload
```

This is now intentionally laptop-biased by default:
- `qlora` is the default finetune mode
- rollout group defaults to `2`
- `max_new_tokens` defaults to `96`
- PPO inner epochs default to `1`
- `top-k` default is `8`
- checkpoints are pushed by default unless you pass `--no-push-to-hub`

If you just want to see whether the pipeline survives end to end, use the smoke preset first:

```bash
python train.py \
  --model nvidia/AceReason-Nemotron-1.1-7B \
  --algo dppo \
  --smoke-test
```

If `--hub-repo` is omitted, the trainer falls back to:

```text
<HF_USERNAME>/<dataset-slug>-<model-slug>-<algo-suffix>
```

Examples:

```text
yourname/aime-2025-hrm-text-1b-grpo-rk2-qlora
yourname/aime-2025-hrm-text-1b-ppo-rk2-qlora-clip0p2
yourname/aime-2025-hrm-text-1b-dppo-rk2-qlora-topk-topk8-delta0p03
yourname/aime-2025-lfm2-5-1-2b-thinking-dppo-rk2-qlora-topk-topk8-delta0p03
```

So by default the naming includes the dataset, model, algorithm, rollout-group `k`, and any algorithm-specific settings that matter for the run.

Other examples:

```bash
python train.py --model Skywork/Skywork-OR1-Math-7B --algo grpo
python train.py --model Skywork/Skywork-OR1-Math-7B --algo ppo
python compare_divergence.py --model nvidia/AceReason-Nemotron-1.1-7B --approx all
python tui.py
```

## Divergence Approximation

Why the approximation story exists at all:

For DPPO, checking full distribution shift directly is expensive for LLMs, so this repo keeps three views of it:

- `binary`
  sampled-token probability vs everything else

- `top-k`
  keep `TopK(mu)` plus the sampled token, and collapse the rest into `other`

- `naive`
  full-vocab divergence

`naive` is eval-only here because that is the expensive reference version.

## Repo Layout

- `train.py`: main training CLI
- `compare_divergence.py`: eval-only divergence comparison
- `tui.py`: grey/white operator console
- `graphs.py`: plot TensorBoard traces
- `llmrl/`: training, auth, divergence, reward, and logging code
