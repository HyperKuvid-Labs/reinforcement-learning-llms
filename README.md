# AIME 2025 RL Comparison for 1B LLMs

This repo compares **GRPO**, **PPO**, and **DPPO** on **AIME 2025** with:

- `sapientinc/HRM-Text-1B`
- `LiquidAI/LFM2.5-1.2B-Thinking`

The training objective is binary-answer reward on math completions: a rollout gets reward `1` only when the extracted final answer matches the gold AIME answer after normalization.

## Scope

Main comparison:
- `GRPO`
- `PPO`
- `DPPO` with `top-k` divergence approximation by default

Additional divergence analysis:
- `naive` exact divergence, eval-only
- `binary` approximation
- `top-k` approximation

Everything is logged to **TensorBoard**. The TUI is only an operator surface; TensorBoard is the system of record for metrics, config, checkpoint events, upload events, and divergence analysis.

## Models and Dataset

- Dataset: `test-time-compute/aime_2025`
- Model 1: `sapientinc/HRM-Text-1B`
- Model 2: `LiquidAI/LFM2.5-1.2B-Thinking`

Notes from the upstream model cards:
- `sapientinc/HRM-Text-1B` requires `trust_remote_code=True`, and the model card says some setups may need a recent `transformers` build with `hrm_text` support.
- `LiquidAI/LFM2.5-1.2B-Thinking` is a reasoning model intended for Transformers/vLLM-style text generation.

Sources:
- https://huggingface.co/sapientinc/HRM-Text-1B
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
- https://huggingface.co/datasets/test-time-compute/aime_2025

## Setup

The local environment in this repo previously had a broken CUDA-linked PyTorch import, so bootstrap from scratch:

```bash
bash install.sh
```

If `torch` still fails to import with missing CUDA shared libraries, reinstall PyTorch for your exact CUDA version before running training.

Before the first training or divergence run, the CLI now prompts for:
- `HF_USERNAME`
- `HF_TOKEN`

They are saved in a repo-local `.env` file and reused automatically on later runs.

## Training

Run a single training job:

```bash
python train.py \
  --model sapientinc/HRM-Text-1B \
  --algo dppo \
  --dataset test-time-compute/aime_2025 \
  --dppo-approx topk \
  --topk 16 \
  --save-every 20 \
  --push-to-hub \
  --hub-repo your-user/your-repo \
  --delete-local-checkpoints \
  --cpu-offload
```

If `--push-to-hub` is set and `--hub-repo` is omitted, the trainer defaults to:

```text
<HF_USERNAME>/<model-slug>-<algo>
```

Run GRPO or PPO:

```bash
python train.py --model LiquidAI/LFM2.5-1.2B-Thinking --algo grpo
python train.py --model LiquidAI/LFM2.5-1.2B-Thinking --algo ppo
```

Run divergence comparison:

```bash
python compare_divergence.py --model sapientinc/HRM-Text-1B --approx all
```

Launch the operator TUI:

```bash
python tui.py
```

## TensorBoard

Every run writes TensorBoard events under `runs/`.

```bash
tensorboard --logdir runs
```

Representative scalar tags:
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

## Approximation Details

Binary approximation:
- collapse the distribution into sampled-token probability vs the rest

Top-k approximation:
- keep `TopK(mu)` plus the sampled token and aggregate the rest into `other`

Naive approximation:
- compute divergence on the full vocabulary
- enabled only for eval/analysis because it is much more expensive

## Files

- `train.py`: main CLI for GRPO/PPO/DPPO training
- `compare_divergence.py`: eval-only divergence comparison
- `tui.py`: grey/white production-style training console
- `graphs.py`: plot TensorBoard scalar traces
- `llmrl/`: training, divergence, dataset, and logging modules
