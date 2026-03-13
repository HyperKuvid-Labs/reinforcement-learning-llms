# TensorBoard-Style Plot Scripts (Per Metric)

Each metric has its own focused script (black background + colorful TensorBoard-style line).
Every script reads one CSV from `../csv` and writes one image plot (`.png`) into `./plots`.

## Output folder

- `data/unsloth_trl/plot_scripts/plots`

## Run all metrics (single command)

From project root:

```bash
python data/unsloth_trl/plot_scripts/run_all_metrics.py
```

## Run each metric script

From project root:

```bash
python data/unsloth_trl/plot_scripts/plot_accuracy_r_mean.py
python data/unsloth_trl/plot_scripts/plot_clipped_ratio.py
python data/unsloth_trl/plot_scripts/plot_completion_length.py
python data/unsloth_trl/plot_scripts/plot_forward_r_mean.py
python data/unsloth_trl/plot_scripts/plot_grad_norm.py
python data/unsloth_trl/plot_scripts/plot_kl.py
python data/unsloth_trl/plot_scripts/plot_loss.py
python data/unsloth_trl/plot_scripts/plot_lr.py
python data/unsloth_trl/plot_scripts/plot_reward.py
python data/unsloth_trl/plot_scripts/plot_reward_std.py
```

## Files

- `metric_plotter.py` (shared plotting helper)
- `run_all_metrics.py` (runs all metric scripts in one shot)
- `plot_*.py` (dedicated script for one metric each)

## Dependency for image export

```bash
pip install kaleido
```