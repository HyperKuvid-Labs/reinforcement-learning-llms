from __future__ import annotations

from metric_plotter import plot_single_metric


def main() -> None:
    plot_single_metric(
        metric_name="accuracy_r_mean",
        title="Accuracy R Mean",
        color="#ffd166",
        y_title="Accuracy",
    )
    plot_single_metric(
        metric_name="clipped_ratio",
        title="Clipped Ratio",
        color="#ef476f",
        y_title="Ratio",
    )
    plot_single_metric(
        metric_name="completion_length",
        title="Completion Length",
        color="#06d6a0",
        y_title="Length",
    )
    plot_single_metric(
        metric_name="forward_r_mean",
        title="Forward Reward Mean",
        color="#118ab2",
        y_title="Reward",
    )
    plot_single_metric(
        metric_name="grad_norm",
        title="Gradient Norm",
        color="#f78c6b",
        y_title="Norm",
    )
    plot_single_metric(
        metric_name="kl",
        title="KL Divergence",
        color="#9b5de5",
        y_title="KL",
    )
    plot_single_metric(
        metric_name="loss",
        title="Loss",
        color="#ff5d8f",
        y_title="Loss",
    )
    plot_single_metric(
        metric_name="lr",
        title="Learning Rate",
        color="#00bbf9",
        y_title="Learning Rate",
    )
    plot_single_metric(
        metric_name="reward",
        title="Reward",
        color="#00f5d4",
        y_title="Reward",
    )
    plot_single_metric(
        metric_name="reward_std",
        title="Reward Std",
        color="#fb8500",
        y_title="Std",
    )


if __name__ == "__main__":
    main()