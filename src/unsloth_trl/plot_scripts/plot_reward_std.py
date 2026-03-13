from metric_plotter import plot_single_metric


if __name__ == "__main__":
    plot_single_metric(
        metric_name="reward_std",
        title="Reward Std",
        color="#fb8500",
        y_title="Std",
    )
