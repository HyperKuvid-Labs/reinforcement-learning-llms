from metric_plotter import plot_single_metric


if __name__ == "__main__":
    plot_single_metric(
        metric_name="forward_r_mean",
        title="Forward Reward Mean",
        color="#118ab2",
        y_title="Reward",
    )
