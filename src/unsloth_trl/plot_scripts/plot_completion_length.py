from metric_plotter import plot_single_metric


if __name__ == "__main__":
    plot_single_metric(
        metric_name="completion_length",
        title="Completion Length",
        color="#06d6a0",
        y_title="Length",
    )
