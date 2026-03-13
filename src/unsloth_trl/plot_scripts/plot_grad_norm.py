from metric_plotter import plot_single_metric


if __name__ == "__main__":
    plot_single_metric(
        metric_name="grad_norm",
        title="Gradient Norm",
        color="#f78c6b",
        y_title="Norm",
    )
