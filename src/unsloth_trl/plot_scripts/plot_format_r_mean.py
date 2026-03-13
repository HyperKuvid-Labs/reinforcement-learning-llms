from metric_plotter import plot_single_metric


if __name__ == "__main__":
    output_path = plot_single_metric(
        metric_name="format_r_mean",
        title="Format Reward Mean",
        color="#118ab2",
        y_title="Reward",
    )
    print(f"Saved plot to: {output_path}")
