"""Single-line learning curves matching common paper figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth_returns(values, window):
    """Return a trailing moving average with edge handling."""
    values = np.asarray(values, dtype=np.float64)
    means = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        means[index] = np.mean(values[start : index + 1])
    return means


def save_reward_curve(
    rewards,
    output_path,
    smoothing_window=10,
    title="LSAC Training",
):
    """Save one blue smoothed returns line as PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8.0, 5.2))

    if rewards:
        reward_array = np.asarray(rewards, dtype=np.float64)
        episodes = np.arange(1, len(reward_array) + 1)
        window = min(smoothing_window, len(reward_array))
        reward_mean = smooth_returns(reward_array, window)
        axis.plot(
            episodes,
            reward_mean,
            color="#1f77b4",
            linewidth=1.8,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No completed episodes",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )

    axis.set_title(title, fontsize=16)
    axis.set_xlabel("Episodes", fontsize=13)
    axis.set_ylabel("Returns", fontsize=13)
    axis.tick_params(labelsize=11)
    axis.grid(False)
    axis.margins(x=0.01)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
