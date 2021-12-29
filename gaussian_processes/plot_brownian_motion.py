from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from utils.utils import file_path, clear_plot


@clear_plot
def plot_brownian_motion(
    output_path: Path,
    number_of_walks: int = 5,
    time_steps: int = 100,
    total_time: float = 1.0,
):
    """Plots paths of a Wiener Process and the delta d taken in each path over time."""
    delta_t = total_time / time_steps
    stdev = np.sqrt(delta_t)
    distances, deltas = _calculate_distances(stdev, number_of_walks, time_steps)
    t = np.arange(0, total_time, delta_t)

    fig, (ax_1, ax_2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    title = f"Brownian Motion \n {number_of_walks} samples"
    plot(ax_1, t, distances, title=title, x_label="$t$", y_label="$d$")

    title = "$\Delta$ d against time"
    plot(ax_2, t, deltas, title=title, x_label="$t$", y_label="$\Delta$ d")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)


def _calculate_distances(stdev: float, number_of_walks: int, time_steps: int):
    mean = np.zeros(time_steps)
    covariance = stdev * np.identity(time_steps)
    deltas = np.random.multivariate_normal(
        mean=mean, cov=covariance, size=number_of_walks
    )
    return np.cumsum(deltas, axis=1), deltas


def plot(
    ax: Axes,
    x: np.ndarray,
    samples: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    plot_kwargs: Dict = {},
):

    for i, sample in enumerate(samples):
        color = plt.cm.viridis(i / samples.shape[0])
        ax.plot(x, sample, color=color, **plot_kwargs)
    ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlim(x[0], x[-1])


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generates n instances of a brownian motion based random walk"
    )
    parser.add_argument("output_path", type=file_path)
    parser.add_argument(
        "--number_of_walks", type=int, default=5, help="Number of walks to sample"
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=100,
        help="Number of timesteps to use across the domain",
    )
    parser.add_argument(
        "--total_time", type=float, default=1.0, help="Total time of the domain"
    )
    args = parser.parse_args()
    plot_brownian_motion(**vars(args))
