from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import file_path, clear_plot


@clear_plot
def plot_univariate(
    means: List[Union[float, int]],
    stds: List[Union[float, int]],
    output_path: Path,
    linspace: int = 100,
    x_lower: Optional[float] = None,
    x_upper: Optional[float] = None,
):
    """
    Plots a collection of univariate gaussian with means <means> and
    variance <variances>. Saves resulting plot to file.
    :param means: list of the means used to plot the gaussians
    :param stds: list of the standard deviations used to plot the gaussians
    :param linspace: number of linear segments used to discretize the domain.
    :param x_lower: lower limit of the domain on which gaussians are plotted
    :param x_upper: upper limit of the domain on which gaussians are plotted
    :return:
    """
    if x_lower is None:
        x_lower = calc_lower_limit(np.array(means), np.array(stds))

    if x_upper is None:
        x_upper = calc_upper_limit(np.array(means), np.array(stds))

    x = np.linspace(x_lower, x_upper, num=linspace)
    for mean, std in zip(means, stds):
        pdf = univariate_normal(x, mean, std ** 2)
        # We avoid f-strings below due to the latex typing used.
        label = "$\mathcal{N}(%.2f, %.2f)$" % (mean, std)
        plt.plot(x, pdf, label=label)

    plt.xlabel("$x$")
    plt.ylabel("$p(x)$")
    plt.title("Univariate Normal Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def calc_lower_limit(means: np.ndarray, variances: np.ndarray) -> float:
    return np.min(means - 2 * variances)


def calc_upper_limit(means: np.ndarray, variances: np.ndarray) -> float:
    return np.max(means + 2 * variances)


def univariate_normal(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """pdf of the univariate normal distribution."""
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Produce univariate normal distribution plots for a list of means and variances"
    )
    parser.add_argument(
        "--means",
        type=float,
        nargs="+",
        help="List of means for the univariate gaussians to plot.",
    )
    parser.add_argument(
        "--stds",
        type=float,
        nargs="+",
        help="List of variances for the univariate gaussians to plot.",
    )
    parser.add_argument(
        "--output_path", type=file_path, help="File path to which plot is saved."
    )
    parser.add_argument(
        "--linspace", default=100, help="Number of linspace discretisations to use."
    )
    parser.add_argument("--x_lower", default=None, help="Lower limit of the plot.")
    parser.add_argument("--x_upper", default=None, help="Upper limit of the plot.")
    args = parser.parse_args()
    plot_univariate(**vars(args))
