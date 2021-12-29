from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from gaussian_processes.squared_exponential import squared_exponential
from gaussian_processes.plot_brownian_motion import plot
from utils.utils import file_path


def plot_gp(
    output_path: Path,
    func_samples: int = 5,
    xlims: Tuple[int, int] = (-4, 4),
    num_random_vars: int = 51,
):
    x = np.linspace(*xlims, num_random_vars)
    cov = squared_exponential(x, x)

    samples = np.random.multivariate_normal(
        mean=np.zeros(num_random_vars), cov=cov, size=func_samples
    )

    title = "Gaussian Process\n Squared Exponential Kernel"
    plot_kwargs = {"linestyle": "-", "marker": "o", "markersize": 3}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    plot(ax, x, samples, title, "$x$", "$y=f(x)$", plot_kwargs)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generates n instances of gaussian process with a squared exponential kernel"
    )
    parser.add_argument("output_path", type=file_path)
    parser.add_argument(
        "--func_samples",
        type=int,
        default=5,
        help="Number of functions to sample across full domain.",
    )
    parser.add_argument(
        "--xlims",
        type=float,
        nargs="+",
        default=[-2.5, 2.5],
        help="limits of the x domain",
    )
    parser.add_argument(
        "--num_random_vars",
        type=int,
        default=51,
        help="For practical purposes should be viewed as segment count for domain",
    )
    args = parser.parse_args()
    args.xlims = tuple(args.xlims)
    plot_gp(**vars(args))
