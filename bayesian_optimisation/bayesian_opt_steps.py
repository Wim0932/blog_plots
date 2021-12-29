from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from gaussian_processes.gp_regression import gp_regression
from utils.utils import file_path, clear_plot, FUNC
from bayesian_optimisation.acquisition_functions import lower_confidence_bound


@clear_plot
def plot_steps(
    output_path: Path,
    num_sample_points: int,
    xlims: Tuple[float, float],
    segments: int = 51,
):
    """Plots the steps undertaken during bayesian optimization"""
    sample_points = np.random.uniform(*xlims, size=(num_sample_points,))
    sample_evals = FUNC(sample_points)
    x = np.linspace(*xlims, num=segments)
    y = FUNC(x)

    mu_cond, cov_cond, _ = gp_regression(sample_points, 1, x)
    sigma = np.sqrt(np.diag(cov_cond))

    next_sample = x[np.argmin(lower_confidence_bound(mu_cond, sigma))]
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 7))
    plt.suptitle("Bayesian Optimisation Steps")

    axes[0, 0].plot(sample_points, sample_evals, "o", color="k", label="Sample Points")
    axes[0, 0].plot(x, y, "-", color="b", label="Unknown Function")
    axes[0, 0].set_title("1. Evaluate Random Points Across Domain")
    axes[0, 0].legend()

    axes[0, 1].plot(sample_points, sample_evals, "o", color="k", label="Sample")
    axes[0, 1].plot(x, y, "-", color="b", label="Unknown")
    axes[0, 1].plot(x, mu_cond, color="r", label="Regressed")
    axes[0, 1].set_title("2. Regress Mean Function")
    axes[0, 1].legend()

    axes[1, 0].plot(sample_points, sample_evals, "o", color="k", label="Sample")
    axes[1, 0].plot(x, y, "-", color="b", label="Unknown")
    axes[1, 0].plot(x, mu_cond, color="r", label="Regressed")
    axes[1, 0].fill_between(
        x,
        mu_cond - 2 * sigma,
        mu_cond + 2 * sigma,
        alpha=0.15,
        color="r",
        label="$\sigma$",
    )
    axes[1, 0].set_title("3. Calculate Uncertainties")
    axes[1, 0].legend()

    axes[1, 1].plot(sample_points, sample_evals, "o", color="k", label="Sample")
    axes[1, 1].plot(x, y, "-", color="b", label="Unknown")
    axes[1, 1].plot(x, mu_cond, color="r", label="Regressed")
    axes[1, 1].fill_between(
        x,
        mu_cond - 2 * sigma,
        mu_cond + 2 * sigma,
        alpha=0.15,
        color="r",
        label="$\sigma$",
    )
    axes[1, 1].axvline(next_sample, color="k", label="Next Eval")
    axes[1, 1].set_title("4. Find Next Eval With Acquisition Func")
    axes[1, 1].legend()
    plt.tight_layout(w_pad=1.8)

    plt.savefig(output_path, dpi=100)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generates plots depicting the steps of bayesian optimisation"
    )
    parser.add_argument("output_path", type=file_path)
    parser.add_argument(
        "num_sample_points",
        type=int,
        help="Number of points to sample from the unknown function.",
    )
    parser.add_argument(
        "--xlims",
        type=float,
        nargs="+",
        default=[-1.5, 1.5],
        help="limits of the x domain",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=51,
        help="For practical purposes should be viewed as number of "
        "linspaces to segment domain into",
    )
    args = parser.parse_args()
    args.xlims = tuple(args.xlims)
    plot_steps(**vars(args))
