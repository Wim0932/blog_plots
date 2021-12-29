from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from gaussian_processes.squared_exponential import squared_exponential
from gaussian_processes.plot_brownian_motion import plot
from utils.utils import clear_plot, file_path, FUNC


@clear_plot
def plot_gp_regression(
    output_path: Path,
    num_sample_points: int,
    xlims: Tuple[float, float],
    segments: int = 51,
):
    """Plots a gaussian processs, with square exp covariance, regressed to an unknown func."""
    sample_points = np.random.uniform(*xlims, size=(num_sample_points,))
    sample_evals = FUNC(sample_points)
    x = np.linspace(*xlims, num=segments)

    mu_cond, cov_cond, samples = gp_regression(sample_points, num_samples=5, x=x)
    sigma = np.sqrt(np.diag(cov_cond))

    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(9, 7))
    plt.suptitle("Conditional GP Sampling")

    title = "Sampled Functions Conditioned on Points"
    plot(axes[0], x, samples, title=title, x_label="$x$", y_label="$f(x)$")
    axes[0].plot(sample_points, sample_evals, "o", color="k", label="Known Points")

    title = "Mean and Variances"
    plot(
        axes[1],
        x,
        samples,
        title=title,
        x_label="$x$",
        y_label="$f(x)$",
        plot_kwargs={"alpha": 0.25},
    )
    axes[1].plot(sample_points, sample_evals, "o", color="k", label="Known Points")
    axes[1].plot(x, mu_cond, color="red", label="Mean Function")
    axes[1].fill_between(
        x,
        mu_cond - 2 * sigma,
        mu_cond + 2 * sigma,
        color="red",
        alpha=0.25,
        label="$\sigma$",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)


def gp_regression(
    sample_points: np.ndarray, num_samples: int, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates conditional mean <mu_cond>, conditional covariance <cov_cond>
    given known function points FUNC(<x>). Pull samples from conditioned
    multivariate gaussian N(mu_cond, cov_cond). Returns sampled processes
    conditional mean and covariance.
    """
    sample_evals = FUNC(sample_points)

    mu_cond, cov_cond = gp_posterior(
        sample_evals, sample_points, x, squared_exponential
    )
    samples = np.random.multivariate_normal(
        mean=mu_cond, cov=cov_cond, size=num_samples
    )

    return mu_cond, cov_cond, samples


def gp_posterior(y1, x1, x2, kernel_func, mu_1=None, mu_2=None):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Default behaviour is to assume a mean of zero.
    if mu_2 is None:
        mu_2 = np.zeros(x2.shape)
    if mu_1 is None:
        mu_1 = np.zeros(x1.shape)

    # Pull the values of the kernel functions
    cov_11 = kernel_func(x1, x1)
    cov_12 = kernel_func(x1, x2)
    cov_22 = kernel_func(x2, x2)

    # Calculated conditional / posterior means and covariances
    inverse = np.dot(np.linalg.inv(cov_11), cov_12)
    mean_cond = mu_2 + np.dot(inverse.T, (y1 - mu_1))
    cov_cond = cov_22 - np.dot(inverse.T, cov_12)
    return mean_cond, cov_cond


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Regresses a gaussian process onto a set of points "
        "derived from an unknown function."
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
        help="For practical purposes should be viewed as segment count for domain",
    )
    args = parser.parse_args()
    args.xlims = tuple(args.xlims)
    plot_gp_regression(**vars(args))
