from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Dict, List, Callable

import matplotlib.pyplot as plt
import numpy as np

from gaussian_processes.gp_regression import gp_regression
from utils.utils import file_path, clear_plot, FUNC
from bayesian_optimisation.acquisition_functions import (
    lower_confidence_bound,
    expected_improvement,
    probability_of_improvement,
)


@clear_plot
def acquisition_func_plots(
    output_path: Path,
    num_sample_points: int,
    xlims: Tuple[float, float],
    segments: int = 51,
):
    """Plots varying acquisition functions for a bayesian optimization routine."""
    sample_points = np.random.uniform(*xlims, size=(num_sample_points,))
    x = np.linspace(*xlims, num=segments)
    y = FUNC(x)

    mu_cond, cov_cond, _ = gp_regression(sample_points, 1, x)
    sigma = np.sqrt(np.diag(cov_cond))
    y_min = np.min(FUNC(sample_points))

    eval_points = {}
    ei, eval_points = _evaluate_acquisition_func(
        x,
        eval_points,
        key="EI",
        func=expected_improvement,
        func_args=[y_min, mu_cond, sigma],
        get_max=True,
    )

    pi, eval_points = _evaluate_acquisition_func(
        x,
        eval_points,
        key="PI",
        func=probability_of_improvement,
        func_args=[y_min, mu_cond, sigma],
        get_max=True,
    )

    lcb, eval_points = _evaluate_acquisition_func(
        x,
        eval_points,
        key="LCB",
        func=lower_confidence_bound,
        func_args=[mu_cond, sigma],
        get_max=False,
    )

    fig, axes = plt.subplots(ncols=1, nrows=4, figsize=(8, 14))
    plt.suptitle("Varying Acquisition Functions:")

    title = "Regressed Function"
    _plot_function(axes[0], x, y, mu_cond, sigma, sample_points, title, eval_points)
    _plot_acqui(axes[1], x, ei, "Expected Improvement", eval_points["EI"])
    _plot_acqui(axes[2], x, pi, "Probability of Improvement", eval_points["PI"])
    _plot_acqui(axes[3], x, lcb, "Lower Confidence Bound", eval_points["LCB"])
    plt.tight_layout(w_pad=1.8)

    plt.savefig(output_path, dpi=100)


def _plot_function(ax, x, y, mu_cond, sigma, sample_points, title, points_dict):
    ax.set_title(title)
    sample_evals = FUNC(sample_points)
    ax.plot(sample_points, sample_evals, "o", color="k", label="Sample")
    ax.plot(x, y, "-", color="b", label="Unknown")
    ax.plot(x, mu_cond, color="r", label="Regressed")
    ax.fill_between(
        x,
        mu_cond - 2 * sigma,
        mu_cond + 2 * sigma,
        alpha=0.15,
        color="r",
        label="$\sigma$",
    )
    for key, coords in points_dict.items():
        ax.plot(
            coords["x"],
            y[coords["ind"]],
            marker="o",
            label=key,
            markersize=10,
            alpha=0.5,
        )
    ax.legend()


def _plot_acqui(ax, x, y, title, points_dict):
    eval_x = points_dict["x"]
    eval_y = points_dict["y"]

    ax.set_title(title)
    ax.plot(x, y)
    ax.plot(eval_x, eval_y, "o", color="k")


# Passing func args as below is really not the best. I don't feel great
# using a mutable type like this.
def _evaluate_acquisition_func(
    x: np.ndarray,
    points_dict: Dict,
    key: str,
    func: Callable,
    func_args: List,
    get_max: bool,
):
    acqui = func(*func_args)
    x, y, ind = _get_next_eval_point(x, acqui, get_max)
    points_dict[key] = {"x": x, "y": y, "ind": ind}
    return acqui, points_dict


def _get_next_eval_point(x: np.ndarray, y: np.ndarray, get_max: bool = True):
    """
    Uses the acquisition function value and domain values to determine
    next eval point.
    :param x: linspace of the domain
    :param y: acquisition function value across domain
    :param max: If try the maximum acquisition function value is sought. If false the min is.
    :return: domain value of next eval point, acquisition function at next eval point.
    """
    if get_max:
        ind = np.argmax(y)
    else:
        ind = np.argmin(y)
    return x[ind], y[ind], ind


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generates plots showing comparing multiple type of aquisition functions."
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
        default=201,
        help="For practical purposes should be viewed as segment count for domain",
    )
    args = parser.parse_args()
    args.xlims = tuple(args.xlims)
    acquisition_func_plots(**vars(args))
