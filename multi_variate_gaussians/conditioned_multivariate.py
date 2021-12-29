from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.contour import QuadContourSet
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from multi_variate_gaussians.multivariate import (
    make_2d_space,
    plot_multivariate_on_axes,
    multivariate_normal,
)
from multi_variate_gaussians.univariate import univariate_normal
from utils.utils import read_yaml, clear_plot, file_path


@clear_plot
def plot_conditional_multivariate(config: Dict, output_path: Path):
    """
    Generates 3 plots via the <config> dict input.
    plot 1: multivariate gaussian contour according to "multivariate_params"
    of config
    plot 2: univariate gaussian derived from conditioning multivariate gaussian
    on "x_condition" of config.
    plot 3: univariate gaussian derived from conditioning multivariate gaussian
    on "y_condition" of config.
    :param config: Dictionary containing all necessary plotting inputs
    :param output_path: path to which plot is written
    """
    mean, cov, y_condition, x_condition = _unpack_config(config["multivariate_params"])
    x_mean, y_mean, x_var, y_var = _calculated_conditioned_params(
        mean, cov, x_condition, y_condition
    )

    space = make_2d_space(**config.get("space_kwargs", {}))
    pdf = multivariate_normal(space, mean, cov)

    x = space[0, :, 0]
    y = space[:, 0, 1]
    x_uni = univariate_normal(x, x_mean, x_var)
    y_uni = univariate_normal(y, y_mean, y_var)

    grid_spec_kwargs = {"width_ratios": [2, 1], "height_ratios": [2, 1]}
    fig, axes = plt.subplots(2, 2, gridspec_kw=grid_spec_kwargs)
    plt.suptitle("Conditional Distributions of a Bivariate Gaussian")

    con = _plot_multivariate_with_lines(
        axes[0, 0], pdf, space, x_condition, y_condition
    )

    plot_kwargs = dict(color="#0057e7", label=f"$p(y|x={x_condition:.1f})$")
    plot_univariate(axes[0, 1], y_uni, y, label_x=True, **plot_kwargs)

    plot_kwargs = dict(color="#d62d20", label=f"$p(x|y={y_condition:.1f})$")
    plot_univariate(axes[1, 0], x, x_uni, label_y=True, **plot_kwargs)

    set_color_bar(fig, axes[1, 1], con)
    axes[1, 1].set_visible(False)

    plt.savefig(output_path, dpi=100)


def plot_univariate(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    label_x: bool = False,
    label_y: bool = False,
    **plot_kwargs,
):
    """plots univariate where pdf can be plotted on x or y axis"""
    ax.plot(x, y, **plot_kwargs)
    if label_x:
        ax.set_xlabel("density", fontsize=13)
        ax.xaxis.set_label_position("top")
    if label_y:
        ax.set_ylabel("density", fontsize=13)
        ax.yaxis.set_label_position("left")
    ax.legend()
    ax.grid()


def _plot_multivariate_with_lines(
    ax: Axes, pdf: np.ndarray, space: np.ndarray, vline: float, hline: float
) -> QuadContourSet:
    """
    Plots univariate with v & hlines to depict conditionality
    :param ax: mpl axis on which contours are plotted
    :param pdf: np array of the pdf to plot as a contour
    :param space: 2D space (d-stacked meshgrid)
    :param vline: float value for conditional  vline
    :param hline: float value for conditional hline
    :return:
    """
    con = plot_multivariate_on_axes(ax, pdf, space)
    ax.axvline(vline, color="#0057e7")
    ax.axhline(hline, color="#d62d20")
    return con


def set_color_bar(fig: Figure, ax: Axes, con: QuadContourSet):
    """Displays the color bar of the contour <con> at the location of <ax>"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="20%", pad=0.05)
    cbar = fig.colorbar(con, cax=cax)
    cbar.ax.set_ylabel("density: $p(x, y)$", fontsize=13)


def _unpack_config(config: Dict) -> Tuple[np.ndarray, np.ndarray, float, float]:
    mean = np.array(config["mean"])
    cov = np.array(config["covariance"])
    y_condition = config["y_condition"]
    x_condition = config["x_condition"]
    return mean, cov, x_condition, y_condition


def _calculated_conditioned_params(
    mean: np.ndarray, cov: np.ndarray, x_condition: float, y_condition: float
) -> Tuple[float, float, float, float]:
    """Calculates all conditioned means and std's"""
    x_mean = _mean_xgiveny(mean, y_condition, cov)
    y_mean = _mean_ygivenx(mean, x_condition, cov)
    x_std = _cov_xgiveny(cov)
    y_std = _cov_ygivenx(cov)
    return x_mean, y_mean, x_std, y_std


def _cov_xgiveny(cov: np.ndarray) -> float:
    return cov[0, 0] - cov[0, 1] * (1.0 / cov[1, 1]) * cov[1, 0]


def _cov_ygivenx(cov: np.ndarray) -> float:
    return cov[1, 1] - cov[1, 0] * (1.0 / cov[0, 0]) * cov[0, 1]


def _mean_xgiveny(mean: np.ndarray, y_condition: float, cov: np.ndarray) -> float:
    return mean[0] + (cov[0, 1] * (1 / cov[1, 1]) * (y_condition - mean[1]))


def _mean_ygivenx(mean: np.ndarray, x_condition: float, cov: np.ndarray) -> float:
    return mean[1] + (cov[1, 0] * (1 / cov[0, 0]) * (x_condition - mean[0]))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Produce conditional multivariate gaussian plots from a "
        "config describing means, covariances and conditions."
    )
    parser.add_argument("config", type=file_path, help="Plot config file")
    parser.add_argument(
        "output_path", type=file_path, help="File path to which plot is saved."
    )
    args = parser.parse_args()
    args.config = read_yaml(args.config)
    plot_conditional_multivariate(**vars(args))
