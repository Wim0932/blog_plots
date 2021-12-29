from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from multi_variate_gaussians.multivariate import (
    make_2d_space,
    plot_multivariate_on_axes,
    multivariate_normal,
)
from multi_variate_gaussians.univariate import univariate_normal
from multi_variate_gaussians.conditioned_multivariate import (
    plot_univariate,
    set_color_bar,
)
from utils.utils import read_yaml, clear_plot, file_path


@clear_plot
def plot_marginalized_multivariate(config: Dict, output_path: Path):
    """
    Plots a Bivariate distribution and its marginalizations along each axis.
    :param config: Dictionary containing all necessary plotting inputs
    :param output_path: path to which plot is written
    """
    mean, cov = _unpack_config(config["multivariate_params"])

    x_mean = mean[0]
    y_mean = mean[1]
    x_std = cov[0, 0]
    y_std = cov[1, 1]

    space = make_2d_space(**config.get("space_kwargs", {}))
    pdf = multivariate_normal(space, mean, cov)

    x = space[0, :, 0]
    y = space[:, 0, 1]
    x_uni = univariate_normal(x, x_mean, x_std)
    y_uni = univariate_normal(y, y_mean, y_std)

    grid_spec_kwargs = {"width_ratios": [2, 1], "height_ratios": [2, 1]}
    fig, axes = plt.subplots(2, 2, gridspec_kw=grid_spec_kwargs)
    plt.suptitle("Marginalized Distributions of a Bivariate Gaussian")

    con = plot_multivariate_on_axes(axes[0, 0], pdf, space)

    plot_kwargs = dict(color="#0057e7", label="$p(y)$")
    plot_univariate(axes[0, 1], y_uni, y, label_x=True, **plot_kwargs)

    plot_kwargs = dict(color="#d62d20", label="$p(x)$")
    plot_univariate(axes[1, 0], x, x_uni, label_y=True, **plot_kwargs)

    set_color_bar(fig, axes[1, 1], con)
    axes[1, 1].set_visible(False)

    plt.savefig(output_path, dpi=100)


def _unpack_config(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.array(config["mean"])
    cov = np.array(config["covariance"])
    return mean, cov


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Produce marginalized multivariate gaussian "
        "plots from a config describing means, and covariances."
    )
    parser.add_argument("config", type=file_path, help="Plot config file")
    parser.add_argument(
        "output_path", type=file_path, help="File path to which plot is saved."
    )
    args = parser.parse_args()
    args.config = read_yaml(args.config)
    plot_marginalized_multivariate(**vars(args))
