from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet

from utils.utils import read_yaml, clear_plot, file_path, CMAP


@clear_plot
def plot_multivariate_subplots(config: Dict, output_path: Path):
    """Plots multivariate gaussians according to contents of config"""
    fig, axes = plt.subplots(nrows=1, ncols=len(config), figsize=(10, 4))
    for entry, ax in zip(config, axes):
        con = _generate_subplot_from_config(entry, ax)
    plt.suptitle("Bivariate Normal Distributions")
    plt.tight_layout()
    cbar = fig.colorbar(con, ax=axes)
    cbar.ax.set_ylabel("$p(x_1, x_2)$", fontsize=13)
    plt.savefig(output_path, dpi=100)


def _generate_subplot_from_config(config_entry: Dict, ax: Axes) -> QuadContourSet:
    mean = np.array(config_entry["mean"])
    covariance = np.array(config_entry["covariance"])
    space = make_2d_space(**config_entry.get("space_kwargs", {}))
    pdf = multivariate_normal(space, mean, covariance)
    return plot_multivariate_on_axes(
        ax, pdf, space, **config_entry.get("plot_kwargs", {})
    )


def multivariate_normal(
    xy: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """pdf of the multivariate normal distribution."""

    def vector_calc(vec):
        x_m = vec - mean
        coeff = 1.0 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance)))
        exponent = -np.dot(x_m, np.dot(np.linalg.inv(covariance), x_m)) / 2.0
        return coeff * np.exp(exponent)

    return np.apply_along_axis(vector_calc, 2, xy)


def make_2d_space(
    x_lims: Union[List[float], Tuple[float, float]] = (-2.5, 2.5),
    y_lims: Union[List[float], Tuple[float, float]] = (-2.5, 2.5),
    segments: int = 51,
) -> np.ndarray:
    x = np.linspace(x_lims[0], x_lims[1], segments)
    y = np.linspace(y_lims[0], y_lims[1], segments)
    coords = np.meshgrid(x, y)
    return np.dstack(coords)


def plot_multivariate_on_axes(
    ax: Axes, pdf: np.ndarray, space: np.ndarray, title: str = "", levels: int = 51
) -> QuadContourSet:
    x = space[0, :, 0]
    y = space[:, 0, 1]
    con = ax.contourf(x, y, pdf, cmap=CMAP, levels=levels)
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_ylabel("$y$", fontsize=13)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    return con


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Produce multivariate gaussian plots according to a config."
    )
    parser.add_argument("config", type=file_path, help="Path to config file.")
    parser.add_argument(
        "output_path", type=file_path, help="File path to which plot is saved."
    )
    args = parser.parse_args()
    args.config = read_yaml(args.config)
    plot_multivariate_subplots(**vars(args))
