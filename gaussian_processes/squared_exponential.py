from argparse import ArgumentParser
from typing import Tuple
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import spatial

from utils.utils import CMAP, file_path


def plot_square_exponential(
    output_path: Path,
    x_cut: float = 0.0,
    length_scale: float = 1,
    xlims: Tuple[float, float] = (-2.5, 2.5),
    segments: int = 51,
):
    """
    Plots imshow of squared exponential covariance matrix and
    the squared exponential function for K(<x_cut>, x).
    """
    x = np.linspace(*xlims, num=segments)
    cov = squared_exponential(x, x, length_scale)
    line = squared_exponential(np.array([x_cut]), x, length_scale)[0, :]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    _plot_mat(ax1, cov, x)
    _plot_line(ax2, x, line, x_cut)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)


def squared_exponential(
    xa: np.ndarray, xb: np.ndarray, length_scale: float = 1.0
) -> np.ndarray:
    """Squared exponential kernel function <l> depicts characteristic length scale"""
    xa = np.expand_dims(xa, 1)
    xb = np.expand_dims(xb, 1)
    sq_norm = spatial.distance.cdist(xa, xb) ** 2
    return np.exp(-sq_norm / (2 * length_scale ** 2))


def _plot_line(ax: Axes, x: np.ndarray, y: np.ndarray, x_cut: float):
    ax.plot(x, y, label=f"$k(x,{x_cut})$")
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("covariance", fontsize=13)
    ax.set_title((f"Squared Exponential \n" f"between $x$ and {x_cut}"))
    ax.legend(loc=1)


def _plot_mat(ax: Axes, mat: np.ndarray, x: np.ndarray):
    im = ax.imshow(mat, cmap=CMAP, extent=[x[-1], x[0], x[0], x[-1]])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("$k(x,x)$", fontsize=10)
    ax.set_title(("Squared Exponential \n" "example of covariance matrix"))
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("x", fontsize=13)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="generates matrix and line plot of squared exponential covariance"
    )
    parser.add_argument(
        "output_path", type=file_path, help="Path to which output plot is saved"
    )
    parser.add_argument(
        "--x_cut",
        type=int,
        default=5,
        help="x coordinate along which covariance line plot is plotted.",
    )
    parser.add_argument(
        "--length_scale",
        type=float,
        default=1,
        help="Length scale of the kernel function",
    )
    parser.add_argument(
        "--xlims",
        type=float,
        nargs="+",
        default=[-2.5, 2.5],
        help="limits of the x domain",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=51,
        help="number of segments the domain is divided into.",
    )
    args = parser.parse_args()
    args.xlims = tuple(args.xlims)

    plot_square_exponential(**vars(args))
