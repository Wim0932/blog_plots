from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import file_path, clear_plot
from bayesian_optimisation.aquisition_functions import normal_pmf
from gaussian_processes.plot_brownian_motion import plot


@clear_plot
def plot_normal_pdf(output_path: Path, xlims: Tuple[float, float], segments: int = 51):
    """Generates plot of the probability mass function of the normal distribution"""
    x = np.linspace(*xlims, segments)
    y = np.array([normal_pmf(x)])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    title = "$\Phi(x) = 0.5(1 + erf(x/\sqrt{2}))$"
    plot(ax, x, y, title, "$x$", "$\Phi(x)$")
    plt.savefig(output_path, dpi=100)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generates plot of Phi (Normal Dist PMF) function."
    )
    parser.add_argument("output_path", type=file_path)
    parser.add_argument(
        "--xlims", type=float, nargs="+", default=[-3, 3], help="limits of the x domain"
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=201,
        help="Number of linspaces to segment domain into.",
    )
    args = parser.parse_args()
    args.xlims = tuple(args.xlims)
    plot_normal_pdf(**vars(args))
