import numpy as np

from math import erf, sqrt, pi


def expected_improvement(
    y_min: float, mu: np.ndarray, sigma: np.ndarray, kappa: float = 0
):
    u = (y_min - mu + kappa) / (sqrt(2) * sigma)
    pmf = normal_pmf(u)
    return (sigma / (sqrt(2) * pi)) * np.exp(-(u ** 2)) + 0.5 * (y_min - mu) * (1 + pmf)


def lower_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa=2):
    return mu - kappa * sigma


def probability_of_improvement(
    y_min: float, mu: np.ndarray, sigma: np.ndarray, kappa: float = 0.0
):
    u = (y_min - mu + kappa) / (sqrt(2) * sigma)
    return normal_pmf(u)


def normal_pmf(x: np.array):
    return 0.5 * np.array([1.0 + erf(y) for y in x])
