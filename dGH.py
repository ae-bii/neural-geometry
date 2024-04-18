import torch
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar



def sample_points_euclidean(n: int, dimension: int) -> torch.Tensor:
    # For Euclidean space, sample each dimension uniformly at random from [0, 1)
    return torch.rand(n, dimension)


def chi(t):
    if isinstance(t, (int, np.integer)):
        return 0.0
    else:
        return np.sin(np.pi * t) * np.exp(-np.sin(np.pi * t) ** (-2))

def psi1(x):
    A, _ = quad(chi, 0, 1)
    integral, _ = quad(lambda t: chi(t), 0, 1+x)
    return (1/A) * integral

def psi2(x):
    A, _ = quad(chi, 0, 1)
    integral, _ = quad(lambda t: chi(t), 0, x)
    return (1/A) * integral

def h(x, y):
    return np.array([
        np.sinh(x) / c * psi1(x) * np.cos(c * y),
        np.sinh(x) / c * psi1(x) * np.sin(c * y),
        psi2(x) * np.cos(c * y),
        psi2(x) * np.sin(c * y)
    ])

def psi(x, y):
    return np.array([
        np.arcsinh(y * np.exp(x)),
        np.log(np.sqrt(np.exp(-2*x) + y**2))
    ])

def f0(x, y):
    p = psi(x, y)
    integral, _ = quad(lambda t: np.sqrt(1 - epsilon(t)**2), 0, p[0])
    return np.concatenate((np.array([integral, p[1]]), h(p)))

def f(x, *y):
    n = len(y) + 1
    kappa = 1 / np.sqrt(n - 1)
    return np.concatenate([f0(x, kappa * yi) for yi in y])

def epsilon(t):
    return (G2**2 / (1 + G2**2)) * c**2

# Constants
G1 = minimize_scalar(lambda x: -np.abs(np.sinh(x) * psi1(x)), bounds=(-2, 2), method='bounded').fun
G2 = minimize_scalar(lambda x: -np.abs(np.sinh(x) * psi2(x)), bounds=(-2, 2), method='bounded').fun
c = 2 * max(G1, G2)

def map_hyperbolic_to_euclidean(points_Hn, n):
    """
    Map a set of points from hyperbolic space H^n to Euclidean space E^(6n-6).

    Args:
        points_Hn (list or numpy.ndarray): A collection of points in hyperbolic space H^n.
        n (int): The dimension of the hyperbolic space.

    Returns:
        numpy.ndarray: The mapped points in Euclidean space E^(6n-6).
    """
    points_E6n_6 = []

    for point in points_Hn:
        x = point[0]
        y = point[1:]

        mapped_point = f(x, *y)
        points_E6n_6.append(mapped_point)

    return np.array(points_E6n_6)
