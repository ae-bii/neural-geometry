import numpy as np
from typing import Tuple
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from skopt.space import Categorical
from skopt.utils import use_named_args


def get_eigendecomposition(
    adjacency_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform eigendecomposition of the adjacency matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Eigenvectors and eigenvalues of the adjacency matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)
    return eigenvectors, np.diag(eigenvalues)


def compute_covariance_matrix(
    eigenvectors: np.ndarray, eigenvalues: np.ndarray, beta: float
) -> np.ndarray:
    """
    Compute the covariance matrix of the Gaussian Process (GP) based on the eigendecomposition of the adjacency matrix.

    Args:
        eigenvectors (np.ndarray): Eigenvectors of the adjacency matrix.
        eigenvalues (np.ndarray): Eigenvalues of the adjacency matrix.
        beta (float): Lengthscale parameter for the covariance matrix.

    Returns:
        np.ndarray: Covariance matrix of the GP.
    """
    covariance_matrix = np.dot(
        eigenvectors, np.dot(np.exp(-beta * eigenvalues), eigenvectors.T)
    )
    return covariance_matrix


class AdjacencyGPKernel(Kernel):
    """
    Custom kernel for the Gaussian Process based on the adjacency matrix of the graph.
    """

    def __init__(self, adjacency_matrix: np.ndarray, beta: float = 1.0):
        self.adjacency_matrix = adjacency_matrix
        self.beta = beta

    def __call__(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        if Y is None:
            Y = X
        v, e = get_eigendecomposition(self.adjacency_matrix)
        K = compute_covariance_matrix(v, e, self.beta)
        return K

    def diag(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        return True


def bayesian_optimization(
    objective_function, signatures, adjacency_matrix, n_iterations=10, callback=None
):
    """
    Perform Bayesian optimization using a Gaussian Process with a custom adjacency-based kernel.

    Args:
        objective_function (Callable[[Any], Tuple[float, List[float]]]): Function to optimize.
        signatures (List[Any]): List of possible signatures to evaluate.
        adjacency_matrix (np.ndarray): Adjacency matrix used for the GP kernel.
        n_iterations (int): Number of optimization iterations.
        callback (Callable): Function to execute after each iteration.

    Returns:
        Tuple[Any, float, List[float]]: Best signature, its validation metric, and training losses.
    """

    # Define the search space with names
    space = [Categorical(signatures, name="signature")]

    # Use named arguments in the objective function
    @use_named_args(space)
    def objective(signature):
        validation_metric, training_losses = objective_function(signature)
        return validation_metric  # Minimize the validation metric

    # Create the GP model using the custom adjacency matrix-based kernel
    kernel = AdjacencyGPKernel(adjacency_matrix)
    gp = GaussianProcessRegressor(kernel=kernel)

    # Perform Bayesian optimization using the GP as the base estimator
    result = gp_minimize(
        objective,
        dimensions=space,
        base_estimator=gp,
        n_calls=n_iterations,
        acq_func="EI",
    )

    # Extract the best result
    best_signature = signatures[result.x]
    best_metric, best_losses = objective_function(best_signature)

    if callback:
        callback(best_signature, best_metric, best_losses)

    return best_signature, best_metric, best_losses
