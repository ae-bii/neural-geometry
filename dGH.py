import torch
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from manifolds import BasicManifold, ProductManifold


def manifold_type(manifold: BasicManifold):
    """
    Helper function to identify manifold type based on curvature. Assumes dimension 2.
    """
    if manifold.dimension != 2:
        raise ValueError(f"Dimension must be 2, found manifold.dimension={manifold.dimension}")

    if manifold.curvature == 0:
        return 'E2'
    elif manifold.curvature > 0:
        return 'S2'
    else:  # manifold.curvature < 0
        return 'H2'
    

def compute_weight(manifold1: ProductManifold, manifold2: ProductManifold):
    """
    Computes the weight between two ProductManifold objects based on the Gromov-Hausdorff distances
    between their component manifolds using the Hungarian algorithm.


    Returns: 
        Inverse Gromov-Haustorff distance between manifold1 and manifold2. If the manifolds 
        have mismatching dimensions, returns 0.

    Note:
        The weight computation is formulated as an optimal matching problem because we want to find
        the best alignment between the component manifolds of the two ProductManifold objects that
        minimizes the total distance. Each component manifold from manifold1 should be matched with
        exactly one component manifold from manifold2 in a way that minimizes the sum of the distances
        between the matched pairs.

        The Hungarian algorithm is used to solve this optimal matching problem efficiently. It finds
        the minimum weight perfect matching in a bipartite graph, where the nodes represent the component
        manifolds and the edges represent the distances between them.
    """
    # GH distances between pairs of geometric spaces
    distances = {
        ('E2', 'S2'): 0.23,
        ('S2', 'E2'): 0.23,
        ('E2', 'H2'): 0.77,
        ('H2', 'E2'): 0.77,
        ('S2', 'H2'): 0.84,
        ('H2', 'S2'): 0.84,
    }

    # Check if the product manifolds have the same number of component manifolds
    if len(manifold1.manifolds) != len(manifold2.manifolds):
        return 0.0  # No connection between product manifolds of different dimensions

    # Identifying the types of manifolds in each ProductManifold
    types1 = [manifold_type(manifold) for manifold in manifold1.manifolds]
    types2 = [manifold_type(manifold) for manifold in manifold2.manifolds]

    # Create the cost matrix based on the distances between manifold types
    cost_matrix = [[distances.get((t1, t2), 1.0) for t2 in types2] for t1 in types1]

    # Use the Hungarian algorithm to find the minimum weight matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    weight = sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind))

    return 1 / weight