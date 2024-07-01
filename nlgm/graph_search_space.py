import numpy as np
import itertools
from tqdm import tqdm
from manifolds import BasicManifold, ProductManifold


def manifold_to_curvature(manifold: str):
    """
    Helper function to convert dimension 2 manifold type to arbitrary curvature that matches.
    """
    if manifold == "E2":
        return 0
    elif manifold == "S2":
        return 1
    else:  # manifold == 'H2'
        return -1


def manifold_type(manifold: BasicManifold):
    """
    Helper function to identify manifold type based on curvature. Assumes dimension 2.
    """
    if manifold.dimension != 2:
        raise ValueError(
            f"Dimension must be 2, found manifold.dimension={manifold.dimension}"
        )

    if manifold.curvature == 0:
        return "E2"
    elif manifold.curvature > 0:
        return "S2"
    else:  # manifold.curvature < 0
        return "H2"


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
        ("E2", "S2"): 0.23,
        ("S2", "E2"): 0.23,
        ("E2", "H2"): 0.77,
        ("H2", "E2"): 0.77,
        ("S2", "H2"): 0.84,
        ("H2", "S2"): 0.84,
    }

    # Check if the product manifolds have the same number of component manifolds
    if len(manifold1.manifolds) != len(manifold2.manifolds):
        return 0.0  # No connection between product manifolds of different dimensions

    # Identifying the types of manifolds in each ProductManifold
    types1 = sorted([manifold_type(manifold) for manifold in manifold1.manifolds])
    types2 = sorted([manifold_type(manifold) for manifold in manifold2.manifolds])

    """
    NOTE: This doesn't seem necessary if you just sort and then perform a linear search!
          It also is just returning incorrect values.

    # Create the cost matrix based on the distances between manifold types
    cost_matrix = [[distances.get((t1, t2), 1.0) for t2 in types2] for t1 in types1]

    # Use the Hungarian algorithm to find the minimum weight matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    weight = sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind))
    """
    i = 0

    while i < len(types1):
        if types1[i] != types2[i]:
            break
        i += 1

    return 1 / distances.get((types1[i], types2[i]))


def construct_graph_search_space(n_p: int, curvature_choices: list = [-1, 0, 1]):
    """
    Constructs the graph search space for finding the optimal latent geometry.

    Args:
        n_p (int): The number of model spaces in each product manifold.
        curvature_choices (list): The possible curvatures to choose from (default: [-1, 0, 1]).

    Returns:
        - adjacency_matrix: A 2D numpy array representing the adjacency matrix of the graph search space.
        - signatures: A list product manifold curvatures. With model space dimension 2, this specifies the signature.
                    The index of the tuple in the list corresponds to the index of the node in the adjacency matrix.


    Note:
        - All model spaces have dimension 2.
        - Nodes in the graph represent different geometries (product manifolds).
        - Edges between nodes represent the distances between the corresponding product manifolds.
        - A distance of 0.0 means there is no edge between the nodes.
        - All product manifolds have n_p model spaces, ensuring a constant total dimension throughout the graph.
    """
    # Generate all possible combinations of curvatures for the given number of model spaces
    curvature_combinations = list(itertools.product(curvature_choices, repeat=n_p))

    # Deduplicate: Sort the curvatures within each combination to avoid duplicates
    curvature_combinations = list(
        set([tuple(sorted(combination)) for combination in curvature_combinations])
    )

    # Create nodes (product manifolds) for each curvature combination
    nodes = [ProductManifold(curvatures) for curvatures in curvature_combinations]

    # Initialize the adjacency matrix with zeros
    adjacency_matrix = np.zeros((len(nodes), len(nodes)))

    # Compute the distances between nodes and add edges to the adjacency matrix
    for i in tqdm(range(len(nodes)), desc="Constructing Graph Search Space"):
        for j in range(i + 1, len(nodes)):
            weight = compute_weight(nodes[i], nodes[j])
            if weight != 0.0:
                adjacency_matrix[i][j] = weight
                adjacency_matrix[j][i] = weight

    return adjacency_matrix, curvature_combinations
