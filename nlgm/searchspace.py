import numpy as np
import itertools
from tqdm import tqdm
from difflib import SequenceMatcher
from math import isclose
from itertools import combinations_with_replacement, chain
from nlgm.manifolds import BasicManifold, ProductManifold


def manifold_to_curvature(manifold: str):
    """
    Map a 2D manifold code to its representative curvature.

    Parameters
    ----------
    manifold : str
        Manifold code. Supported values are ``"E2"``, ``"S2"``, and ``"H2"``.

    Returns
    -------
    int
        Curvature representative for the manifold code.
    """
    if manifold == "E2":
        return 0
    elif manifold == "S2":
        return 1
    elif manifold == "H2":
        return -1
    else:
        raise ValueError("Only E2, S2, and H2 manifolds supported.")


def manifold_type(manifold: BasicManifold):
    """
    Identify manifold code from curvature, assuming dimension 2.

    Parameters
    ----------
    manifold : BasicManifold
        Manifold instance.

    Returns
    -------
    str
        One of ``"E2"``, ``"S2"``, or ``"H2"``.

    Raises
    ------
    ValueError
        If ``manifold.dimension`` is not 2.
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
    Compute inverse Gromov-Hausdorff weight between product manifolds.

    Parameters
    ----------
    manifold1 : ProductManifold
        First manifold.
    manifold2 : ProductManifold
        Second manifold.

    Returns
    -------
    float
        Inverse distance weight. Returns ``0.0`` when manifold dimensions differ.

    Notes
    -----
    This implementation uses a fixed lookup table between manifold type pairs.
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

    i = 0

    while i < len(types1):
        if types1[i] != types2[i]:
            break
        i += 1

    return 1 / distances.get((types1[i], types2[i]))


def __construct_graph_search_space(
    n_p: int, curvature_choices: list = [-1, 0, 1], connectivity: bool = False
):
    """
    Construct graph search space using full Cartesian curvature combinations.

    Parameters
    ----------
    n_p : int
        Number of model spaces in each product manifold.
    curvature_choices : list, default=[-1, 0, 1]
        Curvature values to combine.
    connectivity : bool, default=False
        Present for API compatibility; this helper returns weighted adjacency.

    Returns
    -------
    numpy.ndarray
        Weighted adjacency matrix for candidate product manifolds.
    list[tuple]
        Signature list corresponding to matrix indices.
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


def construct_graph_search_space(
    n_p: int = 7,
    curvature_choices: list = [-1, 0, 1],
    connectivity: bool = False,
):
    """
    Construct graph search space over product-manifold signatures.

    Parameters
    ----------
    n_p : int, default=7
        Maximum number of model spaces in each product manifold.
    curvature_choices : list, default=[-1, 0, 1]
        Allowed curvature values.
    connectivity : bool, default=False
        If ``True``, return unweighted connectivity only.

    Returns
    -------
    numpy.ndarray
        Adjacency or weighted-distance matrix.
    list[tuple]
        Signature list aligned with returned matrix indices.
    """
    if n_p <= 0 or len(curvature_choices) <= 0:
        return np.array([]), []

    curvature_choices = sorted(curvature_choices)

    # evil python trickery!!
    # unpacking is faster than type conversion or list comprehension
    # choose without replacement sum from i=1 to n_r of (n_s + i - 1 choose i)
    product_spaces = {
        dim: [*combinations_with_replacement(curvature_choices, dim)]
        for dim in range(1, n_p + 1)
    }

    indices = list(chain.from_iterable(product_spaces.values()))
    inv_map = {indices[i]: i for i in range(len(indices))}

    nodes = len(indices)
    distances = np.zeros(
        (nodes, nodes), np.float32
    )  # + np.diag(np.ones(nodes) * np.inf)

    # compute distance between all product spaces
    for dim, s in product_spaces.items():
        for curnode in s:
            # compare against adjacent and current dimensions (make sure this is working properly)
            for i in range(-1, 2):
                if dim + i < 1 or dim + i > n_p:
                    continue

                for compnode in product_spaces[dim + i]:
                    if inv_map[compnode] <= inv_map[curnode]:
                        continue

                    if adj_product_spaces(curnode, compnode):
                        distances[inv_map[compnode]][inv_map[curnode]] = 1

    # flip upper triangle onto bottom triangle
    distances = distances + distances.T - np.diag(np.diag(distances))

    # if we only want an adjacency matrix, return early
    if connectivity:
        return distances, indices

    # inefficiently compute the inverse d_GH between product spaces
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if distances[i][j] != 1:
                continue

            pm1 = ProductManifold(indices[i])
            pm2 = ProductManifold(indices[j])
            dgh = compute_weight(pm1, pm2)

            distances[i][j] = dgh if dgh != 0 else 1  # different dims

    return distances, indices


def adj_product_spaces(s1: list, s2: list) -> bool:
    """
    Check whether two manifold signatures are adjacent product spaces.

    Parameters
    ----------
    s1 : list
        First signature.
    s2 : list
        Second signature.

    Returns
    -------
    bool
        ``True`` if signatures are adjacent, otherwise ``False``.

    Notes
    -----
    Adjacency allows a maximum edit-distance approximation corresponding to
    one manifold insertion/deletion or one replacement.
    """
    if s1 == s2:
        return True

    len_delta = abs(len(s1) - len(s2))
    total_len = len(s1) + len(s2)

    if len_delta > 1:
        return False

    matcher = SequenceMatcher()
    matcher.set_seqs(s1, s2)

    # if equal length, check that approx. Levenshtein distance is 2
    # if length differs by 1, check that approx. Levenshtein distance is 1
    return isclose(1 - matcher.ratio(), (1.0 if len_delta == 1 else 2.0) / total_len)


def get_color(weight: float) -> str:
    """
    Map an edge weight to the plotting color used in search-space visualizations.

    Parameters
    ----------
    weight : float
        Edge weight.

    Returns
    -------
    str
        Matplotlib-compatible color name.
    """
    if isclose(weight, 1.0):  # diff dim
        return "grey"
    elif isclose(weight, 4.34782600402832):
        # d_GH(E2, S2)^{-1} = 1.0 / 0.23
        return "black"
    elif isclose(weight, 1.298701286315918):
        # d_GH(E2, H2)^{-1} = 1.0 / 0.77
        return "red"
    else:
        # d_GH(S2, H2)^{-1} = 1.0 / 0.84
        return "blue"
