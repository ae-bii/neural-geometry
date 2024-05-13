import numpy as np

from difflib import SequenceMatcher
from typing import AnyStr
from math import isclose
from itertools import combinations_with_replacement, chain
from manifolds import ProductManifold
from graph_search_space import compute_weight, manifold_to_curvature


def define_space_search_graph(
    spaces: list[AnyStr | int] | set[AnyStr | int],
    maxdim: int,
    connectivity: bool = True,
) -> tuple[np.ndarray, list[AnyStr | int]]:
    """
    Computes a connectivity graph for adjacent product spaces for spaces and maxdim.
    """
    if maxdim <= 0 or len(spaces) <= 0:
        return np.array([])

    spaces = sorted(spaces)

    # evil python trickery!!
    # unpacking is faster than type conversion or list comprehension
    # choose without replacement sum from i=1 to n_r of (n_s + i - 1 choose i)
    product_spaces = {
        dim: [*combinations_with_replacement(spaces, dim)]
        for dim in range(1, maxdim + 1)
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
                if dim + i < 1 or dim + i > maxdim:
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
        return (distances, indices)

    # inefficiently compute the inverse d_GH between product spaces
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if distances[i][j] != 1:
                continue

            pm1 = ProductManifold([manifold_to_curvature(m) for m in indices[i]])
            pm2 = ProductManifold([manifold_to_curvature(m) for m in indices[j]])
            dgh = compute_weight(pm1, pm2)

            distances[i][j] = dgh if dgh != 0 else 1  # different dims

    # compute
    return (distances, indices)


def adj_product_spaces(s1: list[AnyStr | int], s2: list[AnyStr | int]) -> bool:
    """
    Compare whether encoded product spaces are adjacent.
    """
    if s1 == s2:
        return True

    len_delta = abs(len(s1) - len(s2))
    total_len = len(s1) + len(s2)

    if len_delta > 1:
        return False

    matcher = SequenceMatcher()
    matcher.set_seqs(s1, s2)

    # print(f'diff ratio: {1 - matcher.ratio()}')
    # print(f'dist: {(1.0 if len_delta == 1 else 2.0) / total_len}')

    # if equal length, check that approx. Levenshtein distance is 2
    # if length differs by 1, check that approx. Levenshtein distance is 1
    return isclose(1 - matcher.ratio(), (1.0 if len_delta == 1 else 2.0) / total_len)


def get_color(weight: float) -> str:
    if isclose(weight, 1.0):  # diff dim
        return "grey"
    elif isclose(weight, 4.34782600402832):  # d_GH(E2, S2)^{-1} = 1.0 / 0.23
        return "black"
    elif isclose(weight, 1.298701286315918):  # d_GH(E2, H2)^{-1} = 1.0 / 0.77
        return "red"
    else:  # d_GH(S2, H2)^{-1} = 1.0 / 0.84
        return "blue"
