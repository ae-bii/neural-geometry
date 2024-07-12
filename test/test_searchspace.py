from nlgm.searchspace import manifold_to_curvature
import pytest


def test_manifold_to_curvature():
    """
    nlgm.searchspace.manifold_to_curvature
    """
    map = {"E2": 0, "S2": 1, "H2": -1}

    for manifold, curvature in map.items():
        assert manifold_to_curvature(manifold) == curvature

    with pytest.raises(ValueError):
        manifold_to_curvature("H5")


if __name__ == "__main__":
    test_manifold_to_curvature()
