from nlgm.searchspace import manifold_to_curvature


def test_manifold_to_curvature():
    """
    nlgm.searchspace.manifold_to_curvature
    """
    assert manifold_to_curvature("E2") == 0


if __name__ == "__main__":
    test_manifold_to_curvature()
