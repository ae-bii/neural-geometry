from nlgm.manifolds import ProductManifold


def test_product_manifold():
    """
    nlgm.manifolds.ProductManifold
    """
    assert ProductManifold([1]) is not None


if __name__ == "__main__":
    test_product_manifold()
