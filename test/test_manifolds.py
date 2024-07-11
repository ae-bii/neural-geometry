from nlgm.manifolds import ProductManifold


def test_product_manifold():
    """
    nlgm.manifolds.ProductManifold
    """
    m = ProductManifold([1])
    assert m is not None


if __name__ == "__main__":
    test_product_manifold()
