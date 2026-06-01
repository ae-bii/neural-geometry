import torch
from typing import List


class BasicManifold:
    """
    Base manifold class with shared geometry attributes.

    Parameters
    ----------
    dimension : int
        Dimension of the manifold.
    curvature : float
        Scalar curvature of the manifold.
    base_point : torch.Tensor, optional
        Origin point of the tangent space. If omitted, a zero vector of
        length ``dimension`` is used.
    """

    def __init__(
        self, dimension: int, curvature: float, base_point: torch.Tensor = None
    ):
        """
        Initialize a manifold descriptor.

        Parameters
        ----------
        dimension : int
            Dimension of the manifold.
        curvature : float
            Scalar curvature of the manifold.
        base_point : torch.Tensor, optional
            Origin point of the tangent space.
        """
        self.dimension = dimension
        self.curvature = torch.tensor(curvature)
        self.base_point = (
            base_point if base_point is not None else torch.zeros(dimension)
        )

    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Map a tangent-space vector onto the manifold.

        Parameters
        ----------
        tangent_vector : torch.Tensor
            Tangent vector to map.

        Returns
        -------
        torch.Tensor
            Mapped point on the manifold.
        """
        pass

    def distance(self, point_x: torch.Tensor, point_y: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance between two points on the manifold.

        Parameters
        ----------
        point_x : torch.Tensor
            First point.
        point_y : torch.Tensor
            Second point.

        Returns
        -------
        torch.Tensor
            Geodesic distance between ``point_x`` and ``point_y``.
        """
        pass


class EuclideanManifold(BasicManifold):
    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Map a tangent vector in Euclidean space.

        Parameters
        ----------
        tangent_vector : torch.Tensor
            Tangent vector to map.

        Returns
        -------
        torch.Tensor
            The mapped point, equal to ``tangent_vector`` in Euclidean space.
        """
        # For Euclidean, the exponential map is an identity function
        return tangent_vector

    def distance(self, point_x: torch.Tensor, point_y: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance between two points.

        Parameters
        ----------
        point_x : torch.Tensor
            First point.
        point_y : torch.Tensor
            Second point.

        Returns
        -------
        torch.Tensor
            Euclidean distance between ``point_x`` and ``point_y``.
        """
        # Compute the Euclidean distance between point_x and point_y
        return torch.norm(point_x - point_y, dim=-1)


class SphericalManifold(BasicManifold):
    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Map a tangent vector onto a spherical manifold.

        Parameters
        ----------
        tangent_vector : torch.Tensor
            Tangent vector to map.

        Returns
        -------
        torch.Tensor
            Point on the spherical manifold after exponential mapping.
        """
        device = torch.get_device(tangent_vector)

        # Compute the L2 norm of the tangent vector
        # $\sqrt{K_S}|x|$
        norm_v = torch.sqrt(torch.abs(self.curvature.to(device))) * torch.norm(
            tangent_vector, dim=-1, keepdim=True
        )

        # Compute the unit direction vector by dividing the tangent vector by its norm
        # $\frac{x}{\sqrt{K_S}\|x\|}$
        direction = tangent_vector / norm_v.clamp_min(1e-6)

        # Apply the exponential map formula for spherical manifold
        # $\cos(\sqrt{K_S}\|x\|)x_p + \sin(\sqrt{K_S}\|x\|)\frac{x}{\sqrt{K_S}\|x\|}$
        return (
            torch.cos(norm_v) * self.base_point.to(device)
            + torch.sin(norm_v) * direction
        )

    def distance(self, point_x: torch.Tensor, point_y: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical geodesic distance between two points.

        Parameters
        ----------
        point_x : torch.Tensor
            First point.
        point_y : torch.Tensor
            Second point.

        Returns
        -------
        torch.Tensor
            Geodesic distance between ``point_x`` and ``point_y``.
        """
        device = torch.get_device(point_x)

        # Compute the inner product between point_x and point_y
        # $(x,y)_2 := \langle\mathbf{x}, \mathbf{y}\rangle$
        inner_product = (point_x * point_y).sum(dim=-1)

        # Compute the geodesic distance using the arc-cosine of the inner product
        # $\arccos(K_S * (x,y)_2) / \sqrt{|K|}$
        return torch.acos(
            self.curvature.to(device) * inner_product.clamp(-1.0, 1.0)
        ) / torch.sqrt(self.curvature)


class HyperbolicManifold(BasicManifold):
    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Map a tangent vector onto a hyperbolic manifold.

        Parameters
        ----------
        tangent_vector : torch.Tensor
            Tangent vector to map.

        Returns
        -------
        torch.Tensor
            Point on the hyperbolic manifold after exponential mapping.
        """
        device = torch.get_device(tangent_vector)

        # Compute the L2 norm of the tangent vector
        # $\sqrt{-K_H}|x|$
        norm_v = torch.sqrt(torch.abs(self.curvature.to(device))) * torch.norm(
            tangent_vector, dim=-1, keepdim=True
        )

        # Compute the unit direction vector by dividing the tangent vector by its norm
        # $\frac{x}{\sqrt{-K_H}\|x\|}$
        direction = tangent_vector / norm_v.clamp_min(1e-6)

        # Apply the exponential map formula for hyperbolic manifold
        # $\cosh(\sqrt{-K_H}\|x\|)x_p + \sinh(\sqrt{-K_H}\|x\|)\frac{x}{\sqrt{-K_H}\|x\|}$
        return (
            torch.cosh(norm_v) * self.base_point.to(device)
            + torch.sinh(norm_v) * direction
        )

    def distance(self, point_x: torch.Tensor, point_y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic geodesic distance between two points.

        Parameters
        ----------
        point_x : torch.Tensor
            First point.
        point_y : torch.Tensor
            Second point.

        Returns
        -------
        torch.Tensor
            Geodesic distance between ``point_x`` and ``point_y``.
        """
        device = torch.get_device(point_x)

        # Compute the Lorentz inner product between point_x and point_y
        # $-(x_p,y_1)_L$, where $(x,y)_L$ denotes the Lorentz inner product
        inner_product = -(point_x[..., 0] * point_y[..., 0]) + (
            point_x[..., 1:] * point_y[..., 1:]
        ).sum(dim=-1)

        # Compute the geodesic distance using the arc-hyperbolic cosine of the inner product
        # $\frac{1}{\sqrt{-K_H}} \mathrm{arccosh}(K_H*(x,y)_L)$
        return torch.acosh(
            self.curvature.to(device) * inner_product.clamp_min(1.0)
        ) / torch.sqrt(-self.curvature)


class ProductManifold:
    """
    Product manifold assembled from fixed-size component manifolds.

    Parameters
    ----------
    curvatures : list[float]
        Curvatures for each component manifold.

    Notes
    -----
    Each component manifold is assumed to have dimension 2.
    """

    def __init__(self, curvatures: List[float]):
        """
        Initialize a product manifold from curvature values.

        Parameters
        ----------
        curvatures : list[float]
            Curvatures for each component manifold.
        """
        self.manifolds = []
        self.dimensions = [2 for _ in curvatures]
        for dimension, curvature in zip(self.dimensions, curvatures):
            if curvature == 0:
                self.manifolds.append(EuclideanManifold(dimension, curvature))
            elif curvature > 0:
                self.manifolds.append(SphericalManifold(dimension, curvature))
            else:  # curvature < 0
                self.manifolds.append(HyperbolicManifold(dimension, curvature))

    def exponential_map(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Apply per-component exponential maps and concatenate the result.

        Parameters
        ----------
        latent_vector : torch.Tensor
            Latent vector in Euclidean space. Its last dimension should match
            the sum of component manifold dimensions.

        Returns
        -------
        torch.Tensor
            Projection of ``latent_vector`` into the product manifold space.
        """
        # Apply mapping projection of component manifold to corresponding segment of latent vector
        segments = self._get_segments(latent_vector)

        mapped_segments = [
            manifold.exponential_map(segment)
            for manifold, segment in zip(self.manifolds, segments)
        ]

        # Concatenate the mapped segments along the last dimension to form a single tensor
        return torch.cat(mapped_segments, dim=-1)

    def _get_segments(self, tangent_vector):
        """
        Split a vector into per-component manifold segments.

        Parameters
        ----------
        tangent_vector : torch.Tensor
            Vector to segment.

        Returns
        -------
        list[torch.Tensor]
            Segments matching ``self.dimensions``.
        """
        segments = []
        start = 0
        for dim in self.dimensions:
            end = start + dim
            segment = tangent_vector[..., start:end]  # Supports batch operations
            segments.append(segment)
            start = end
        return segments

    def distance(self, point_x: torch.Tensor, point_y: torch.Tensor) -> torch.Tensor:
        """
        Compute product-manifold distance between two points.

        Parameters
        ----------
        point_x : torch.Tensor
            First point in product-manifold coordinates.
        point_y : torch.Tensor
            Second point in product-manifold coordinates.

        Returns
        -------
        torch.Tensor
            Distance between ``point_x`` and ``point_y``.
        """
        x_segments = self._get_segments(point_x)
        y_segments = self._get_segments(point_y)

        # Compute the squared distances for each component manifold
        squared_distances = [
            manifold.distance(x_segment, y_segment) ** 2
            for manifold, x_segment, y_segment in zip(
                self.manifolds, x_segments, y_segments
            )
        ]

        # Sum the squared distances and take the square root
        return torch.sqrt(torch.stack(squared_distances, dim=-1).sum(dim=-1))
