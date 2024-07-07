import torch
from typing import List, Tuple


class BasicManifold:
    """
    A base class for manifolds, providing a common interface for dimension, curvature, and base point.

    Args:
        dimension (int): The dimension of the manifold.
        curvature (float): The curvature of the manifold.
        base_point (torch.Tensor): The origin point of the tangent space.
    """

    def __init__(
        self, dimension: int, curvature: float, base_point: torch.Tensor = None
    ):
        """
        Initializes a BasicManifold object.

        Args:
            dimension (int): The dimension of the manifold.
            curvature (float): The curvature of the manifold.
            base_point (torch.Tensor, optional): The origin point of the tangent space. Defaults to None.
        """
        self.dimension = dimension
        self.curvature = torch.tensor(curvature)
        self.base_point = (
            base_point if base_point is not None else torch.zeros(dimension)
        )

    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Applies the exponential map to the given tangent vector in Euclidean space.

        Args:
            tangent_vector (torch.Tensor): The tangent vector to be mapped.

        Returns:
            torch.Tensor: The result of applying the exponential map to the tangent vector.
        """
        pass

    def distance(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Computes the geodesic distance between two points.

        Args:
            point_x (torch.Tensor): The first point.
            point_y (torch.Tensor): The second point.

        Returns:
            torch.Tensor: The Euclidean distance between the two points.
        """
        pass


class EuclideanManifold(BasicManifold):
    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Applies the exponential map to the given tangent vector in Euclidean space.

        Args:
            tangent_vector (torch.Tensor): The tangent vector to be mapped.

        Returns:
            torch.Tensor: The result of applying the exponential map to the tangent vector.
        """
        # For Euclidean, the exponential map is an identity function
        return tangent_vector

    def distance(self, point_x: torch.Tensor, point_y: torch.Tensor) -> torch.Tensor:
        """
        Computes the geodesic distance between two points in Euclidean space.

        Args:
            point_x (torch.Tensor): The first point.
            point_y (torch.Tensor): The second point.

        Returns:
            torch.Tensor: The Euclidean distance between the two points.
        """
        # Compute the Euclidean distance between point_x and point_y
        return torch.norm(point_x - point_y, dim=-1)


class SphericalManifold(BasicManifold):
    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Applies the exponential map to the given tangent vector in Spherical space.

        Args:
            tangent_vector (torch.Tensor): The tangent vector to be mapped.

        Returns:
            torch.Tensor: The result of applying the exponential map to the tangent vector.
        """
        # Compute the L2 norm of the tangent vector
        # $\sqrt{K_S}|x|$
        norm_v = torch.sqrt(torch.abs(self.curvature)) * torch.norm(
            tangent_vector, dim=-1, keepdim=True
        )

        # Compute the unit direction vector by dividing the tangent vector by its norm
        # $\frac{x}{\sqrt{K_S}\|x\|}$
        direction = tangent_vector / norm_v.clamp_min(1e-6)

        # Apply the exponential map formula for spherical manifold
        # $\cos(\sqrt{K_S}\|x\|)x_p + \sin(\sqrt{K_S}\|x\|)\frac{x}{\sqrt{K_S}\|x\|}$
        return torch.cos(norm_v) * self.base_point + torch.sin(norm_v) * direction

    def distance(self, point_x: torch.Tensor, point_y: torch.Tensor) -> torch.Tensor:
        """
        Computes the geodesic distance between two points in Spherical space.

        Args:
            point_x (torch.Tensor): The first point.
            point_y (torch.Tensor): The second point.

        Returns:
            torch.Tensor: The geodesic distance between the two points.
        """
        # Compute the inner product between point_x and point_y
        # $(x,y)_2 := \langle\mathbf{x}, \mathbf{y}\rangle$
        inner_product = (point_x * point_y).sum(dim=-1)

        # Compute the geodesic distance using the arc-cosine of the inner product
        # $\arccos(K_S * (x,y)_2) / \sqrt{|K|}$
        return torch.acos(self.curvature * inner_product.clamp(-1.0, 1.0)) / torch.sqrt(
            self.curvature
        )


class HyperbolicManifold(BasicManifold):
    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Applies the exponential map to the given tangent vector in Hyperbolic space.

        Args:
            tangent_vector (torch.Tensor): The tangent vector to be mapped.

        Returns:
            torch.Tensor: The result of applying the exponential map to the tangent vector.
        """
        # Compute the L2 norm of the tangent vector
        # $\sqrt{-K_H}|x|$
        norm_v = torch.sqrt(torch.abs(self.curvature)) * torch.norm(
            tangent_vector, dim=-1, keepdim=True
        )

        # Compute the unit direction vector by dividing the tangent vector by its norm
        # $\frac{x}{\sqrt{-K_H}\|x\|}$
        direction = tangent_vector / norm_v.clamp_min(1e-6)

        # Apply the exponential map formula for hyperbolic manifold
        # $\cosh(\sqrt{-K_H}\|x\|)x_p + \sinh(\sqrt{-K_H}\|x\|)\frac{x}{\sqrt{-K_H}\|x\|}$
        return torch.cosh(norm_v) * self.base_point + torch.sinh(norm_v) * direction

    def distance(self, point_x: torch.Tensor, point_y: torch.Tensor) -> torch.Tensor:
        """
        Computes the geodesic distance between two points in Hyperbolic space.

        Args:
            point_x (torch.Tensor): The first point.
            point_y (torch.Tensor): The second point.

        Returns:
            torch.Tensor: The geodesic distance between the two points.
        """
        # Compute the Lorentz inner product between point_x and point_y
        # $-(x_p,y_1)_L$, where $(x,y)_L$ denotes the Lorentz inner product
        inner_product = -(point_x[..., 0] * point_y[..., 0]) + (
            point_x[..., 1:] * point_y[..., 1:]
        ).sum(dim=-1)

        # Compute the geodesic distance using the arc-hyperbolic cosine of the inner product
        # $\frac{1}{\sqrt{-K_H}} \mathrm{arccosh}(K_H*(x,y)_L)$
        return torch.acosh(self.curvature * inner_product.clamp_min(1.0)) / torch.sqrt(
            -self.curvature
        )


class ProductManifold:
    """
    Represents a product manifold constructed from multiple manifold components,
    each characterized by its dimension and curvature.

    Args:
        curvatures (List[float]): A list containing the curvatures of component manifolds.

    Attributes:
        manifolds (List[BasicManifold]): A list of manifold objects representing the components of the product manifold.
        dimensions (List[int]): A list of dimensions of each component manifold.

    Note: dimension of each component manifold is assumed to be 2.
    """

    def __init__(self, curvatures: List[Tuple[float]]):
        """
        Initializes a ProductManifold object.

        Args:
            curvatures (List[float]): A list containing the curvatures of component manifolds.
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
        Applies the exponential map of each component manifold to the corresponding segment of the input latent vector
        and returns a concatenated tensor representing the projection into the product manifold space.

        Args:
            latent_vector (torch.Tensor): A latent vector in Euclidean space to be mapped to the product manifold space.
                                          Its dimension should match the sum of the dimensions of the component manifolds.

        Returns:
            torch.Tensor: A tensor representing the projection of the input latent vector into the product manifold space,
                          preserving the differentiability for gradient-based optimization.
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
        Aligns component manifolds with corresponding segments of input tangent vector.

        Args:
            tangent_vector (torch.Tensor): The input tangent vector.

        Returns:
            List[torch.Tensor]: A list of tensor segments, each corresponding to a component manifold.
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
        Computes the distance between two points in the product manifold space.

        Args:
            point_x (torch.Tensor): The first point in the product manifold space.
            point_y (torch.Tensor): The second point in the product manifold space.

        Returns:
            torch.Tensor: The distance between the two points in the product manifold space.
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
