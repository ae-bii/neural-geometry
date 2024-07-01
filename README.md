# Neural Latent Geometry Manifolds

> [!CAUTION]
> The package is not yet fully implemented for use.

This is a Python implementation heavily inspired by the approach taken in [Neural Latent Geometry Search: Product Manifold Inference via Gromov-Hausdorff-Informed Bayesian Optimization](https://arxiv.org/pdf/2309.04810.pdf).

## Description

`nlgm` is a Python package that implements the neural latent geometry search framework. This algorithm is a novel approach to infer product manifolds by leveraging Gromov-Hausdorff distances.

## Installation

To install `nlgm`, you can use pip:

```bash
pip install nlgm
```

## Usage

After installing, you can import the package and use it as follows:

```python
from nlgm import NLGM

# Initialize the NLGM optimizer
optimizer = NLGM()

# Use the optimizer on your data
optimized_data = optimizer.optimize(your_data)
```

Replace `your_data` with the data you want to optimize.

## Contributing

Contributions to `nlgm` are welcome! To contribute:

1. Fork the repository.
2. Install the pre-commit hooks using `pre-commit install`.
3. Create a new branch for your changes.
4. Make your changes in your branch.
5. Submit a pull request.

Before submitting your pull request, please make sure your changes pass all tests.

## License

`nlgm` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
