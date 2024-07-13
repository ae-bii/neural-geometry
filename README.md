<h3 align="center">
    <img src="https://raw.githubusercontent.com/ae-bii/neural-geometry/main/assets/nlgm_banner.png" alt="nlgm banner"/>
</h3>

--------------------------------------------------------------------------------

> [!WARNING]
> This package is still in its early stages. Updates may cause breaking changes.

Neural Geometry is a Python library designed to explore and manipulate the geometric properties of neural network latent spaces. It provides a set of tools and methods to understand the complex, high-dimensional spaces that neural networks operate in, inspired by recent approaches (e.g. Borde et al., [2023](https://arxiv.org/pdf/2309.04810.pdf)).


The primary features of Neural Geometry include:

- An implementation of the neural latent geometry search framework. This framework provides a unique approach to product manifold inference, which can be beneficial in various fields such as machine learning and data analysis.
- A selection of optimization methods to cater to different needs and requirements. These methods can be used to fine-tune the performance of the neural latent geometry search framework.

This package is designed to be compatible with popular scientific computing libraries such as NumPy and PyTorch, making it a versatile tool for researchers and developers working in these environments. Comprehensive documentation is available at [docs](https://ae-bii.github.io/neural-geometry/).

## Installation

To install Neural Geometry, you can use pip:

```bash
pip install neural-geometry
```

You can install optional packages for development or visualization using:

```bash
pip install .[dev,vis]                # install from pyproject.toml
pip install neural-geometry[dev,vis]  # install from pypi
```

## Usage

After installing, you can import the package and use it by following the [example](examples/example.py).

## Contributing

Contributions to Neural Geometry are welcome! To contribute:

1. Fork the repository.
2. Install the pre-commit hooks using `pre-commit install`.
3. Create a new branch for your changes.
4. Make your changes in your branch.
5. Submit a pull request.

Before submitting your pull request, please make sure your changes pass all tests.

## License

Neural Geometry is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
