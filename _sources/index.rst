.. nlgm documentation master file, created by on Thu Jul  4 16:03:45 2024.

Welcome to ``nlgm``'s documentation!
=======================================

.. warning::
   The package is still in its early stages. Updates may cause breaking changes.

Neural Latent Geometry Manifolds (``nlgm``) is a Python package inspired by the approach taken in `Neural Latent Geometry Search: Product Manifold Inference via Gromov-Hausdorff-Informed Bayesian Optimization <https://arxiv.org/pdf/2309.04810.pdf>`_.

Features
--------

The main high-level features include:

- Implementation of the neural latent geometry search framework, a novel approach to infer product manifolds by leveraging Gromov-Hausdorff distances.
- Various optimization methods to suit different requirements.

This package is compatible with libraries like NumPy and PyTorch.

Installation
------------

To install ``nlgm``, you can use pip:

.. code-block:: bash

   pip install nlgm

Usage
-----

After installing, you can import the package and use it by following the `example <https://github.com/ae-bii/nlgm/blob/main/examples/example.py>`_.

Contributing
------------

Contributions to ``nlgm`` are welcome! To contribute:

1. Fork the repository.
2. Install the pre-commit hooks using ``pre-commit install``.
3. Create a new branch for your changes.
4. Make your changes in your branch.
5. Submit a pull request.

Before submitting your pull request, please make sure your changes pass all tests.

License
-------

Please refer to the `LICENSE <https://github.com/ae-bii/nlgm/blob/main/LICENSE>`_ file in the repository for information on the project's license.

-------------------

**Sub-Modules:**

.. toctree::
   :maxdepth: 2

   nlgm.autoencoder
   nlgm.manifolds
   nlgm.optimizers
   nlgm.searchspace
   nlgm.train
   nlgm.visualize


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
