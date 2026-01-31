Tangent Documentation
====================

**Header-only generic optimizer for manifold-based nonlinear least squares with built in marginalization support**

Features
--------

- SE3/SO3 manifold optimization with Lie algebra
- **Python** support through JIT compilation
- Built-in marginalization support through Sparse Gaussian Prior
- Sparse Schur Solver for exploiting sparsity in uncorrelated variables
- Compile-time type safety with template metaprogramming
- Optional parallel algorithms for multi-threaded optimization


Table of Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started-cpp
   getting-started-python

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/index

.. toctree::
   :maxdepth: 2
   :caption: C++ API Reference

   api/optimization
   api/variables
   api/containers
   api/types
   api/error-terms

.. toctree::
   :maxdepth: 2
   :caption: Python API Reference

   api/python


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
