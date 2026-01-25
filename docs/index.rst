Tangent Documentation
====================

**Header-only generic optimizer for manifold-based nonlinear least squares**

Tangent is a C++20 library originally designed for sliding window estimation in visual SLAM and odometry.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index


Features
--------

- SE3/SO3 manifold optimization with Lie algebra
- Built-in marginalization support through Sparse Gaussian Prior
- Sparse Schur complement solver for exploiting sparsity in uncorrelated variables
- Cache-friendly SlotMap data structures (O(1) operations)
- Compile-time type safety with template metaprogramming
- Optional parallel algorithms for multi-threaded optimization


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
