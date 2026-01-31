Getting Started (Python)
========================

Installation
------------

Install ``tangent_py`` from the project root:

.. code-block:: bash

   pip install ./python

Or in development mode, if you plan to modify the wrapper:

.. code-block:: bash

   pip install -e ./python


Requirements
------------

- Python 3.8+
- cppyy 2.0+ (installed automatically with pip)
- NumPy 1.20+
- A C++ compiler accessible to cppyy (GCC or Clang)
- Eigen and Sophus headers (bundled in the repository under ``extern/``)

.. note::

   ``tangent_py`` uses `cppyy <https://cppyy.readthedocs.io/>`_ to JIT compile
   Tangent's C++ headers at runtime. The first call to ``tg.init()`` compiles the
   headers and takes a few seconds. Subsequent calls are instant.


Testing
-------

Using Docker:

.. code-block:: bash

   docker-compose up python-test  # Run Python tests


Quick Start
-----------

**1. Define an error term**

Error terms are where the math lives, so they are written in C++ even when
working from Python. The ``error_term_template()`` helper generates the
boilerplate for you -- you just supply the error computation:

.. code-block:: python

   import tangent_py as tg

   tg.init()

   # Generate a "difference" error term: wants two scalars to be equal
   code = tg.error_term_template(
       name="DifferenceError",
       residual_dim=1,
       var_types=["SimpleScalar", "SimpleScalar"],
       compute_body="err(0) = v1 - v0;"
   )

   # JIT compile it
   tg.define_error_term(code)

The ``compute_body`` uses the same C++ syntax as ``computeError()`` in the C++
API. Variables are available as ``v0``, ``v1``, etc., and the residual vector
is ``err``.

**2. Set up and run the optimizer**

.. code-block:: python

   # Create optimizer -- just pass type names as strings
   opt = tg.Optimizer(
       variables=["SimpleScalar"],
       error_terms=["DifferenceError"]
   )

   # Add variables
   x = tg.Tangent.SimpleScalar(0.0)
   y = tg.Tangent.SimpleScalar(10.0)
   k1 = opt.add_variable(x)
   k2 = opt.add_variable(y)

   # Add the error term
   opt.add_error_term("DifferenceError", k1, k2)

   # Optimize
   result = opt.optimize()

   # Access results
   print(opt.get_variable(k1).value)  # ~5.0
   print(opt.get_variable(k2).value)  # ~5.0
   print(f"Converged: {result.converged}")
   print(f"Final error: {result.final_error}")


**Key concepts:**

- **Variables**: Use built-in types directly -- ``tg.Tangent.SimpleScalar(v)``, ``tg.Tangent.SE3()``, ``tg.Tangent.SO3()``, ``tg.Tangent.InverseDepth(v)``
- **Error terms**: Defined as C++ and JIT compiled. Use ``error_term_template()`` for common patterns, or ``define_error_term()`` with raw C++ for full control.
- **Optimizer**: ``tg.Optimizer`` handles all the C++ template machinery behind the scenes -- you just pass type names as strings.

See :doc:`concepts/index` for more details on variables, error terms, and autodiff.

For the full Python API, see :doc:`api/python`.
