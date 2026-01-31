Automatic Differentiation
=========================

Deriving Jacobians is often quite challenging for complex error terms and error prone. Numerical differentiation is very simple, but comes at the cost of accuracy and runtime.

This is where AutoDiff comes in. Tangent has a header only implementation of AutoDiff inspired by Ceres's implementation. This allows users to define errors as if they were just implementing a function to compute errors and the compiler takes care of the rest through a special scalar type called a Jet.

Jet Numbers
-----------

Jets act like a basic scalar (float, double) but they also keep track of all partial derivatives throughout each basic operation applied to them.

For example, when multiplying two Jets, the product rule is applied automatically:

.. math::

   d(f \cdot g) = f \cdot dg + g \cdot df

.. code-block:: cpp

   // Create Jets: x = 3 with dx/dx = 1, y = 4 with dy/dy = 1
   Jet<double, 2> x(3.0, 0);  // value=3, derivatives=[1, 0]
   Jet<double, 2> y(4.0, 1);  // value=4, derivatives=[0, 1]

   auto z = x * y;
   // z.a = 12 (the value: 3 * 4)
   // z.v = [4, 3] (the partial derivatives)

The derivative vector ``z.v`` is computed using the product rule:

.. math::

   z.v = x.a \cdot y.v + y.a \cdot x.v = 3 \cdot [0, 1] + 4 \cdot [1, 0] = [4, 3]

This gives us :math:`\partial z/\partial x = 4` and :math:`\partial z/\partial y = 3`, which matches
what we expect from calculus: :math:`\partial(xy)/\partial x = y = 4` and :math:`\partial(xy)/\partial y = x = 3`.

Lifting Variables to Jets
-------------------------

Some variables have complex updates when they are not simple scalars. In these cases, converting these variables into their Jet equivalent is special.

This is done through "lifting" and during this operation, we show how the perturbation, dx, is related to the variable. You can refer to SO3.h for an example on this.

Creating an Autodiff Error Term
-------------------------------

Inherit from ``AutoDiffErrorTerm`` (see :doc:`/api/error-terms`) and implement a single templated ``computeError()`` method:

.. tab:: C++

   .. code-block:: cpp

      class DifferenceErrorAutoDiff
          : public AutoDiffErrorTerm<DifferenceErrorAutoDiff, double, 1,
                                     SimpleScalar, SimpleScalar> {
       public:
        DifferenceErrorAutoDiff(VariableKey<SimpleScalar> k1,
                                VariableKey<SimpleScalar> k2) {
          std::get<0>(variableKeys) = k1;
          std::get<1>(variableKeys) = k2;
          information.setIdentity();
        }

        // This single method works for both double (fast) and Jet (autodiff)
        template <typename T, typename Scalar1, typename Scalar2>
        Eigen::Matrix<T, 1, 1> computeError(const Scalar1& v1,
                                            const Scalar2& v2) const {
          Eigen::Matrix<T, 1, 1> error;
          error(0) = v2 - v1;
          return error;
        }
      };

.. tab:: Python

   The ``error_term_template()`` helper generates the C++ boilerplate for autodiff
   error terms. You only need to provide the error computation:

   .. code-block:: python

      import tangent_py as tg

      tg.init()

      code = tg.error_term_template(
          name="DifferenceError",
          residual_dim=1,
          var_types=["SimpleScalar", "SimpleScalar"],
          compute_body="err(0) = v1 - v0;"
      )
      tg.define_error_term(code)

   The ``compute_body`` is a C++ expression using the same autodiff-compatible
   syntax as ``computeError()``. Variables are available as ``v0``, ``v1``, etc.
   The residual vector is ``err``.

   For error terms with additional data (like a measurement or target value):

   .. code-block:: python

      code = tg.error_term_template(
          name="ScalarPrior",
          residual_dim=1,
          var_types=["SimpleScalar"],
          compute_body="err(0) = v0 - target;",
          extra_members="double target;",
          extra_constructor_params="double t",
          extra_constructor_init="target = t;"
      )
      tg.define_error_term(code)

   For full control, you can also pass raw C++ to ``define_error_term()`` directly
   (see :doc:`error-terms`).

The ``computeError`` method receives either:

- ``double`` values when computing residuals only (fast path)
- ``Jet<double, N>`` values when computing Jacobians (autodiff path)

Performance Considerations
--------------------------

Autodiff error terms have two evaluation paths:

1. **Fast path (residual only)**: When ``relinearize=false``, the error term uses
   ``getValue()`` to extract raw values and calls ``computeError<double>(...)``.
   This avoids all autodiff overhead.

2. **Autodiff path (with Jacobians)**: When ``relinearize=true``, variables are
   lifted to Jets via ``liftToJet()`` and ``computeError<Jet<T,N>>`` is called.
   The Jacobians are automatically extracted from the Jet derivatives.

For performance-critical error terms that are evaluated frequently, consider
implementing manual Jacobians (see :doc:`error-terms`). For most use cases, the autodiff
overhead is acceptable and the simplicity is worth it.
