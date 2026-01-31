Variables
=========

In Tangent, every variable is treated like a Lie Group. This means that all variables need to define a manifold to optimize on.

What is a Variable?
-------------------

Every variable should provide the following:

1) a value variable: This is the actual variable that will be optimized.
2) a Dimension: This is the dimension of the manifold for a variable.
3) an update function: This describes how to apply a vector update of size dimension to the variable.
4) [Optional] a function to ensure updates are reversible: This function is called before all updates by the optimizer and can clamp an update to ensure update(dx)->update(-dx) results in a noop to the variable.
5) [Optional] a function to lift the variable into a JET variable
6) [Optional] a function to retrieve the variable from a JET variable

Manifold Updates
----------------

As we said above all variables are optimized on their manifold. This is allows us to optimize variables that must abide by some constraint. For example, a Rotation matrix has 9 parameters and must follow R*R^t = I and det(R) = 1. If we were to let the optimizer solve for all 9 paramaters freely, we would inevitably violate those constraints. However, the variable's manifold space is only given 3 dimensions and remains euclidean. This means that any delta the optimizer solves for will produce a valid rotation matrix estimate.


Built-in Variables
------------------

Tangent provides the following built in variables (see :doc:`/api/variables` for full API details):

- ``SimpleScalar``: A single double or float
- ``InverseDepth``: A variable that is constrained on [0, inf)
- ``SO3``: A 3D rotation
- ``SE3``: A 3D transformation

Creating Custom Variables
-------------------------

.. tab:: C++

   To create a custom variable, inherit from ``OptimizableVariableBase`` and implement the ``update()`` method:

   .. code-block:: cpp

      #include "tangent/Variables/OptimizableVariable.h"

      class MyScalar : public OptimizableVariableBase<double, 1> {
       public:
        double value = 0.0;

        MyScalar() = default;
        MyScalar(double v) : value(v) {}

        void update(const Eigen::Matrix<double, 1, 1>& dx) {
          value += dx(0);
        }
      };

.. tab:: Python

   Custom variable types are defined in C++. From Python, you have access to all
   built-in variable types without any additional setup:

   .. code-block:: python

      import tangent_py as tg

      tg.init()

      x = tg.Tangent.SimpleScalar(10.0)
      d = tg.Tangent.InverseDepth(0.5)
      pose = tg.Tangent.SE3()
      rot = tg.Tangent.SO3()

   If you need a custom variable type, define it in C++ and include the header in
   a C++ project, or use ``tg.define_error_term()`` to JIT compile a custom type
   for advanced use cases.


Autodiff Support
----------------

.. note::

   This section applies to C++ custom variables only. Python users working with
   the built-in variable types (``SimpleScalar``, ``SE3``, ``SO3``, ``InverseDepth``)
   do not need to implement these functions -- they are already provided.

To support :doc:`autodiff`, you must define two free functions:

.. code-block:: cpp

   // Extract the underlying value for residual-only computation.
   // Refer to SO3.h for an example.
   inline const MyVar::value_type& getValue(const MyVar& var) {
     return var.value;
   }

   // Lift the variable to Jet space with seeded perturbations.
   // This describes how the variable changes with respect to the delta.
   // Refer to SO3.h for an example.
   template <typename T, int N>
   MyVar::JetType<T, N> liftToJet(const MyVar& var, int offset) {
     // ...
   }