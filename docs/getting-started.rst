Getting Started
===============

Installation
------------

Add Tangent to your CMake project using FetchContent:

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(tangent
     GIT_REPOSITORY https://github.com/k-sheridan/Tangent.git
     GIT_TAG main
   )
   FetchContent_MakeAvailable(tangent)
   target_link_libraries(your_target PRIVATE Tangent::Tangent)


Requirements
------------

- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20+
- Eigen 3.4+ (auto-fetched if not found)
- Sophus 1.22+ (auto-fetched if not found)


Testing & Benchmarking
----------------------

Using Docker:

.. code-block:: bash

   docker-compose up test       # Run all Tangent tests
   docker-compose up benchmark  # Run performance benchmarks


Quick Start
-----------

**1. Define a custom error term**

.. code-block:: cpp

   #include "tangent/ErrorTerms/AutoDiffErrorTerm.h"
   using namespace Tangent;

   // Error term: wants two scalars to be equal
   class DifferenceError
       : public AutoDiffErrorTerm<DifferenceError, double, 1,
                                  SimpleScalar, SimpleScalar> {
    public:
     DifferenceError(VariableKey<SimpleScalar> k1, VariableKey<SimpleScalar> k2) {
       std::get<0>(variableKeys) = k1;
       std::get<1>(variableKeys) = k2;
       information.setIdentity();
     }

     // Templated error function - Jacobians are computed automatically
     template <typename T, typename Scalar1, typename Scalar2>
     Eigen::Matrix<T, 1, 1> computeError(const Scalar1& v1,
                                         const Scalar2& v2) const {
       Eigen::Matrix<T, 1, 1> error;
       error(0) = v2 - v1;
       return error;
     }
   };

**2. Set up and run the optimizer**

.. code-block:: cpp

   #include "tangent/Optimization/SSEOptimizer.h"
   #include "tangent/Optimization/PSDSchurSolver.h"
   #include "tangent/Optimization/HuberLossFunction.h"

   // Define optimizer type (typically a type alias)
   using Solver = PSDSchurSolver<
       Scalar<double>,
       LossFunction<HuberLossFunction<double>>,
       ErrorTermGroup<DifferenceError>,
       VariableGroup<SimpleScalar>,
       VariableGroup<>>;  // uncorrelated variables (empty here)

   using Optimizer = SSEOptimizer<
       Scalar<double>, Solver,
       GaussianPrior<Scalar<double>, VariableGroup<SimpleScalar>>,
       Marginalizer<Scalar<double>, VariableGroup<SimpleScalar>,
                    ErrorTermGroup<DifferenceError>>,
       VariableGroup<SimpleScalar>,
       ErrorTermGroup<DifferenceError>>;

   // Create optimizer
   HuberLossFunction<double> loss(1000);
   Optimizer opt(Solver(loss));

   // Add variables
   auto k1 = opt.addVariable(SimpleScalar(0.0));
   auto k2 = opt.addVariable(SimpleScalar(10.0));

   // Add error term
   opt.addErrorTerm(DifferenceError(k1, k2));

   // Optimize
   auto result = opt.optimize();

   // Access results
   double v1 = opt.getVariable(k1).value;  // ~5.0
   double v2 = opt.getVariable(k2).value;  // ~5.0

**Key concepts:**

- **Variables**: Inherit from ``OptimizableVariableBase``, define ``dimension`` and ``update(delta)``
- **Error terms**: Inherit from ``AutoDiffErrorTerm``, implement ``computeError()`` - Jacobians are computed automatically
- **Keys**: Type-safe handles for accessing variables/error terms in containers

See :doc:`concepts/index` for more details on variables, error terms, and autodiff.

For a complete working example with SE3 poses and landmarks, see
`test/TestSSEOptimizer.cpp <https://github.com/k-sheridan/Tangent/blob/master/test/TestSSEOptimizer.cpp>`_.
