Getting Started
===============

Installation
------------

Add ArgMin to your CMake project using FetchContent:

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(argmin
     GIT_REPOSITORY https://github.com/k-sheridan/ArgMin.git
     GIT_TAG main
   )
   FetchContent_MakeAvailable(argmin)
   target_link_libraries(your_target PRIVATE ArgMin::ArgMin)


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

   docker-compose up test       # Run all ArgMin tests
   docker-compose up benchmark  # Run performance benchmarks


Quick Start
-----------

**1. Define a custom error term**

.. code-block:: cpp

   #include "argmin/ErrorTerms/ErrorTermBase.h"
   using namespace ArgMin;

   // Error term: wants two scalars to be equal
   class DifferenceError
       : public ErrorTermBase<Scalar<double>, Dimension<1>,
                              VariableGroup<SimpleScalar, SimpleScalar>> {
    public:
     DifferenceError(VariableKey<SimpleScalar> k1, VariableKey<SimpleScalar> k2) {
       std::get<0>(variableKeys) = k1;
       std::get<1>(variableKeys) = k2;
       information.setIdentity();
     }

     template <typename... Vars>
     void evaluate(VariableContainer<Vars...>& vars, bool relinearize) {
       auto& v1 = *std::get<0>(variablePointers);
       auto& v2 = *std::get<1>(variablePointers);

       // Compute residual
       residual(0) = v2.value - v1.value;

       // Compute Jacobians if requested
       if (relinearize) {
         std::get<0>(variableJacobians)(0, 0) = -1;  // d(residual)/d(v1)
         std::get<1>(variableJacobians)(0, 0) =  1;  // d(residual)/d(v2)
         linearizationValid = true;
       }
     }
   };

**2. Set up and run the optimizer**

.. code-block:: cpp

   #include "argmin/Optimization/SSEOptimizer.h"
   #include "argmin/Optimization/PSDSchurSolver.h"
   #include "argmin/Optimization/HuberLossFunction.h"

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

- **Variables**: Inherit from ``OptimizableVariable``, define ``dimension`` and ``update(delta)``
- **Error terms**: Inherit from ``ErrorTermBase``, implement ``evaluate()`` to compute residual + Jacobians
- **Keys**: Type-safe handles for accessing variables/error terms in containers

For a complete working example with SE3 poses and landmarks, see
`test/TestSSEOptimizer.cpp <https://github.com/k-sheridan/ArgMin/blob/master/test/TestSSEOptimizer.cpp>`_.
