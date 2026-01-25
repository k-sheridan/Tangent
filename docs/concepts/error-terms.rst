Error Terms
===========

Error Terms are where you define your constraints for the optimization problem.

What is an Error Term?
----------------------

Each error term can be a function of an arbitrary number of variables and produce an arbitrary error vector through the implementation of its evaluate function.

The optimizer will evaluate all errors given to it and compute their corresponding Jacobians to compute a linear approximation of the error.

The evaluate function has an optional boolean parameter to relinearize the error term around the input variables and store the jacobians inside the error term.

There are two ways for users to produce these jacobians:
- Manual: Allows for the most efficient implementation of a Jacobian, but requires manual derivation by users
- Automatic: By leveraging the JET concept (inspired by Ceres), analytical jacobians can be automatically derived by the compiler transparently to the user at the cost of some speed.

Manual Jacobians
----------------

When implementing manual Jacobians, you inherit from ``ErrorTermBase`` (see :doc:`/api/error-terms`) and implement the ``evaluate()`` method directly:

.. code-block:: cpp

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

Using Automatic Differentiation
-------------------------------

Automatic differentiation is generally slower than an optimized manual jacobian, but it is far simpler for quickly getting new constraints working.

See :doc:`autodiff` for details on the autodiff system.
