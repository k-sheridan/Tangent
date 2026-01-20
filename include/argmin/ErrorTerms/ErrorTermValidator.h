#pragma once

#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Containers/Key.h"
#include "argmin/Types/MetaHelpers.h"

namespace ArgMin {

namespace internal {
template <typename Tuple>
struct errorterm_validator_helper;

template <typename... T>
struct errorterm_validator_helper<std::tuple<VariableKey<T>...>> {
  typedef VariableContainer<T...> variable_container;
  typedef std::tuple<T...> variables;
};
}  // namespace internal

/**
 * Numerically validates the jacobians of a given error term.
 */
template <class ErrorTerm>
class ErrorTermValidator {
 public:
  ErrorTerm errorTerm;
  double delta = 1e-8;
  double threshold = 1e-4;

  ErrorTermValidator(ErrorTerm errorTerm) : errorTerm(errorTerm) {}

  /**
   * Numerically differentiate the error term w.r.t to the independent variables
   * and verify that the jacobians match.
   */
  template <typename... Variables>
  bool validate(VariableContainer<Variables...>& variableContainer) {
    errorTerm.updateVariablePointers(variableContainer);

    errorTerm.evaluate(variableContainer, true);

    auto referenceJacobians = errorTerm.variableJacobians;

    bool success = true;

    internal::static_for(errorTerm.variableKeys, [&](auto i, auto& key) {
      typedef typename std::tuple_element<
          i, typename internal::errorterm_validator_helper<
                 typename ErrorTerm::VariableKeys>::variables>::type
          ThisVariable;
      using JacobianType =
          Eigen::Matrix<typename ErrorTerm::scalar_type,
                        ErrorTerm::residual_dimension, ThisVariable::dimension>;

      JacobianType J = JacobianType::Zero();

      ThisVariable previousState = variableContainer.at(key);

      for (int col = 0; col < ThisVariable::dimension; ++col) {
        Eigen::Matrix<typename ErrorTerm::scalar_type,
                      ErrorTerm::residual_dimension, 1>
            high, low;
        Eigen::Matrix<typename ThisVariable::scalar_type,
                      ThisVariable::dimension, 1>
            dx = Eigen::Matrix<typename ThisVariable::scalar_type,
                               ThisVariable::dimension, 1>::Zero();

        dx(col, 0) = delta;
        variableContainer.at(key).update(dx);

        errorTerm.evaluate(variableContainer, false);

        high = errorTerm.residual;
        variableContainer.at(key) = previousState;

        dx(col, 0) = -delta;
        variableContainer.at(key).update(dx);

        errorTerm.evaluate(variableContainer, false);

        low = errorTerm.residual;
        variableContainer.at(key) = previousState;

        J.block(0, col, ErrorTerm::residual_dimension, 1) =
            (high - low) / (2 * delta);
      }

      if (!J.isApprox(std::get<i>(referenceJacobians), threshold)) {
        std::cout << "Analytical: " << std::endl
                  << J << std::endl
                  << "Numerical: " << std::endl
                  << std::get<i>(referenceJacobians) << std::endl;
        success = false;
      }
    });

    return success;
  }
};

}  // namespace ArgMin.
