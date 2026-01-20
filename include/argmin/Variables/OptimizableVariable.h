#pragma once

#include <Eigen/Core>

namespace ArgMin {

/**
 * This base class defines the concept of an optimizable variable.
 * This allows ArgMin to update/perturb the variable with a minimal dimension
 * vector while the variable is stored in another representation.
 */
template <typename ScalarType, size_t Dimension>
class OptimizableVariable {
 public:
  typedef ScalarType scalar_type;
  static const size_t dimension = Dimension;

  /// This function will modify a perturbation to ensure that
  /// exp(dx)^(-1) = exp(-dx)
  void ensureUpdateIsRevertible(Eigen::Matrix<double, Dimension, 1> &dx) {}
};

}  // namespace ArgMin
