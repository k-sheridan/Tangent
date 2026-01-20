#pragma once

#include <limits>

#include "argmin/Variables/OptimizableVariable.h"

namespace ArgMin {

/**
 * Stores the inverse depth of an observed point.
 *
 * The inverse depth is always a member of (0, infinity].
 */
class InverseDepth : public ArgMin::OptimizableVariable<double, 1> {
 public:
  double value;

  InverseDepth() = default;

  InverseDepth(double dinv) : value(dinv) {}

  void update(const Eigen::Matrix<double, 1, 1> &dx) {
    // The inverse depth must come in as a valid
    assert(value >= 0 && value <= std::numeric_limits<double>::max());
    value += dx(0, 0);

    if (value < std::numeric_limits<double>::min()) {
      value = std::numeric_limits<double>::min();
    }
  }

  /// This function will modify a perturbation to ensure that
  /// exp(dx)^(-1) = exp(-dx)
  void ensureUpdateIsRevertible(Eigen::Matrix<double, 1, 1> &dx) {
    if (value + dx(0) < 0) {
      dx(0) = std::numeric_limits<double>::min() - value;
    }
  }
};

}  // namespace ArgMin
