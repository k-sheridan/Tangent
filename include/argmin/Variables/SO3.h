#pragma once

#include <sophus/so3.hpp>

#include "argmin/Variables/OptimizableVariable.h"

namespace ArgMin {

/**
 * Sophus SO3 transformation.
 *
 * The order of the delta vector is [wx,wy,wz].
 */
class SO3 : public ArgMin::OptimizableVariable<double, 3> {
 public:
  Sophus::SO3<double> value;

  SO3() = default;

  SO3(const Sophus::SO3<double> &so3) : value(so3) {}

  void update(const Eigen::Matrix<double, 3, 1> &dx) {
    value *= Sophus::SO3d::exp(dx);
  }
};

}  // namespace ArgMin
