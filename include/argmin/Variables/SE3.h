#pragma once

#include <sophus/se3.hpp>

#include "argmin/Variables/OptimizableVariable.h"

namespace ArgMin {

/**
 * Sophus SE3 transformation.
 *
 * The update is applied as a so3 + translation update not an se3 update.
 *
 * The order of the delta vector is [rotation, translation].
 */
class SE3 : public ArgMin::OptimizableVariable<double, 6> {
 public:
  Sophus::SE3<double> value;

  SE3() = default;

  SE3(const Sophus::SE3<double> &se3) : value(se3) {}

  SE3(Sophus::SE3<double> se3) : value(se3) {}

  void update(const Eigen::Matrix<double, 6, 1> &dx) {
    value.so3() *= Sophus::SO3d::exp(dx.block<3, 1>(0, 0));
    value.translation() += dx.block<3, 1>(3, 0);
  }
};

}  // namespace ArgMin
