#pragma once

#include <gtest/gtest.h>

#include "argmin/ErrorTerms/ErrorTermBase.h"
#include "argmin/Variables/InverseDepth.h"
#include "argmin/Variables/SE3.h"
#include "argmin/Variables/SimpleScalar.h"

namespace ArgMin::Test {

/**
 * A SimpleScalar variant for testing type safety.
 * Used to verify that different variable types are handled correctly.
 */
class DifferentSimpleScalar : public SimpleScalar {
 public:
  DifferentSimpleScalar(double val) : SimpleScalar(val) {}
};

/**
 * Simple error term that computes the difference between two scalar variables.
 * Useful for testing the solver with minimal complexity.
 */
class DifferenceErrorTerm
    : public ErrorTermBase<Scalar<double>, Dimension<1>,
                           VariableGroup<SimpleScalar, DifferentSimpleScalar>> {
 public:
  DifferenceErrorTerm(VariableKey<SimpleScalar> key1,
                      VariableKey<DifferentSimpleScalar> key2) {
    std::get<0>(variableKeys) = key1;
    std::get<1>(variableKeys) = key2;
  }

  template <typename... Variables>
  void evaluate(VariableContainer<Variables...> &variables, bool relinearize) {
    EXPECT_TRUE(checkVariablePointerConsistency(variables));

    auto &var1 = *(std::get<0>(variablePointers));
    auto &var2 = *(std::get<1>(variablePointers));

    residual(0, 0) = var2.value - var1.value;

    if (relinearize) {
      auto &jac1 = (std::get<0>(variableJacobians));
      auto &jac2 = (std::get<1>(variableJacobians));

      jac1(0, 0) = -1;
      jac2(0, 0) = 1;

      linearizationValid = true;
    } else {
      linearizationValid = false;
    }

    information.setIdentity();
  }
};

/**
 * Relative reprojection error term for testing SLAM-like problems.
 * Models the reprojection of a 3D point from a host frame to a target frame.
 */
class RelativeReprojectionError
    : public ErrorTermBase<
          Scalar<double>, Dimension<2>,
          VariableGroup<ArgMin::SE3, ArgMin::SE3, ArgMin::InverseDepth>> {
 public:
  Eigen::Vector2d bearing;
  Eigen::Vector2d z;

  RelativeReprojectionError(VariableKey<ArgMin::SE3> hostFrame,
                            VariableKey<ArgMin::SE3> targetFrame,
                            VariableKey<ArgMin::InverseDepth> dinv,
                            Eigen::Vector2d bearingMeasurement,
                            Eigen::Vector2d bearingInHost) {
    std::get<0>(variableKeys) = hostFrame;
    std::get<1>(variableKeys) = targetFrame;
    std::get<2>(variableKeys) = dinv;
    z = bearingMeasurement;
    bearing = bearingInHost;
    information.setIdentity();
  }

  template <typename... Variables>
  void evaluate(VariableContainer<Variables...> &variables,
                bool relinearize = false) {
    EXPECT_TRUE(checkVariablePointerConsistency(variables));

    Sophus::SE3d &host = std::get<0>(variablePointers)->value;
    Sophus::SE3d &target = std::get<1>(variablePointers)->value;
    double &inverseDepth = std::get<2>(variablePointers)->value;

    EXPECT_GT(inverseDepth, 0);

    // Compute the residual.
    auto pointInHost =
        Eigen::Vector3d(bearing(0, 0) / inverseDepth,
                        bearing(1, 0) / inverseDepth, 1 / inverseDepth);
    Eigen::Vector3d pointInTarget = target.inverse() * host * pointInHost;

    EXPECT_GT(pointInTarget(2, 0), 0);

    residual = z - Eigen::Vector2d(pointInTarget(0, 0) / pointInTarget(2, 0),
                                   pointInTarget(1, 0) / pointInTarget(2, 0));

    if (relinearize) {
      Eigen::Matrix<double, 2, 6> &hostJacobian =
          (std::get<0>(variableJacobians));
      Eigen::Matrix<double, 2, 6> &targetJacobian =
          (std::get<1>(variableJacobians));
      Eigen::Matrix<double, 2, 1> &dinvJacobian =
          (std::get<2>(variableJacobians));

      Eigen::Matrix<double, 2, 3> dPi;
      dPi << 1 / pointInTarget(2, 0), 0,
          -pointInTarget(0, 0) / (pointInTarget(2, 0) * pointInTarget(2, 0)), 0,
          1 / pointInTarget(2, 0),
          -pointInTarget(1, 0) / (pointInTarget(2, 0) * pointInTarget(2, 0));

      EXPECT_NEAR(dPi(1, 1), 1 / pointInTarget(2, 0), 1e-6);

      auto A = target.so3().matrix();
      auto B = host.so3().matrix();

      auto &d = target.translation();
      auto &e = host.translation();

      dinvJacobian = dPi * (A.transpose() * B * pointInHost * 1 / inverseDepth);

      targetJacobian.block(0, 0, 2, 3) =
          -dPi * (Sophus::SO3d::hat(A.transpose() * (B * pointInHost + e - d)));
      targetJacobian.block(0, 3, 2, 3) = dPi * (A.transpose());

      hostJacobian.block(0, 0, 2, 3) =
          dPi * A.transpose() * B * Sophus::SO3d::hat(pointInHost);
      hostJacobian.block(0, 3, 2, 3) = -dPi * (A.transpose());

      linearizationValid = true;
    } else {
      linearizationValid = false;
    }
  }
};

/**
 * Helper to create a RelativeReprojectionError with correct measurement.
 * Computes the expected measurement from the current variable values.
 */
template <typename VariableContainerType>
RelativeReprojectionError createReprojectionErrorTerm(
    VariableContainerType &variableContainer,
    VariableKey<ArgMin::SE3> hostK,
    VariableKey<ArgMin::SE3> targetK,
    VariableKey<ArgMin::InverseDepth> dinvK,
    Eigen::Vector2d bearing) {
  const auto &dinv = variableContainer.at(dinvK).value;
  Eigen::Vector3d p =
      (variableContainer.at(targetK).value.inverse() *
       variableContainer.at(hostK).value *
       Eigen::Vector3d(bearing(0, 0) / dinv, bearing(1, 0) / dinv, 1 / dinv));
  Eigen::Vector2d z(p(0, 0) / p(2, 0), p(1, 0) / p(2, 0));
  return RelativeReprojectionError(hostK, targetK, dinvK, z, bearing);
}

}  // namespace ArgMin::Test
