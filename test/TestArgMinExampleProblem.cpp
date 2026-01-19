#include <gtest/gtest.h>

#include <random>
#include <type_traits>

#include "argmin/BlockVector.h"
#include "argmin/ErrorTermBase.h"
#include "argmin/ErrorTermValidator.h"
#include "argmin/GaussianPrior.h"
#include "argmin/HuberLossFunction.h"
#include "argmin/Key.h"
#include "argmin/Marginalizer.h"
#include "argmin/MetaHelpers.h"
#include "argmin/PSDSchurSolver.h"
#include "argmin/SSEOptimizer.h"
#include "argmin/SparseBlockMatrix.h"
#include "argmin/SparseBlockRow.h"
#include "argmin/Variables/InverseDepth.h"
#include "argmin/Variables/SE3.h"
#include "argmin/Variables/SimpleScalar.h"

#include "argmin/Logging.h"

using namespace ArgMin;

class DifferentSimpleScalar : public SimpleScalar {
 public:
  DifferentSimpleScalar(double val) : SimpleScalar(val) {}
};

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

      // Compute the jacobians.
      auto A = target.so3().matrix();
      auto B = host.so3().matrix();
      // T =
      // graph.extrinsics.getImu2CameraTransform(graph.FrameContainer{observationFrameIdx}.camID);
      // C = T(1:3, 1:3);

      auto &d = target.translation();
      auto &e = host.translation();
      // f = T(1:3, 4);

      // dinv =
      // graph.FrameContainer{landmarkFrameIdx}.landmarks{landmarkIdx}.dinv; u0
      // =
      // [graph.FrameContainer{landmarkFrameIdx}.landmarks{landmarkIdx}.bearing;
      // 1];

      // J_dinv = -projJac * dPi * (C'*A'*B*C*u0*1/dinv^2);
      dinvJacobian = dPi * (A.transpose() * B * pointInHost * 1 / inverseDepth);

      // J_dphi_o = projJac * dPi * (C' * so3Hat(A'*(B*C*u0*(1/dinv) + B*f + e -
      // d)));
      targetJacobian.block(0, 0, 2, 3) =
          -dPi * (Sophus::SO3d::hat(A.transpose() * (B * pointInHost + e - d)));
      // J_dt_o = -projJac * dPi * (C'*A');
      targetJacobian.block(0, 3, 2, 3) = dPi * (A.transpose());

      // J_dphi_o = -projJac * dPi * C'*A'*B * so3Hat(C*u0/dinv + f);
      hostJacobian.block(0, 0, 2, 3) =
          dPi * A.transpose() * B * Sophus::SO3d::hat(pointInHost);
      // J_dt_o = projJac * dPi * (C'*A');
      hostJacobian.block(0, 3, 2, 3) = -dPi * (A.transpose());

      linearizationValid = true;
    } else {
      linearizationValid = false;
    }
  }
};

/**
 * Set up a basic SLAM problem with 4 variable types. 2 correlated and 2
 * uncorrelated.
 */
class PSDSchurSolverTest : public ::testing::Test {
 protected:
  /**
   * Builds the solver with the current variables and errorterms.
   * This should only be ran once per test.
   */
  void SetUpProblem() {
    hostKey = variableContainer.insert(hostPose);
    targetKey = variableContainer.insert(targetPose);

    l1Key = variableContainer.insert(landmark1);
    l2Key = variableContainer.insert(landmark2);
    l3Key = variableContainer.insert(landmark3);

    ssKey = variableContainer.insert(unobservableScalar);
    dssKey = variableContainer.insert(uncorrelatedScalar);

    prior.addVariable(hostKey);
    prior.addVariable(targetKey);
    prior.addVariable(l1Key);
    prior.addVariable(l2Key);
    prior.addVariable(l3Key);
    prior.addVariable(ssKey);
    prior.addVariable(dssKey);
  }

  RelativeReprojectionError createErrorTerm(
      VariableKey<ArgMin::SE3> hostK, VariableKey<ArgMin::SE3> targetK,
      VariableKey<ArgMin::InverseDepth> dinvK, Eigen::Vector2d bearing) {
    const auto &dinv = variableContainer.at(dinvK).value;
    Eigen::Vector3d p =
        (variableContainer.at(targetK).value.inverse() *
         variableContainer.at(hostK).value *
         Eigen::Vector3d(bearing(0, 0) / dinv, bearing(1, 0) / dinv, 1 / dinv));
    Eigen::Vector2d z(p(0, 0) / p(2, 0), p(1, 0) / p(2, 0));
    RelativeReprojectionError errorTerm(hostK, targetK, dinvK, z, bearing);
    return errorTerm;
  }

  ArgMin::GaussianPrior<
      ArgMin::Scalar<double>,
      ArgMin::VariableGroup<ArgMin::SE3, ArgMin::InverseDepth,
                            ArgMin::SimpleScalar, DifferentSimpleScalar>>
      prior;

  ArgMin::VariableContainer<ArgMin::SE3, ArgMin::InverseDepth,
                            ArgMin::SimpleScalar, DifferentSimpleScalar>
      variableContainer;

  ArgMin::ErrorTermContainer<RelativeReprojectionError, DifferenceErrorTerm>
      errorTermContainer;

  ArgMin::HuberLossFunction<double> lossFunction =
      ArgMin::HuberLossFunction<double>(1000);

  using Solver = ArgMin::PSDSchurSolver<
      Scalar<double>, ArgMin::LossFunction<ArgMin::HuberLossFunction<double>>,
      ArgMin::ErrorTermGroup<RelativeReprojectionError, DifferenceErrorTerm>,
      ArgMin::VariableGroup<ArgMin::SE3, ArgMin::InverseDepth,
                            ArgMin::SimpleScalar, DifferentSimpleScalar>,
      ArgMin::VariableGroup<ArgMin::InverseDepth, DifferentSimpleScalar>>;

  Solver solver = Solver(lossFunction);

  using Marginalizer = ArgMin::Marginalizer<
      ArgMin::Scalar<double>,
      ArgMin::VariableGroup<ArgMin::SE3, ArgMin::InverseDepth,
                            ArgMin::SimpleScalar, DifferentSimpleScalar>,
      ArgMin::ErrorTermGroup<RelativeReprojectionError, DifferenceErrorTerm>>;

  Marginalizer marginalizer;

  // correlated variables.
  ArgMin::SE3 hostPose;
  ArgMin::SE3 targetPose;
  ArgMin::SimpleScalar unobservableScalar = ArgMin::SimpleScalar(10);
  // uncorrelated variables.
  DifferentSimpleScalar uncorrelatedScalar = DifferentSimpleScalar(20);
  ArgMin::InverseDepth landmark1 = ArgMin::InverseDepth(1);
  ArgMin::InverseDepth landmark2 = ArgMin::InverseDepth(0.75);
  ArgMin::InverseDepth landmark3 = ArgMin::InverseDepth(0.5);

  VariableKey<SE3> hostKey, targetKey;
  VariableKey<InverseDepth> l1Key, l2Key, l3Key;
  VariableKey<SimpleScalar> ssKey;
  VariableKey<DifferentSimpleScalar> dssKey;
};

TEST_F(PSDSchurSolverTest, IterateWithOnlyPrior) {
  SetUpProblem();

  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  // Since the prior is the only constraint, the dx vector should be ~0;
  EXPECT_NEAR(solver.dx.block(0, 0, solver.totalDimension, 1).norm(), 0, 1e-9);
  int previousDimension = solver.totalDimension;

  // Remove a variable and compute another iteration.
  variableContainer.erase(targetKey);
  prior.removeUnsedVariables(variableContainer);

  // Verify that the row was erased.
  EXPECT_EQ(prior.A0.getRowMap<ArgMin::SE3>().count(targetKey), 0);

  // Run another iteration from scratch and verify everything still works as
  // expected.
  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  EXPECT_EQ(solver.totalDimension, previousDimension - ArgMin::SE3::dimension);
  EXPECT_NEAR(solver.dx.block(0, 0, solver.totalDimension, 1).norm(), 0, 1e-9);
}

TEST_F(PSDSchurSolverTest, SolveSmallSlamProblem) {
  SetUpProblem();

  // Setup camera poses.
  variableContainer.at(targetKey).value.so3() =
      Sophus::SO3d::exp(Eigen::Vector3d(0, -0.1, 0));
  variableContainer.at(targetKey).value.translation() =
      Eigen::Vector3d(1, 0, 0);

  // Set up each of the error terms.
  auto errorTerm1 =
      createErrorTerm(hostKey, targetKey, l1Key, Eigen::Vector2d(0, 0));
  auto errorTermKey1 = errorTermContainer.insert(errorTerm1);

  auto errorTerm2 =
      createErrorTerm(hostKey, targetKey, l2Key, Eigen::Vector2d(0.3, 0));
  auto errorTermKey2 = errorTermContainer.insert(errorTerm2);

  auto errorTerm3 =
      createErrorTerm(hostKey, targetKey, l3Key, Eigen::Vector2d(0, 0.3));
  auto errorTermKey3 = errorTermContainer.insert(errorTerm3);

  // variableContainer.erase(ssKey);
  // variableContainer.erase(dssKey);
  // prior.removeUnsedVariables(variableContainer);
  auto ssErrorTerm = DifferenceErrorTerm(ssKey, dssKey);
  auto sserrorTermKey = errorTermContainer.insert(ssErrorTerm);

  // Verify the error terms evaluate to a zero error state.
  solver.initialize(variableContainer,
                    errorTermContainer);  // Updates the variable pointers.

  errorTermContainer.at(errorTermKey1).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey1).residual.norm(), 0, 1e-9);
  errorTermContainer.at(errorTermKey2).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey2).residual.norm(), 0, 1e-9);
  errorTermContainer.at(errorTermKey3).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey3).residual.norm(), 0, 1e-9);

  // Solve one iteration and verify that the residuals still read 0.
  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);

  // Verify A was computed correctly
  Eigen::MatrixXd A_expected;
  A_expected.setIdentity(12, 12);
  A_expected = 1e-24 * A_expected;

  auto jtj = [&](auto key) {
    const auto &hostJ =
        std::get<0>(errorTermContainer.at(key).variableJacobians);
    const auto &targetJ =
        std::get<1>(errorTermContainer.at(key).variableJacobians);
    size_t hostIdx = variableContainer.variableIndex(
        std::get<0>(errorTermContainer.at(key).variableKeys));
    size_t targetIdx = variableContainer.variableIndex(
        std::get<1>(errorTermContainer.at(key).variableKeys));
    const auto &inf = errorTermContainer.at(key).information;

    A_expected.block(hostIdx, hostIdx, 6, 6) += hostJ.transpose() * inf * hostJ;
    A_expected.block(targetIdx, targetIdx, 6, 6) +=
        targetJ.transpose() * inf * targetJ;
    A_expected.block(targetIdx, hostIdx, 6, 6) +=
        targetJ.transpose() * inf * hostJ;
    A_expected.block(hostIdx, targetIdx, 6, 6) +=
        hostJ.transpose() * inf * targetJ;
  };

  jtj(errorTermKey1);
  jtj(errorTermKey2);
  jtj(errorTermKey3);

  EXPECT_TRUE(A_expected.isApprox(solver.A.block(0, 0, 12, 12)));

  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);
  solver.applyUpdateToVariables(variableContainer);

  errorTermContainer.at(errorTermKey1).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey1).residual.norm(), 0, 1e-9);
  errorTermContainer.at(errorTermKey2).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey2).residual.norm(), 0, 1e-9);
  errorTermContainer.at(errorTermKey3).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey3).residual.norm(), 0, 1e-9);
}

TEST_F(PSDSchurSolverTest, SolveSmallSlamProblemLM) {
  SetUpProblem();

  // Setup camera poses.
  variableContainer.at(targetKey).value.so3() =
      Sophus::SO3d::exp(Eigen::Vector3d(0, -0.1, 0));
  variableContainer.at(targetKey).value.translation() =
      Eigen::Vector3d(1, 0, 0);

  // Set up each of the error terms.
  auto errorTerm1 =
      createErrorTerm(hostKey, targetKey, l1Key, Eigen::Vector2d(0, 0));
  auto errorTermKey1 = errorTermContainer.insert(errorTerm1);

  auto errorTerm2 =
      createErrorTerm(hostKey, targetKey, l2Key, Eigen::Vector2d(0.3, 0));
  auto errorTermKey2 = errorTermContainer.insert(errorTerm2);

  auto errorTerm3 =
      createErrorTerm(hostKey, targetKey, l3Key, Eigen::Vector2d(0, 0.2));
  auto errorTermKey3 = errorTermContainer.insert(errorTerm3);

  // variableContainer.erase(ssKey);
  // variableContainer.erase(dssKey);
  // prior.removeUnsedVariables(variableContainer);
  auto ssErrorTerm = DifferenceErrorTerm(ssKey, dssKey);
  auto sserrorTermKey = errorTermContainer.insert(ssErrorTerm);

  // Randomly perturb the inverse depths.
  Eigen::Matrix<double, 1, 1> perturbation(0.01);
  variableContainer.at(l1Key).update(perturbation);
  perturbation(0, 0) = 0.03;
  variableContainer.at(l2Key).update(perturbation);
  perturbation(0, 0) = -0.01;
  variableContainer.at(l3Key).update(perturbation);

  auto result = solver.solveLevenbergMarquardt(variableContainer,
                                               errorTermContainer, prior);

  LOG_INFO("Iterations: {}  final error: {}", result.whitenedSqError.size(),
           result.whitenedSqError.back());

  // Verify that the errors have been reduced.
  errorTermContainer.at(errorTermKey1).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey1).residual.norm(), 0, 1e-9);
  errorTermContainer.at(errorTermKey2).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey2).residual.norm(), 0, 1e-9);
  errorTermContainer.at(errorTermKey3).evaluate(variableContainer);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey3).residual.norm(), 0, 1e-9);

  // Marginalize the simple scalar.
  EXPECT_TRUE(marginalizer.marginalizeVariable(ssKey, prior, errorTermContainer,
                                               VariableGroup<>()));
  variableContainer.erase(ssKey);

  // Verify that the dss key part of the prior is equal to 1.
  EXPECT_NEAR(prior.A0.getBlock(dssKey, dssKey)(0, 0), 1, 1e-6);

  // Run another solve, and verify that dssKey stays the same.
  auto previousDssValue = variableContainer.at(dssKey).value;
  auto previousHostValue = variableContainer.at(hostKey).value;
  auto previousTargetValue = variableContainer.at(targetKey).value;
  result = solver.solveLevenbergMarquardt(variableContainer, errorTermContainer,
                                          prior);
  LOG_INFO("Iterations: {}  final error: {}  solver dimension: {}",
           result.whitenedSqError.size(), result.whitenedSqError.back(),
           solver.totalDimension);
  EXPECT_NEAR(previousDssValue, variableContainer.at(dssKey).value, 1e-6);
  EXPECT_NEAR(
      (previousHostValue.inverse() * variableContainer.at(hostKey).value)
          .log()
          .norm(),
      0, 1e-6);
  EXPECT_NEAR(
      (previousTargetValue.inverse() * variableContainer.at(targetKey).value)
          .log()
          .norm(),
      0, 1e-6);

  // Try a full keyframe marginalization.
  EXPECT_TRUE(marginalizer.marginalizeVariable(l1Key, prior, errorTermContainer,
                                               VariableGroup<>()));
  variableContainer.erase(l1Key);
  EXPECT_TRUE(marginalizer.marginalizeVariable(l2Key, prior, errorTermContainer,
                                               VariableGroup<>()));
  variableContainer.erase(l2Key);
  EXPECT_TRUE(marginalizer.marginalizeVariable(l3Key, prior, errorTermContainer,
                                               VariableGroup<>()));
  variableContainer.erase(l3Key);

  // Solve and verify the host and target key poses.
  // prior.removeUnsedVariables(variableContainer);
  result = solver.solveLevenbergMarquardt(variableContainer, errorTermContainer,
                                          prior);
  LOG_INFO("Iterations: {}  final error: {}  solver dimension: {}",
           result.whitenedSqError.size(), result.whitenedSqError.back(),
           solver.totalDimension);
  LOG_TRACE("Finished solve.");
  EXPECT_NEAR(previousDssValue, variableContainer.at(dssKey).value, 1e-6);
  EXPECT_NEAR(
      (previousHostValue.inverse() * variableContainer.at(hostKey).value)
          .log()
          .norm(),
      0, 1e-6);
  EXPECT_NEAR(
      (previousTargetValue.inverse() * variableContainer.at(targetKey).value)
          .log()
          .norm(),
      0, 1e-6);

  LOG_TRACE("Marginalizing host key.");

  // Marginalize the target pose while ignoring Inverse Depth correlations.
  EXPECT_TRUE(
      marginalizer.marginalizeVariable(hostKey, prior, errorTermContainer,
                                       VariableGroup<ArgMin::InverseDepth>()));
  variableContainer.erase(hostKey);

  // prior.removeUnsedVariables(variableContainer);
  result = solver.solveLevenbergMarquardt(variableContainer, errorTermContainer,
                                          prior);
  LOG_INFO("Iterations: {}  final error: {}  solver dimension: {}",
           result.whitenedSqError.size(), result.whitenedSqError.back(),
           solver.totalDimension);
  EXPECT_NEAR(previousDssValue, variableContainer.at(dssKey).value, 1e-6);
//  EXPECT_NEAR(
//      (previousHostValue.inverse() * variableContainer.at(hostKey).value)
//          .log()
//          .norm(),
//      0, 1e-6);
}

TEST(ErrorTermValidation, ValidateReprojectionError) {
  ArgMin::SE3 host, target;
  ArgMin::InverseDepth dinv(1);

  VariableContainer<ArgMin::SE3, ArgMin::InverseDepth> vc;
  auto hostKey = vc.insert(host);
  auto targetKey = vc.insert(target);
  auto dinvKey = vc.insert(dinv);
  auto z = Eigen::Vector2d(0, 0);
  auto bearing = Eigen::Vector2d(0, 0);

  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-0.1, 0.1);

  for (int i = 0; i < 100; ++i) {
    // Randomly set the variables
    vc.at(dinvKey).value += dis(gen);

    Eigen::Matrix<double, 6, 1> perturbation;
    perturbation << dis(gen), dis(gen), dis(gen), dis(gen), dis(gen), dis(gen);
    vc.at(targetKey).update(perturbation);

    perturbation << dis(gen), dis(gen), dis(gen), dis(gen), dis(gen), dis(gen);
    vc.at(hostKey).update(perturbation);

    RelativeReprojectionError errorTerm(
        hostKey, targetKey, dinvKey, z + Eigen::Vector2d(dis(gen), dis(gen)),
        bearing + Eigen::Vector2d(dis(gen), dis(gen)));

    ArgMin::ErrorTermValidator<RelativeReprojectionError> validator(errorTerm);

    EXPECT_TRUE(validator.validate(vc));

    vc.at(dinvKey) = dinv;
    vc.at(targetKey) = target;
    vc.at(hostKey) = host;
  }
}

/**
 * This test tries to verify that the marginalizer works properly.
 */
TEST_F(PSDSchurSolverTest, MarginalizationTest) {
  SetUpProblem();

  // Setup camera poses.
  variableContainer.at(targetKey).value.so3() =
      Sophus::SO3d::exp(Eigen::Vector3d(0, -0.1, 0));
  variableContainer.at(targetKey).value.translation() =
      Eigen::Vector3d(1, 0, 0);

  // Set up each of the error terms.
  auto errorTerm1 =
      createErrorTerm(hostKey, targetKey, l1Key, Eigen::Vector2d(0, 0));
  auto errorTermKey1 = errorTermContainer.insert(errorTerm1);

  auto errorTerm2 =
      createErrorTerm(hostKey, targetKey, l2Key, Eigen::Vector2d(0.3, 0));
  auto errorTermKey2 = errorTermContainer.insert(errorTerm2);

  auto errorTerm3 =
      createErrorTerm(hostKey, targetKey, l3Key, Eigen::Vector2d(0, 0.2));
  auto errorTermKey3 = errorTermContainer.insert(errorTerm3);

  // variableContainer.erase(ssKey);
  // variableContainer.erase(dssKey);
  // prior.removeUnsedVariables(variableContainer);
  auto ssErrorTerm = DifferenceErrorTerm(ssKey, dssKey);
  auto sserrorTermKey = errorTermContainer.insert(ssErrorTerm);

  // Randomly perturb the inverse depths.
  Eigen::Matrix<double, 1, 1> perturbation(0.01);
  variableContainer.at(l1Key).update(perturbation);
  perturbation(0, 0) = 0.03;
  variableContainer.at(l2Key).update(perturbation);
  perturbation(0, 0) = -0.01;
  variableContainer.at(l3Key).update(perturbation);

  auto result = solver.solveLevenbergMarquardt(variableContainer,
                                               errorTermContainer, prior);

  LOG_INFO("Iterations: {}  final error: {}", result.whitenedSqError.size(),
           result.whitenedSqError.back());

  // Verify that the errors have been reduced.
  errorTermContainer.at(errorTermKey1).evaluate(variableContainer, true);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey1).residual.norm(), 0, 1e-9);
  errorTermContainer.at(errorTermKey2).evaluate(variableContainer, true);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey2).residual.norm(), 0, 1e-9);
  errorTermContainer.at(errorTermKey3).evaluate(variableContainer, true);
  EXPECT_NEAR(errorTermContainer.at(errorTermKey3).residual.norm(), 0, 1e-9);

  // Marginalize all landmarks and guarantee that the prior captures the same
  // information.
  EXPECT_TRUE(errorTermContainer.at(errorTermKey1).linearizationValid);
  EXPECT_TRUE(errorTermContainer.at(errorTermKey2).linearizationValid);
  EXPECT_TRUE(errorTermContainer.at(errorTermKey3).linearizationValid);

  // Marginalize all landmarks.
  EXPECT_TRUE(marginalizer.marginalizeVariable(l1Key, prior, errorTermContainer,
                                               VariableGroup<>()));
  variableContainer.erase(l1Key);
  EXPECT_TRUE(marginalizer.marginalizeVariable(l2Key, prior, errorTermContainer,
                                               VariableGroup<>()));
  variableContainer.erase(l2Key);
  EXPECT_TRUE(marginalizer.marginalizeVariable(l3Key, prior, errorTermContainer,
                                               VariableGroup<>()));
  variableContainer.erase(l3Key);

  // relinearize and build the linear system again.
  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  // Now that we have marginalized all landmarks, there are no more relative
  // reprojection error terms.
  EXPECT_EQ(
      errorTermContainer.getErrorTermMap<RelativeReprojectionError>().size(),
      0);
  EXPECT_NEAR(solver.dx.block(0, 0, solver.totalDimension, 1).norm(), 0, 1e-9);
}
