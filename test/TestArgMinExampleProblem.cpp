#include <gtest/gtest.h>

#include <random>
#include <type_traits>

#include "TestUtils.h"
#include "argmin/Types/BlockVector.h"
#include "argmin/ErrorTerms/ErrorTermBase.h"
#include "argmin/ErrorTerms/ErrorTermValidator.h"
#include "argmin/Types/GaussianPrior.h"
#include "argmin/Optimization/HuberLossFunction.h"
#include "argmin/Containers/Key.h"
#include "argmin/Utilities/Logging.h"
#include "argmin/Optimization/Marginalizer.h"
#include "argmin/Types/MetaHelpers.h"
#include "argmin/Optimization/PSDSchurSolver.h"
#include "argmin/Optimization/SSEOptimizer.h"
#include "argmin/Types/SparseBlockMatrix.h"
#include "argmin/Types/SparseBlockRow.h"
#include "argmin/Variables/InverseDepth.h"
#include "argmin/Variables/SE3.h"
#include "argmin/Variables/SimpleScalar.h"

using namespace ArgMin;
using namespace ArgMin::Test;

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
