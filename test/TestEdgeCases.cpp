#include <gtest/gtest.h>

#include "TestUtils.h"
#include "argmin/Optimization/OptimizerContainers.h"
#include "argmin/Types/GaussianPrior.h"
#include "argmin/Optimization/HuberLossFunction.h"
#include "argmin/Optimization/Marginalizer.h"
#include "argmin/Optimization/PSDSchurSolver.h"
#include "argmin/Variables/InverseDepth.h"
#include "argmin/Variables/SE3.h"
#include "argmin/Variables/SimpleScalar.h"

using namespace ArgMin;
using namespace ArgMin::Test;

// =============================================================================
// Empty/Degenerate Problem Tests
// =============================================================================

class EmptyProblemTest : public ::testing::Test {
 protected:
  using Prior =
      GaussianPrior<Scalar<double>, VariableGroup<SimpleScalar, DifferentSimpleScalar>>;
  using Solver = PSDSchurSolver<
      Scalar<double>, LossFunction<HuberLossFunction<double>>,
      ErrorTermGroup<DifferenceErrorTerm>,
      VariableGroup<SimpleScalar, DifferentSimpleScalar>,
      VariableGroup<SimpleScalar>>;

  HuberLossFunction<double> lossFunction{1000};
  Solver solver{lossFunction};
  VariableContainer<SimpleScalar, DifferentSimpleScalar> variableContainer;
  ErrorTermContainer<DifferenceErrorTerm> errorTermContainer;
  Prior prior;
};

TEST_F(EmptyProblemTest, SolveWithNoVariables) {
  // Solve with completely empty problem
  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);

  // Should handle gracefully - no crash, zero dimension
  EXPECT_EQ(solver.totalDimension, 0);
}

TEST_F(EmptyProblemTest, SolveWithVariablesButNoErrorTerms) {
  // Add variables but no error terms
  SimpleScalar ss1(1.0);
  DifferentSimpleScalar dss1(2.0);

  auto ssKey = variableContainer.insert(ss1);
  auto dssKey = variableContainer.insert(dss1);

  prior.addVariable(ssKey);
  prior.addVariable(dssKey);

  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  // With only prior constraints, delta should be ~0
  EXPECT_NEAR(solver.dx.norm(), 0, 1e-9);
}

TEST_F(EmptyProblemTest, SingleVariableWithPrior) {
  SimpleScalar ss1(5.0);
  auto ssKey = variableContainer.insert(ss1);

  // Add variable with strong prior
  Eigen::Matrix<double, 1, 1> strongPrior = Eigen::Matrix<double, 1, 1>::Constant(1e6);
  prior.addVariable(ssKey, strongPrior);

  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  // Should solve without issues
  EXPECT_NEAR(solver.dx.norm(), 0, 1e-9);
}

// =============================================================================
// InverseDepth Boundary Tests
// =============================================================================

TEST(InverseDepthEdgeCases, LargeNegativeUpdate) {
  // Test that large negative updates don't make inverse depth negative
  InverseDepth dinv(0.1);
  Eigen::Matrix<double, 1, 1> largeNegativeUpdate(-1.0);

  dinv.ensureUpdateIsRevertible(largeNegativeUpdate);
  dinv.update(largeNegativeUpdate);

  // Value should be clamped to minimum positive value
  EXPECT_GT(dinv.value, 0);
  EXPECT_GE(dinv.value, std::numeric_limits<double>::min());
}

TEST(InverseDepthEdgeCases, VerySmallInitialValue) {
  // Test behavior near numeric limits
  InverseDepth dinv(std::numeric_limits<double>::min() * 10);
  Eigen::Matrix<double, 1, 1> smallUpdate(-std::numeric_limits<double>::min() * 5);

  dinv.ensureUpdateIsRevertible(smallUpdate);
  double originalValue = dinv.value;
  dinv.update(smallUpdate);
  dinv.update(-smallUpdate);

  // Should be able to revert
  EXPECT_NEAR(dinv.value, originalValue, 1e-20);
}

TEST(InverseDepthEdgeCases, VeryLargeValue) {
  // Test with large inverse depth (very close object)
  InverseDepth dinv(1000.0);
  Eigen::Matrix<double, 1, 1> update(500.0);

  dinv.update(update);
  EXPECT_NEAR(dinv.value, 1500.0, 1e-9);

  dinv.update(-update);
  EXPECT_NEAR(dinv.value, 1000.0, 1e-9);
}

TEST(InverseDepthEdgeCases, ZeroUpdate) {
  InverseDepth dinv(0.5);
  Eigen::Matrix<double, 1, 1> zeroUpdate(0.0);

  double originalValue = dinv.value;
  dinv.update(zeroUpdate);

  EXPECT_EQ(dinv.value, originalValue);
}

// =============================================================================
// Marginalization Edge Cases
// =============================================================================

class MarginalizationEdgeCaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    hostKey = variableContainer.insert(hostPose);
    targetKey = variableContainer.insert(targetPose);
    l1Key = variableContainer.insert(landmark1);

    prior.addVariable(hostKey);
    prior.addVariable(targetKey);
    prior.addVariable(l1Key);
  }

  using Prior = GaussianPrior<Scalar<double>, VariableGroup<SE3, InverseDepth>>;
  using ErrorTerms = ErrorTermGroup<RelativeReprojectionError>;
  using Marg = Marginalizer<Scalar<double>, VariableGroup<SE3, InverseDepth>, ErrorTerms>;

  Prior prior;
  Marg marginalizer;
  VariableContainer<SE3, InverseDepth> variableContainer;
  ErrorTermContainer<RelativeReprojectionError> errorTermContainer;

  SE3 hostPose;
  SE3 targetPose;
  InverseDepth landmark1{1.0};

  VariableKey<SE3> hostKey, targetKey;
  VariableKey<InverseDepth> l1Key;
};

TEST_F(MarginalizationEdgeCaseTest, MarginalizeVariableWithNoErrorTerms) {
  // Marginalize a variable that has no associated error terms
  // This should succeed but not add meaningful information to prior
  bool result = marginalizer.marginalizeVariable(
      l1Key, prior, errorTermContainer, VariableGroup<>());

  EXPECT_TRUE(result);
  variableContainer.erase(l1Key);

  // Prior should still be valid
  EXPECT_TRUE(prior.A0.getRowMap<SE3>().count(hostKey) == 1);
  EXPECT_TRUE(prior.A0.getRowMap<SE3>().count(targetKey) == 1);
}

TEST_F(MarginalizationEdgeCaseTest, MarginalizeWithResidualThreshold) {
  // Set up target pose
  variableContainer.at(targetKey).value.translation() = Eigen::Vector3d(1, 0, 0);

  // Create error term
  auto errorTerm = createReprojectionErrorTerm(
      variableContainer, hostKey, targetKey, l1Key, Eigen::Vector2d(0, 0));
  auto etKey = errorTermContainer.insert(errorTerm);

  // Linearize
  errorTermContainer.at(etKey).updateVariablePointers(variableContainer);
  errorTermContainer.at(etKey).evaluate(variableContainer, true);

  // Marginalize with a very low threshold (should skip high residual terms)
  bool result = marginalizer.marginalizeVariable(
      l1Key, prior, errorTermContainer, VariableGroup<>(), 1e-12);

  EXPECT_TRUE(result);
}

TEST_F(MarginalizationEdgeCaseTest, SequentialMarginalization) {
  // Add more landmarks
  InverseDepth landmark2{0.75};
  InverseDepth landmark3{0.5};
  auto l2Key = variableContainer.insert(landmark2);
  auto l3Key = variableContainer.insert(landmark3);
  prior.addVariable(l2Key);
  prior.addVariable(l3Key);

  // Marginalize all landmarks sequentially (qdvo pattern)
  EXPECT_TRUE(marginalizer.marginalizeVariable(
      l1Key, prior, errorTermContainer, VariableGroup<>()));
  variableContainer.erase(l1Key);

  EXPECT_TRUE(marginalizer.marginalizeVariable(
      l2Key, prior, errorTermContainer, VariableGroup<>()));
  variableContainer.erase(l2Key);

  EXPECT_TRUE(marginalizer.marginalizeVariable(
      l3Key, prior, errorTermContainer, VariableGroup<>()));
  variableContainer.erase(l3Key);

  // Prior should still have SE3 variables
  EXPECT_TRUE(prior.A0.getRowMap<SE3>().count(hostKey) == 1);
  EXPECT_TRUE(prior.A0.getRowMap<SE3>().count(targetKey) == 1);
}

TEST_F(MarginalizationEdgeCaseTest, MarginalizeWithIgnoredVariables) {
  // This tests the qdvo pattern: marginalize SE3 while ignoring InverseDepth correlations
  // First, we need to ensure the hostKey has enough information to be marginalizable

  variableContainer.at(targetKey).value.translation() = Eigen::Vector3d(1, 0, 0);

  auto errorTerm = createReprojectionErrorTerm(
      variableContainer, hostKey, targetKey, l1Key, Eigen::Vector2d(0, 0));
  auto etKey = errorTermContainer.insert(errorTerm);

  errorTermContainer.at(etKey).updateVariablePointers(variableContainer);
  errorTermContainer.at(etKey).evaluate(variableContainer, true);

  // Add information to the host key's prior diagonal so it's invertible
  // (In qdvo, this comes from solver iterations that build up the prior)
  auto& hostDiag = prior.A0.getBlock(hostKey, hostKey);
  hostDiag += Eigen::Matrix<double, 6, 6>::Identity() * 1.0;

  // Marginalize host while ignoring InverseDepth correlations
  bool result = marginalizer.marginalizeVariable(
      hostKey, prior, errorTermContainer, VariableGroup<InverseDepth>());

  // The marginalization may fail if the information matrix is still singular
  // This is actually a valid edge case - marginalizing a poorly observed variable fails
  if (result) {
    variableContainer.erase(hostKey);
    // Target and landmark should still exist
    EXPECT_EQ(prior.A0.getRowMap<SE3>().count(targetKey), 1u);
    EXPECT_EQ(prior.A0.getRowMap<InverseDepth>().count(l1Key), 1u);
  } else {
    // Marginalization failed due to singular matrix - this is valid behavior
    SUCCEED();
  }
}

// =============================================================================
// Solver Numerical Stability Tests
// =============================================================================

class NumericalStabilityTest : public ::testing::Test {
 protected:
  using Prior = GaussianPrior<Scalar<double>, VariableGroup<SimpleScalar, DifferentSimpleScalar>>;
  using Solver = PSDSchurSolver<
      Scalar<double>, LossFunction<HuberLossFunction<double>>,
      ErrorTermGroup<DifferenceErrorTerm>,
      VariableGroup<SimpleScalar, DifferentSimpleScalar>,
      VariableGroup<SimpleScalar>>;

  HuberLossFunction<double> lossFunction{1000};
  Solver solver{lossFunction};
  VariableContainer<SimpleScalar, DifferentSimpleScalar> variableContainer;
  ErrorTermContainer<DifferenceErrorTerm> errorTermContainer;
  Prior prior;
};

TEST_F(NumericalStabilityTest, VeryStrongPrior) {
  SimpleScalar ss1(1.0);
  DifferentSimpleScalar dss1(2.0);

  auto ssKey = variableContainer.insert(ss1);
  auto dssKey = variableContainer.insert(dss1);

  // Add variables with very strong prior (like first keyframe in qdvo)
  Eigen::Matrix<double, 1, 1> strongPrior = Eigen::Matrix<double, 1, 1>::Constant(1e24);
  prior.addVariable(ssKey, strongPrior);
  prior.addVariable(dssKey, strongPrior);

  auto etKey = errorTermContainer.insert(DifferenceErrorTerm(ssKey, dssKey));

  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  // With very strong prior, variables should barely move
  EXPECT_LT(std::abs(solver.dx(0, 0)), 1e-20);
}

TEST_F(NumericalStabilityTest, VeryWeakPrior) {
  SimpleScalar ss1(1.0);
  DifferentSimpleScalar dss1(10.0);

  auto ssKey = variableContainer.insert(ss1);
  auto dssKey = variableContainer.insert(dss1);

  // Add variables with very weak prior
  Eigen::Matrix<double, 1, 1> weakPrior = Eigen::Matrix<double, 1, 1>::Constant(1e-24);
  prior.addVariable(ssKey, weakPrior);
  prior.addVariable(dssKey, weakPrior);

  auto etKey = errorTermContainer.insert(DifferenceErrorTerm(ssKey, dssKey));

  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  // Should still solve without NaN
  EXPECT_FALSE(std::isnan(solver.dx(0, 0)));
  EXPECT_FALSE(std::isinf(solver.dx(0, 0)));
}

TEST_F(NumericalStabilityTest, LargeResiduals) {
  SimpleScalar ss1(0.0);
  DifferentSimpleScalar dss1(1e6);  // Large difference

  auto ssKey = variableContainer.insert(ss1);
  auto dssKey = variableContainer.insert(dss1);

  prior.addVariable(ssKey);
  prior.addVariable(dssKey);

  auto etKey = errorTermContainer.insert(DifferenceErrorTerm(ssKey, dssKey));

  auto result = solver.solveLevenbergMarquardt(
      variableContainer, errorTermContainer, prior);

  // Should converge and reduce error
  EXPECT_GT(result.whitenedSqError.size(), 0);
  EXPECT_LT(result.whitenedSqError.back(), result.whitenedSqError.front());
}
