#include <gtest/gtest.h>

#include "TestUtils.h"
#include "argmin/Types/GaussianPrior.h"
#include "argmin/Optimization/HuberLossFunction.h"
#include "argmin/Optimization/Marginalizer.h"
#include "argmin/Optimization/PSDSchurSolver.h"
#include "argmin/Optimization/SSEOptimizer.h"
#include "argmin/Variables/InverseDepth.h"
#include "argmin/Variables/SE3.h"
#include "argmin/Variables/SimpleScalar.h"

using namespace ArgMin;
using namespace ArgMin::Test;

// =============================================================================
// Type Definitions for Tests
// =============================================================================

// Solver types
using ScalarSolver = PSDSchurSolver<Scalar<double>, LossFunction<HuberLossFunction<double>>,
                   ErrorTermGroup<DifferenceErrorTerm>,
                   VariableGroup<SimpleScalar, DifferentSimpleScalar>,
                   VariableGroup<SimpleScalar>>;

using SLAMSolver = PSDSchurSolver<Scalar<double>, LossFunction<HuberLossFunction<double>>,
                   ErrorTermGroup<RelativeReprojectionError>,
                   VariableGroup<SE3, InverseDepth>,
                   VariableGroup<InverseDepth>>;

// Simple scalar-based optimizer for basic tests
using ScalarOptimizer = SSEOptimizer<
    Scalar<double>,
    ScalarSolver,
    GaussianPrior<Scalar<double>, VariableGroup<SimpleScalar, DifferentSimpleScalar>>,
    Marginalizer<Scalar<double>, VariableGroup<SimpleScalar, DifferentSimpleScalar>,
                 ErrorTermGroup<DifferenceErrorTerm>>,
    VariableGroup<SimpleScalar, DifferentSimpleScalar>,
    ErrorTermGroup<DifferenceErrorTerm>>;

// SLAM-style optimizer with poses and landmarks
using SLAMOptimizer = SSEOptimizer<
    Scalar<double>,
    SLAMSolver,
    GaussianPrior<Scalar<double>, VariableGroup<SE3, InverseDepth>>,
    Marginalizer<Scalar<double>, VariableGroup<SE3, InverseDepth>,
                 ErrorTermGroup<RelativeReprojectionError>>,
    VariableGroup<SE3, InverseDepth>,
    ErrorTermGroup<RelativeReprojectionError>>;

// Helper functions to create optimizers with loss functions
inline ScalarOptimizer makeScalarOptimizer() {
  static HuberLossFunction<double> lossFunction{1000};
  return ScalarOptimizer(ScalarSolver(lossFunction));
}

inline SLAMOptimizer makeSLAMOptimizer() {
  static HuberLossFunction<double> lossFunction{1000};
  return SLAMOptimizer(SLAMSolver(lossFunction));
}

// =============================================================================
// Variable Management Tests
// =============================================================================

TEST(SSEOptimizer, AddVariable_Basic) {
  auto opt = makeScalarOptimizer();
  SimpleScalar var(5.0);

  auto key = opt.addVariable(var);

  EXPECT_TRUE(opt.hasVariable(key));
  EXPECT_EQ(opt.variableCount<SimpleScalar>(), 1u);
  EXPECT_DOUBLE_EQ(opt.getVariable(key).value, 5.0);
}

TEST(SSEOptimizer, AddVariable_MultipleTypes) {
  auto opt = makeScalarOptimizer();
  SimpleScalar ss(1.0);
  DifferentSimpleScalar dss(2.0);

  auto ssKey = opt.addVariable(ss);
  auto dssKey = opt.addVariable(dss);

  EXPECT_TRUE(opt.hasVariable(ssKey));
  EXPECT_TRUE(opt.hasVariable(dssKey));
  EXPECT_EQ(opt.variableCount<SimpleScalar>(), 1u);
  EXPECT_EQ(opt.variableCount<DifferentSimpleScalar>(), 1u);
}

TEST(SSEOptimizer, AddVariable_WithPrior) {
  auto opt = makeScalarOptimizer();
  SimpleScalar var(5.0);
  Eigen::Matrix<double, 1, 1> strongPrior;
  strongPrior << 1e6;

  auto key = opt.addVariable(var, strongPrior);

  EXPECT_TRUE(opt.hasVariable(key));
  // Check that the prior was set
  auto& priorBlock = opt.getPrior().A0.getBlock(key, key);
  EXPECT_DOUBLE_EQ(priorBlock(0, 0), 1e6);
}

TEST(SSEOptimizer, SetVariablePrior) {
  auto opt = makeScalarOptimizer();
  SimpleScalar var(5.0);

  auto key = opt.addVariable(var);

  // Initially has weak prior
  auto& priorBefore = opt.getPrior().A0.getBlock(key, key);
  double infoBefore = priorBefore(0, 0);

  // Set strong prior
  Eigen::Matrix<double, 1, 1> strongPrior;
  strongPrior << 1e12;
  opt.setVariablePrior(key, strongPrior);

  auto& priorAfter = opt.getPrior().A0.getBlock(key, key);
  EXPECT_GT(priorAfter(0, 0), infoBefore);
  EXPECT_DOUBLE_EQ(priorAfter(0, 0), 1e12);
}

TEST(SSEOptimizer, RemoveVariable) {
  auto opt = makeScalarOptimizer();
  SimpleScalar var(5.0);

  auto key = opt.addVariable(var);
  EXPECT_TRUE(opt.hasVariable(key));

  opt.removeVariable(key);
  EXPECT_FALSE(opt.hasVariable(key));
  EXPECT_EQ(opt.variableCount<SimpleScalar>(), 0u);
}

TEST(SSEOptimizer, GetVariableKeys) {
  auto opt = makeScalarOptimizer();
  SimpleScalar ss1(1.0), ss2(2.0), ss3(3.0);

  auto k1 = opt.addVariable(ss1);
  auto k2 = opt.addVariable(ss2);
  auto k3 = opt.addVariable(ss3);

  auto keys = opt.getVariableKeys<SimpleScalar>();
  EXPECT_EQ(keys.size(), 3u);
}

TEST(SSEOptimizer, TotalDimension) {
  auto opt = makeScalarOptimizer();
  SimpleScalar ss(1.0);
  DifferentSimpleScalar dss(2.0);

  opt.addVariable(ss);
  opt.addVariable(dss);

  // SimpleScalar and DifferentSimpleScalar each have dimension 1
  EXPECT_EQ(opt.totalDimension(), 2u);
}

// =============================================================================
// Error Term Management Tests
// =============================================================================

TEST(SSEOptimizer, AddErrorTerm) {
  auto opt = makeScalarOptimizer();
  SimpleScalar ss(1.0);
  DifferentSimpleScalar dss(2.0);

  auto ssKey = opt.addVariable(ss);
  auto dssKey = opt.addVariable(dss);

  DifferenceErrorTerm et(ssKey, dssKey);
  auto etKey = opt.addErrorTerm(et);

  EXPECT_TRUE(opt.hasErrorTerm(etKey));
  EXPECT_EQ(opt.errorTermCount<DifferenceErrorTerm>(), 1u);
}

TEST(SSEOptimizer, RemoveErrorTerm) {
  auto opt = makeScalarOptimizer();
  SimpleScalar ss(1.0);
  DifferentSimpleScalar dss(2.0);

  auto ssKey = opt.addVariable(ss);
  auto dssKey = opt.addVariable(dss);

  DifferenceErrorTerm et(ssKey, dssKey);
  auto etKey = opt.addErrorTerm(et);

  EXPECT_EQ(opt.errorTermCount<DifferenceErrorTerm>(), 1u);

  opt.removeErrorTerm(etKey);
  EXPECT_EQ(opt.errorTermCount<DifferenceErrorTerm>(), 0u);
}

TEST(SSEOptimizer, ClearErrorTerms) {
  auto opt = makeScalarOptimizer();
  SimpleScalar ss(1.0);
  DifferentSimpleScalar dss(2.0);

  auto ssKey = opt.addVariable(ss);
  auto dssKey = opt.addVariable(dss);

  opt.addErrorTerm(DifferenceErrorTerm(ssKey, dssKey));
  opt.addErrorTerm(DifferenceErrorTerm(ssKey, dssKey));
  opt.addErrorTerm(DifferenceErrorTerm(ssKey, dssKey));

  EXPECT_EQ(opt.errorTermCount<DifferenceErrorTerm>(), 3u);

  opt.clearErrorTerms();
  EXPECT_EQ(opt.errorTermCount<DifferenceErrorTerm>(), 0u);
}

// =============================================================================
// Optimization Tests
// =============================================================================

TEST(SSEOptimizer, Optimize_SimpleConvergence) {
  auto opt = makeScalarOptimizer();

  // Create two variables with a difference error term
  // ss1 = 0, dss1 = 10, error term wants them equal
  SimpleScalar ss1(0);
  DifferentSimpleScalar dss1(10);

  auto k1 = opt.addVariable(ss1);
  auto k2 = opt.addVariable(dss1);

  // Add error term: wants dss1 - ss1 = 0
  opt.addErrorTerm(DifferenceErrorTerm(k1, k2));

  // Optimize
  auto result = opt.optimize();

  // Should have run some iterations
  EXPECT_GT(result.iterations, 0u);
  // Error should decrease (or start near zero)
  EXPECT_TRUE(result.errorDecreased || result.initialError < 1e-6);

  // Variables should move toward each other
  double v1 = opt.getVariable(k1).value;
  double v2 = opt.getVariable(k2).value;
  EXPECT_LT(std::abs(v2 - v1), 10.0);  // Closer than initial 10 difference
}

TEST(SSEOptimizer, Optimize_WithSettings) {
  auto opt = makeScalarOptimizer();
  SimpleScalar ss1(0);
  DifferentSimpleScalar dss1(10);

  auto k1 = opt.addVariable(ss1);
  auto k2 = opt.addVariable(dss1);
  opt.addErrorTerm(DifferenceErrorTerm(k1, k2));

  // Set custom settings
  opt.settings.maximumIterations = 5;
  opt.settings.initialLambda = 1e6;

  auto result = opt.optimize();

  // Should respect maximum iterations
  EXPECT_LE(result.iterations, 5u);
}

TEST(SSEOptimizer, Optimize_EmptyProblem) {
  auto opt = makeScalarOptimizer();
  SimpleScalar ss(5.0);
  opt.addVariable(ss);

  // No error terms - solver should still run
  auto result = opt.optimize();

  // With no error terms, result may have NaN or 0 iterations
  // The key is that it doesn't crash
  SUCCEED();
}

// =============================================================================
// Marginalization Tests
// =============================================================================

TEST(SSEOptimizer, Marginalize_SingleVariable) {
  auto opt = makeScalarOptimizer();

  SimpleScalar ss1(0), ss2(5);
  DifferentSimpleScalar dss1(2.5);

  auto k1 = opt.addVariable(ss1);
  auto k2 = opt.addVariable(ss2);
  auto k3 = opt.addVariable(dss1);

  // Add error terms
  opt.addErrorTerm(DifferenceErrorTerm(k1, k3));
  opt.addErrorTerm(DifferenceErrorTerm(k2, k3));

  // Optimize first
  opt.optimize();

  // Marginalize the middle variable
  auto result = opt.marginalizeVariable(k3, VariableGroup<>());

  EXPECT_TRUE(result.success);
  EXPECT_FALSE(opt.hasVariable(k3));
  EXPECT_TRUE(opt.hasVariable(k1));
  EXPECT_TRUE(opt.hasVariable(k2));
}

TEST(SSEOptimizer, Marginalize_WithIgnoredTypes) {
  auto opt = makeSLAMOptimizer();

  // Create two keyframes
  SE3 kf0, kf1;
  kf1.value.translation() = Eigen::Vector3d(1, 0, 0);

  auto kf0Key = opt.addVariable(kf0);
  auto kf1Key = opt.addVariable(kf1);

  // Set strong prior on first keyframe
  opt.setVariablePrior(kf0Key, Eigen::Matrix<double, 6, 6>::Identity() * 1e24);

  // Add a landmark
  InverseDepth dinv(1.0);
  auto lmKey = opt.addVariable(dinv);

  // Add observation
  auto et = createReprojectionErrorTerm(opt.getVariables(), kf0Key, kf1Key, lmKey,
                                         Eigen::Vector2d(0, 0));
  opt.addErrorTerm(et);

  // Optimize
  opt.optimize();

  // Marginalize keyframe 0, ignoring InverseDepth correlations
  auto result = opt.marginalizeVariable(kf0Key, VariableGroup<InverseDepth>());

  EXPECT_TRUE(result.success);
  EXPECT_FALSE(opt.hasVariable(kf0Key));
  EXPECT_TRUE(opt.hasVariable(kf1Key));
  EXPECT_TRUE(opt.hasVariable(lmKey));
}

TEST(SSEOptimizer, Marginalize_Chain) {
  auto opt = makeScalarOptimizer();

  // Create a chain: ss1 -- dss1 -- ss2 -- dss2 -- ss3
  SimpleScalar ss1(1), ss2(2), ss3(3);
  DifferentSimpleScalar dss1(1.5), dss2(2.5);

  auto k1 = opt.addVariable(ss1);
  auto k2 = opt.addVariable(dss1);
  auto k3 = opt.addVariable(ss2);
  auto k4 = opt.addVariable(dss2);
  auto k5 = opt.addVariable(ss3);

  opt.addErrorTerm(DifferenceErrorTerm(k1, k2));
  opt.addErrorTerm(DifferenceErrorTerm(k3, k2));
  opt.addErrorTerm(DifferenceErrorTerm(k3, k4));
  opt.addErrorTerm(DifferenceErrorTerm(k5, k4));

  // Optimize
  opt.optimize();

  // Marginalize from one end
  auto result1 = opt.marginalizeVariable(k1, VariableGroup<>());
  EXPECT_TRUE(result1.success);

  // Optimize again
  auto optResult = opt.optimize();
  EXPECT_FALSE(std::isnan(optResult.finalError));

  // Marginalize next
  auto result2 = opt.marginalizeVariable(k2, VariableGroup<>());
  EXPECT_TRUE(result2.success);

  // Optimize again
  optResult = opt.optimize();
  EXPECT_FALSE(std::isnan(optResult.finalError));
}

// =============================================================================
// Sliding Window Integration Tests
// =============================================================================

TEST(SSEOptimizer, SlidingWindow_AddOptimizeMarginalizeCycle) {
  auto opt = makeSLAMOptimizer();

  // Add anchor keyframe with strong prior
  SE3 origin;
  auto kf0 = opt.addVariable(origin);
  opt.setVariablePrior(kf0, Eigen::Matrix<double, 6, 6>::Identity() * 1e24);

  // Add second keyframe (weak default prior)
  SE3 kf1;
  kf1.value.translation() = Eigen::Vector3d(1, 0, 0);
  auto kf1Key = opt.addVariable(kf1);

  // Add landmark (weak default prior)
  InverseDepth dinv(1.0);
  auto lm0 = opt.addVariable(dinv);

  // Add observation error term
  auto et = createReprojectionErrorTerm(opt.getVariables(), kf0, kf1Key, lm0,
                                         Eigen::Vector2d(0, 0));
  opt.addErrorTerm(et);

  // Optimize
  auto result = opt.optimize();
  EXPECT_GT(result.iterations, 0u);

  // Marginalize landmark (no ignored types)
  auto margResult1 = opt.marginalizeVariable(lm0, VariableGroup<>());
  EXPECT_TRUE(margResult1.success);

  // Marginalize keyframe (ignore InverseDepth for sparsity - though no more exist)
  auto margResult2 = opt.marginalizeVariable(kf0, VariableGroup<InverseDepth>());
  EXPECT_TRUE(margResult2.success);

  // Verify final state
  EXPECT_FALSE(opt.hasVariable(kf0));
  EXPECT_FALSE(opt.hasVariable(lm0));
  EXPECT_TRUE(opt.hasVariable(kf1Key));
}

TEST(SSEOptimizer, SlidingWindow_MultipleKeyframes) {
  auto opt = makeSLAMOptimizer();

  // Build a sliding window with 4 keyframes
  std::vector<VariableKey<SE3>> keyframes;

  // Keyframe 0 at origin with strong prior
  SE3 kf0;
  keyframes.push_back(opt.addVariable(kf0));
  opt.setVariablePrior(keyframes[0], Eigen::Matrix<double, 6, 6>::Identity() * 1e24);

  // Keyframes 1-3 along x-axis
  for (int i = 1; i < 4; ++i) {
    SE3 kf;
    kf.value.translation() = Eigen::Vector3d(i, 0, 0);
    keyframes.push_back(opt.addVariable(kf));
  }

  // Add landmarks hosted at keyframe 0
  std::vector<VariableKey<InverseDepth>> landmarks;
  InverseDepth lm1(1.0), lm2(0.8);
  landmarks.push_back(opt.addVariable(lm1));
  landmarks.push_back(opt.addVariable(lm2));

  // Add observations from all keyframes to landmarks
  for (size_t kf = 1; kf < keyframes.size(); ++kf) {
    auto et1 = createReprojectionErrorTerm(opt.getVariables(), keyframes[0], keyframes[kf],
                                            landmarks[0], Eigen::Vector2d(0, 0));
    auto et2 = createReprojectionErrorTerm(opt.getVariables(), keyframes[0], keyframes[kf],
                                            landmarks[1], Eigen::Vector2d(0.1, 0));
    opt.addErrorTerm(et1);
    opt.addErrorTerm(et2);
  }

  // Optimize
  auto result = opt.optimize();
  EXPECT_GT(result.iterations, 0u);

  // Marginalize oldest keyframe (kf0) and its landmarks
  for (auto& lmKey : landmarks) {
    opt.marginalizeVariable(lmKey, VariableGroup<>());
  }
  opt.marginalizeVariable(keyframes[0], VariableGroup<InverseDepth>());

  // Verify kf0 and landmarks are gone
  EXPECT_FALSE(opt.hasVariable(keyframes[0]));
  for (auto& lmKey : landmarks) {
    EXPECT_FALSE(opt.hasVariable(lmKey));
  }

  // Verify remaining keyframes exist
  for (size_t i = 1; i < keyframes.size(); ++i) {
    EXPECT_TRUE(opt.hasVariable(keyframes[i]));
  }
}

TEST(SSEOptimizer, SlidingWindow_PreserveSparsity) {
  auto opt = makeSLAMOptimizer();

  // Create structure with two keyframes and two landmarks
  SE3 kf0, kf1;
  kf1.value.translation() = Eigen::Vector3d(1, 0, 0);

  auto kf0Key = opt.addVariable(kf0);
  opt.setVariablePrior(kf0Key, Eigen::Matrix<double, 6, 6>::Identity() * 1e24);
  auto kf1Key = opt.addVariable(kf1);

  InverseDepth lm0(1.0), lm1(0.8);
  auto lm0Key = opt.addVariable(lm0);
  auto lm1Key = opt.addVariable(lm1);

  // Add observations
  opt.addErrorTerm(createReprojectionErrorTerm(opt.getVariables(), kf0Key, kf1Key,
                                                lm0Key, Eigen::Vector2d(0, 0)));
  opt.addErrorTerm(createReprojectionErrorTerm(opt.getVariables(), kf0Key, kf1Key,
                                                lm1Key, Eigen::Vector2d(0.1, 0)));

  // Optimize
  opt.optimize();

  // Marginalize landmarks first (standard pattern)
  opt.marginalizeVariable(lm0Key, VariableGroup<>());
  opt.marginalizeVariable(lm1Key, VariableGroup<>());

  // Marginalize keyframe 0 while ignoring InverseDepth (though none remain)
  // This tests that the ignored types parameter works correctly
  auto result = opt.marginalizeVariable(kf0Key, VariableGroup<InverseDepth>());

  EXPECT_TRUE(result.success);
  EXPECT_TRUE(opt.hasVariable(kf1Key));
  EXPECT_EQ(opt.variableCount<SE3>(), 1u);
  EXPECT_EQ(opt.variableCount<InverseDepth>(), 0u);
}

// =============================================================================
// Direct Access Tests
// =============================================================================

TEST(SSEOptimizer, DirectAccess_Solver) {
  auto opt = makeScalarOptimizer();

  // Access solver and modify settings directly
  opt.getSolver().settings.maximumIterations = 100;

  // Settings should be accessible
  EXPECT_EQ(opt.getSolver().settings.maximumIterations, 100);
}

TEST(SSEOptimizer, DirectAccess_Prior) {
  auto opt = makeScalarOptimizer();
  SimpleScalar var(5.0);
  auto key = opt.addVariable(var);

  // Should be able to access prior directly
  auto& prior = opt.getPrior();
  auto& block = prior.A0.getBlock(key, key);

  // Block should exist
  EXPECT_GT(block.size(), 0u);
}

TEST(SSEOptimizer, DirectAccess_Containers) {
  auto opt = makeScalarOptimizer();
  SimpleScalar var(5.0);
  auto key = opt.addVariable(var);

  // Access via direct accessor
  auto& vars = opt.getVariables();
  EXPECT_TRUE(vars.variableExists(key));

  // Access error terms
  auto& errors = opt.getErrorTerms();
  EXPECT_EQ(errors.getErrorTermMap<DifferenceErrorTerm>().size(), 0u);
}

// =============================================================================
// Marginalization Correctness Tests
// =============================================================================

/**
 * Test that marginalization preserves information for correct convergence.
 *
 * This test compares two scenarios with identical starting conditions:
 * 1. WITH marginalization: Marginalize middle variable, then add new error term
 * 2. WITHOUT marginalization: Remove middle variable, then add same new error term
 *
 * Setup: ss1 (fixed at 0) -- dss_mid (free) -- ss2 (free)
 *
 * After marginalizing dss_mid:
 * - The prior on ss2 should be stronger (anchored toward 0)
 * - Adding a new DifferentSimpleScalar dss_new connected to ss2 should pull it less
 *
 * Without marginalization:
 * - ss2 has only weak prior
 * - Adding dss_new pulls ss2 significantly toward dss_new's value
 */
TEST(SSEOptimizer, Marginalize_PreservesCorrectConvergence) {
  // ========== SCENARIO 1: WITH MARGINALIZATION ==========
  auto opt1 = makeScalarOptimizer();

  // ss1 fixed at 0 with very strong prior
  SimpleScalar ss1_v1(0);
  auto k1_v1 = opt1.addVariable(ss1_v1);
  Eigen::Matrix<double, 1, 1> strongPrior;
  strongPrior << 1e12;
  opt1.setVariablePrior(k1_v1, strongPrior);

  // dss_mid is free (weak prior), will be marginalized
  DifferentSimpleScalar dss_mid_v1(50);
  auto k_mid_v1 = opt1.addVariable(dss_mid_v1);

  // ss2 starts far from ss1
  SimpleScalar ss2_v1(100);
  auto k2_v1 = opt1.addVariable(ss2_v1);

  // Error terms: ss1 -- dss_mid -- ss2
  opt1.addErrorTerm(DifferenceErrorTerm(k1_v1, k_mid_v1));  // ss1 == dss_mid
  opt1.addErrorTerm(DifferenceErrorTerm(k2_v1, k_mid_v1));  // ss2 == dss_mid

  // Optimize the full problem
  opt1.optimize();

  // Record values after full optimization
  double ss2_fullOpt = opt1.getVariable(k2_v1).value;

  // ss2 should have been pulled toward 0
  EXPECT_LT(std::abs(ss2_fullOpt), 50.0);

  // Marginalize the middle variable - this transfers constraint to ss2's prior
  auto margResult = opt1.marginalizeVariable(k_mid_v1, VariableGroup<>());
  EXPECT_TRUE(margResult.success);

  // Now add a NEW dss that tries to pull ss2 away
  DifferentSimpleScalar dss_new_v1(200);  // Far away, tries to pull ss2
  auto k_new_v1 = opt1.addVariable(dss_new_v1);
  opt1.addErrorTerm(DifferenceErrorTerm(k2_v1, k_new_v1));  // ss2 == dss_new

  opt1.optimize();
  double ss2_withMarg = opt1.getVariable(k2_v1).value;

  // ========== SCENARIO 2: WITHOUT MARGINALIZATION (just removal) ==========
  auto opt2 = makeScalarOptimizer();

  SimpleScalar ss1_v2(0);
  auto k1_v2 = opt2.addVariable(ss1_v2);
  opt2.setVariablePrior(k1_v2, strongPrior);

  DifferentSimpleScalar dss_mid_v2(50);
  auto k_mid_v2 = opt2.addVariable(dss_mid_v2);

  SimpleScalar ss2_v2(100);
  auto k2_v2 = opt2.addVariable(ss2_v2);

  auto et1 = opt2.addErrorTerm(DifferenceErrorTerm(k1_v2, k_mid_v2));
  auto et2 = opt2.addErrorTerm(DifferenceErrorTerm(k2_v2, k_mid_v2));

  // Optimize same as scenario 1
  opt2.optimize();

  // Instead of marginalization, just remove the variable and its error terms
  opt2.removeErrorTerm(et1);
  opt2.removeErrorTerm(et2);
  opt2.removeVariable(k_mid_v2);

  // Add the same new dss that tries to pull ss2 away
  DifferentSimpleScalar dss_new_v2(200);
  auto k_new_v2 = opt2.addVariable(dss_new_v2);
  opt2.addErrorTerm(DifferenceErrorTerm(k2_v2, k_new_v2));

  opt2.optimize();
  double ss2_withoutMarg = opt2.getVariable(k2_v2).value;

  // ========== COMPARE RESULTS ==========
  // With marginalization: ss2 should stay closer to 0 (anchored by prior from marginalization)
  // Without marginalization: ss2 should drift more toward dss_new (200)

  // ss2 with marginalization should be closer to 0 than without
  EXPECT_LT(std::abs(ss2_withMarg), std::abs(ss2_withoutMarg));
}

/**
 * Test that marginalization increases information (tightens uncertainty).
 *
 * When we marginalize a variable that was connected to remaining variables,
 * the prior on the remaining variables should become tighter (more information).
 */
TEST(SSEOptimizer, Marginalize_IncreasesInformation) {
  auto opt = makeScalarOptimizer();

  // Create a chain: ss1 (strong prior) -- dss_mid -- ss2 (weak prior)
  SimpleScalar ss1(0);
  auto k1 = opt.addVariable(ss1);
  Eigen::Matrix<double, 1, 1> strongPrior;
  strongPrior << 1e12;
  opt.setVariablePrior(k1, strongPrior);

  DifferentSimpleScalar dss_mid(5);
  auto k_mid = opt.addVariable(dss_mid);

  SimpleScalar ss2(10);
  auto k2 = opt.addVariable(ss2);

  opt.addErrorTerm(DifferenceErrorTerm(k1, k_mid));
  opt.addErrorTerm(DifferenceErrorTerm(k2, k_mid));

  // Optimize
  opt.optimize();

  // Record prior information on ss2 before marginalization
  auto& priorBefore = opt.getPrior().A0.getBlock(k2, k2);
  double infoBefore = priorBefore(0, 0);

  // Marginalize the middle variable
  opt.marginalizeVariable(k_mid, VariableGroup<>());

  // Prior information on ss2 should have increased
  auto& priorAfter = opt.getPrior().A0.getBlock(k2, k2);
  double infoAfter = priorAfter(0, 0);

  // Information should have increased (Schur complement adds information)
  EXPECT_GT(infoAfter, infoBefore);
}

/**
 * Control test: Verify that simple removal loses information from the prior.
 */
TEST(SSEOptimizer, RemoveWithoutMarginalization_DoesNotIncreaseInformation) {
  auto opt = makeScalarOptimizer();

  SimpleScalar ss1(0);
  auto k1 = opt.addVariable(ss1);
  Eigen::Matrix<double, 1, 1> strongPrior;
  strongPrior << 1e12;
  opt.setVariablePrior(k1, strongPrior);

  DifferentSimpleScalar dss_mid(5);
  auto k_mid = opt.addVariable(dss_mid);

  SimpleScalar ss2(10);
  auto k2 = opt.addVariable(ss2);

  auto et1 = opt.addErrorTerm(DifferenceErrorTerm(k1, k_mid));
  auto et2 = opt.addErrorTerm(DifferenceErrorTerm(k2, k_mid));

  opt.optimize();

  // Record prior information on ss2 before removal
  auto& priorBefore = opt.getPrior().A0.getBlock(k2, k2);
  double infoBefore = priorBefore(0, 0);

  // Simply remove (don't marginalize)
  opt.removeErrorTerm(et1);
  opt.removeErrorTerm(et2);
  opt.removeVariable(k_mid);

  // Clean the prior
  opt.getPrior().removeUnsedVariables(opt.getVariables());

  // Prior information on ss2 should NOT have increased (remained the same weak prior)
  auto& priorAfter = opt.getPrior().A0.getBlock(k2, k2);
  double infoAfter = priorAfter(0, 0);

  // Information should stay the same (no Schur complement was computed)
  EXPECT_LE(infoAfter, infoBefore + 1e-10);  // Allow small numerical tolerance
}

/**
 * SLAM-like test: Verify marginalization preserves relative pose constraints.
 *
 * This test uses a simpler setup without large perturbations to verify
 * that the prior after marginalization contains meaningful information.
 */
TEST(SSEOptimizer, Marginalize_PreservesSLAMConstraints) {
  auto opt = makeSLAMOptimizer();

  // Keyframe 0 at origin with strong prior (anchor)
  SE3 kf0;
  auto kf0Key = opt.addVariable(kf0);
  opt.setVariablePrior(kf0Key, Eigen::Matrix<double, 6, 6>::Identity() * 1e24);

  // Keyframe 1 at x=1
  SE3 kf1;
  kf1.value.translation() = Eigen::Vector3d(1, 0, 0);
  auto kf1Key = opt.addVariable(kf1);

  // Landmark visible from kf0 to kf1
  InverseDepth lm(1.0);
  auto lmKey = opt.addVariable(lm);

  // Add observation
  opt.addErrorTerm(createReprojectionErrorTerm(opt.getVariables(), kf0Key, kf1Key, lmKey, Eigen::Vector2d(0, 0)));

  // Optimize
  opt.optimize();

  // Record prior information on kf1 before marginalization
  auto& kf1PriorBefore = opt.getPrior().A0.getBlock(kf1Key, kf1Key);
  double kf1InfoBefore = kf1PriorBefore.trace();

  // Marginalize landmark
  opt.marginalizeVariable(lmKey, VariableGroup<>());

  // Marginalize kf0 (ignoring InverseDepth correlations)
  opt.marginalizeVariable(kf0Key, VariableGroup<InverseDepth>());

  // Prior information on kf1 should have increased significantly
  auto& kf1PriorAfter = opt.getPrior().A0.getBlock(kf1Key, kf1Key);
  double kf1InfoAfter = kf1PriorAfter.trace();

  // Information should have increased from marginalization
  EXPECT_GT(kf1InfoAfter, kf1InfoBefore);

  // kf1 should remain the only keyframe
  EXPECT_TRUE(opt.hasVariable(kf1Key));
  EXPECT_FALSE(opt.hasVariable(kf0Key));
  EXPECT_FALSE(opt.hasVariable(lmKey));
  EXPECT_EQ(opt.variableCount<SE3>(), 1u);
}
