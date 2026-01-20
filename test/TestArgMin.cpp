
#include <gtest/gtest.h>

#include <type_traits>

#include "TestUtils.h"
#include "argmin/Types/BlockVector.h"
#include "argmin/ErrorTerms/ErrorTermBase.h"
#include "argmin/Types/GaussianPrior.h"
#include "argmin/Optimization/HuberLossFunction.h"
#include "argmin/Containers/Key.h"
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

TEST(ArgMin, Basic) {}

TEST(ArgMin, SparseBlockRowOperations) {
  SE3 pose;
  InverseDepth zinv;

  using SBR = ArgMin::SparseBlockRow<Scalar<double>, Dimension<2>,
                                     ArgMin::VariableGroup<SE3, InverseDepth>>;

  SBR sbr;

  VariableContainer<SE3, InverseDepth> variableContainer;

  // insert variables
  auto se3Key1 = variableContainer.getVariableMap<SE3>().insert(pose);
  auto se3Key2 = variableContainer.getVariableMap<SE3>().insert(pose);

  auto dinvKey1 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);
  auto dinvKey2 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);

  SBR::MatrixBlock<SE3> matrix = SBR::MatrixBlock<SE3>::Constant(1);
  sbr.getVariableMap<SE3>().insert(std::make_pair(se3Key1, matrix));
  matrix = SBR::MatrixBlock<SE3>::Constant(2);
  sbr.getVariableMap<SE3>().insert(std::make_pair(se3Key2, matrix));

  SBR::MatrixBlock<InverseDepth> matrix2 =
      SBR::MatrixBlock<InverseDepth>::Constant(3);
  sbr.getVariableMap<InverseDepth>().insert(std::make_pair(dinvKey1, matrix2));

  // prepare a vector for multiplication
  EXPECT_EQ(variableContainer.totalDimensions(), 14);

  Eigen::Matrix<double, Eigen::Dynamic, 1> dx;
  dx.resize(variableContainer.totalDimensions(), 1);

  EXPECT_EQ(variableContainer.variableIndex(se3Key1), 0);
  EXPECT_EQ(variableContainer.variableIndex(se3Key2), 6);
  EXPECT_EQ(variableContainer.variableIndex(dinvKey1), 12);
  EXPECT_EQ(variableContainer.variableIndex(dinvKey2), 13);

  // try dotting with zeros
  dx.setZero();
  Eigen::Matrix<double, 2, 1> result;
  sbr.dot(variableContainer, dx, result);
  EXPECT_EQ(result(0, 0), 0);
  EXPECT_EQ(result(1, 0), 0);

  // set blocks to zero.
  sbr.setZero();

  EXPECT_EQ(sbr.getVariableMap<SE3>().at(se3Key1)(0, 0), 0);
  EXPECT_EQ(sbr.getVariableMap<SE3>().at(se3Key2)(0, 0), 0);
  EXPECT_EQ(sbr.getVariableMap<InverseDepth>().at(dinvKey1)(0, 0), 0);
}

TEST(ArgMin, SparseBlockMatrixOperations) {
  SE3 pose;
  InverseDepth zinv;

  using SBM =
      ArgMin::SparseBlockMatrix<Scalar<double>,
                                ArgMin::VariableGroup<SE3, InverseDepth>>;

  SBM sbm;

  VariableContainer<SE3, InverseDepth> variableContainer;

  // insert variables
  auto se3Key1 = variableContainer.getVariableMap<SE3>().insert(pose);
  auto se3Key2 = variableContainer.getVariableMap<SE3>().insert(pose);

  auto dinvKey1 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);
  auto dinvKey2 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);

  // Test insert.
  Eigen::Matrix<double, 6, 6> matrix = Eigen::Matrix<double, 6, 6>::Identity();
  sbm.setBlock(se3Key1, se3Key1, matrix);

  Eigen::Matrix<double, Eigen::Dynamic, 2> v =
      Eigen::Matrix<double, 14, 2>::Ones();

  // Test sparse dot product.
  Eigen::Matrix<double, Eigen::Dynamic, 2> result;
  result.resize(14, 2);
  sbm.dot(variableContainer, v, result);

  Eigen::Matrix<double, Eigen::Dynamic, 2> expectedResult =
      Eigen::Matrix<double, 14, 2>::Zero();
  expectedResult.block(0, 0, 6, 2) = Eigen::Matrix<double, 6, 2>::Ones();

  EXPECT_EQ(result, expectedResult);

  // insert another diagonal member
  matrix *= 2;
  sbm.setBlock(se3Key2, se3Key2, matrix);

  EXPECT_EQ(sbm.getBlock(se3Key2, se3Key2), matrix);

  // insert a inverse depth diagonal.
  auto matrix2 = Eigen::Matrix<double, 1, 1>::Ones();
  sbm.setBlock(dinvKey2, dinvKey2, matrix2);

  expectedResult.block(6, 0, 6, 2) = Eigen::Matrix<double, 6, 2>::Constant(2);
  expectedResult.block(13, 0, 1, 2) = Eigen::Matrix<double, 1, 2>::Ones();

  sbm.dot(variableContainer, v, result);

  EXPECT_EQ(result, expectedResult);

  // insert an off diagonal element
  matrix = Eigen::Matrix<double, 6, 6>::Identity();
  sbm.setBlock(se3Key1, se3Key2, matrix);

  expectedResult.block(0, 0, 6, 2) = Eigen::Matrix<double, 6, 2>::Constant(2);

  sbm.dot(variableContainer, v, result);

  EXPECT_EQ(result, expectedResult);

  // erase an element
  sbm.removeBlock(se3Key1, se3Key1);

  expectedResult.block(0, 0, 6, 2) = Eigen::Matrix<double, 6, 2>::Constant(1);

  sbm.dot(variableContainer, v, result);

  EXPECT_EQ(result, expectedResult);
}

TEST(ArgMin, BlockVector) {
  SE3 pose;
  InverseDepth zinv;

  using V = ArgMin::BlockVector<Scalar<double>, Dimension<1>,
                                ArgMin::VariableGroup<SE3, InverseDepth>>;

  V vec;

  VariableContainer<SE3, InverseDepth> variableContainer;

  // insert variables
  auto se3Key1 = variableContainer.getVariableMap<SE3>().insert(pose);
  auto se3Key2 = variableContainer.getVariableMap<SE3>().insert(pose);

  auto dinvKey1 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);
  auto dinvKey2 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);

  V::MatrixBlock<SE3> mat = V::MatrixBlock<SE3>::Ones();
  vec.addRowBlock(se3Key1, mat);
  V::MatrixBlock<SE3> mat2 = V::MatrixBlock<SE3>::Constant(2);
  vec.addRowBlock(se3Key2, mat2);

  EXPECT_EQ(vec.getRowBlock(se3Key2), mat2);
  EXPECT_EQ(vec.getRowBlock(se3Key1), mat);

  vec.getRowBlock(se3Key1) = mat2;

  EXPECT_EQ(vec.getRowBlock(se3Key1), mat2);

  vec.getRowBlock(se3Key1) = mat;
  EXPECT_TRUE(vec.blockExists(se3Key1));
  Eigen::Matrix<double, Eigen::Dynamic, 1> dx;
  dx.setOnes(14, 1);

  // subtract the vector from the block vec.
  vec.subtractVector(variableContainer, dx);

  // verify the values
  EXPECT_TRUE(vec.blockExists(se3Key1));
  EXPECT_EQ(vec.getRowBlock(se3Key1), V::MatrixBlock<SE3>::Zero());

  EXPECT_TRUE(vec.blockExists(se3Key2));
  EXPECT_EQ(vec.getRowBlock(se3Key2), V::MatrixBlock<SE3>::Ones());

  EXPECT_TRUE(vec.blockExists(dinvKey1));
  EXPECT_EQ(vec.getRowBlock(dinvKey1), -V::MatrixBlock<InverseDepth>::Ones());

  EXPECT_TRUE(vec.blockExists(dinvKey2));
  EXPECT_EQ(vec.getRowBlock(dinvKey2), -V::MatrixBlock<InverseDepth>::Ones());

  vec.removeRowBlock(se3Key1);

  EXPECT_FALSE(vec.blockExists(se3Key1));

  // Not safe to assume death will occur.
  // ASSERT_DEATH(vec.getRowBlock(se3Key1), "");
}

TEST(ArgMin, GaussianPrior) {
  SE3 pose;
  InverseDepth zinv;

  using Prior = ArgMin::GaussianPrior<Scalar<double>,
                                      ArgMin::VariableGroup<SE3, InverseDepth>>;
  Prior prior;

  VariableContainer<SE3, InverseDepth> variableContainer;

  // insert variables
  auto se3Key1 = variableContainer.getVariableMap<SE3>().insert(pose);
  auto se3Key2 = variableContainer.getVariableMap<SE3>().insert(pose);

  auto dinvKey1 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);
  auto dinvKey2 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);

  // Add variables to prior.
  prior.addVariable(se3Key1);
  prior.addVariable(se3Key2);
  prior.addVariable(dinvKey1);
  prior.addVariable(dinvKey2);

  Eigen::Matrix<double, Eigen::Dynamic, 1> dx;
  dx.resize(14, Eigen::NoChange);
  dx.setOnes();

  prior.update(variableContainer, dx);

  EXPECT_TRUE(prior.b0.blockExists(se3Key1));
  EXPECT_TRUE(prior.b0.getRowBlock(se3Key1).isApprox(
      Prior::BV::MatrixBlock<SE3>::Constant(-Prior::DefaultInverseVariance)));

  EXPECT_TRUE(prior.b0.blockExists(se3Key2));
  EXPECT_TRUE(prior.b0.getRowBlock(se3Key2).isApprox(
      Prior::BV::MatrixBlock<SE3>::Constant(-Prior::DefaultInverseVariance)));

  EXPECT_TRUE(prior.b0.blockExists(dinvKey1));
  EXPECT_TRUE(prior.b0.getRowBlock(dinvKey1).isApprox(
      Prior::BV::MatrixBlock<InverseDepth>::Constant(
          -Prior::DefaultInverseVariance)));

  EXPECT_TRUE(prior.b0.blockExists(dinvKey2));
  EXPECT_TRUE(prior.b0.getRowBlock(dinvKey2).isApprox(
      Prior::BV::MatrixBlock<InverseDepth>::Constant(
          -Prior::DefaultInverseVariance)));
}

TEST(ArgMin, ErrorTermBasePointer) {
  using ET = ArgMin::ErrorTermBase<ArgMin::Scalar<double>, ArgMin::Dimension<2>,
                                   ArgMin::VariableGroup<SE3, SE3>>;

  SE3 pose;
  InverseDepth zinv;
  VariableContainer<SE3, InverseDepth> variableContainer;

  // insert variables
  auto se3Key1 = variableContainer.getVariableMap<SE3>().insert(pose);
  auto se3Key2 = variableContainer.getVariableMap<SE3>().insert(pose);

  auto dinvKey1 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);
  auto dinvKey2 = variableContainer.getVariableMap<InverseDepth>().insert(zinv);

  ET errorTerm;

  std::get<0>(errorTerm.variableKeys) = se3Key1;
  std::get<1>(errorTerm.variableKeys) = se3Key2;

  std::get<0>(errorTerm.variablePointers) = nullptr;
  std::get<1>(errorTerm.variablePointers) = nullptr;

  EXPECT_FALSE(errorTerm.checkVariablePointerConsistency(variableContainer));

  errorTerm.updateVariablePointers(variableContainer);

  EXPECT_TRUE(errorTerm.checkVariablePointerConsistency(variableContainer));
}

/**
 * This is a large test. It basically sets up a simple linear problem with
 * uncorrelated variables, and solves it.
 */
TEST(ArgMin, PSDSchurSolverSimple) {
  SimpleScalar ss1 = 1;
  SimpleScalar ss2 = 2;
  DifferentSimpleScalar dss1 = 5;

  using ErrorTermSet = ArgMin::ErrorTermGroup<DifferenceErrorTerm>;

  using LS = ArgMin::PSDSchurSolver<
      Scalar<double>, ArgMin::LossFunction<ArgMin::HuberLossFunction<double>>,
      ErrorTermSet, ArgMin::VariableGroup<SimpleScalar, DifferentSimpleScalar>,
      ArgMin::VariableGroup<SimpleScalar>>;

  ArgMin::HuberLossFunction<double> lossFunction(
      1000);  // set the huber width to much higher than any residuals.

  LS solver(lossFunction);

  VariableContainer<SimpleScalar, DifferentSimpleScalar> variableContainer;

  // insert variables
  auto ssKey1 = variableContainer.getVariableMap<SimpleScalar>().insert(ss1);
  auto ssKey2 = variableContainer.getVariableMap<SimpleScalar>().insert(ss2);

  auto dssKey1 =
      variableContainer.getVariableMap<DifferentSimpleScalar>().insert(dss1);

  ErrorTermContainer<DifferenceErrorTerm> errorTermContainer;

  // insert error terms
  auto errorTermKey1 =
      errorTermContainer.getErrorTermMap<DifferenceErrorTerm>().insert(
          DifferenceErrorTerm(ssKey1, dssKey1));
  auto errorTermKey2 =
      errorTermContainer.getErrorTermMap<DifferenceErrorTerm>().insert(
          DifferenceErrorTerm(ssKey2, dssKey1));

  using Prior = ArgMin::GaussianPrior<
      Scalar<double>,
      ArgMin::VariableGroup<SimpleScalar, DifferentSimpleScalar>>;
  Prior prior;

  // Insert the variables into the prior.
  prior.addVariable(ssKey1, Prior::BV::MatrixBlock<SimpleScalar>::Constant(1));
  prior.addVariable(ssKey2, Prior::BV::MatrixBlock<SimpleScalar>::Constant(2));

  prior.addVariable(dssKey1,
                    Prior::BV::MatrixBlock<DifferentSimpleScalar>::Constant(3));

  // Add an off diagonal term to the prior
  prior.A0.getRowMap<DifferentSimpleScalar>()
      .at(dssKey1)
      .getVariableMap<SimpleScalar>()
      .insert(std::make_pair(
          ssKey2, Eigen::Matrix<double, DifferentSimpleScalar::dimension,
                                SimpleScalar::dimension>::Ones()));

  // Add to the rhs of the prior.
  prior.b0.getRowBlock(ssKey1) =
      Eigen::Matrix<double, SimpleScalar::dimension, 1>::Constant(1);
  prior.b0.getRowBlock(dssKey1) =
      Eigen::Matrix<double, DifferentSimpleScalar::dimension, 1>::Constant(2);

  // Test that initialize runs
  solver.initialize(variableContainer, errorTermContainer);

  // Verify that the prior is added the problem
  solver.setProblemToPrior(prior, variableContainer);

  // Verify that the prior has been added.
  EXPECT_TRUE(
      (std::get<LS::DVector<SimpleScalar>>(solver.D).at(ssKey1))
          ->isApprox(Prior::BV::MatrixBlock<SimpleScalar>::Constant(1)));
  EXPECT_TRUE(
      (std::get<LS::DVector<SimpleScalar>>(solver.D).at(ssKey2))
          ->isApprox(Prior::BV::MatrixBlock<SimpleScalar>::Constant(2)));
  EXPECT_TRUE(solver.A
                  .block(0, 0, DifferentSimpleScalar::dimension,
                         DifferentSimpleScalar::dimension)
                  .isApprox(Prior::BV::MatrixBlock<SimpleScalar>::Constant(3)));
  EXPECT_TRUE(
      (std::get<LS::BVector<SimpleScalar>>(solver.B).at(ssKey2))
          ->block(0, 0, DifferentSimpleScalar::dimension,
                  SimpleScalar::dimension)
          .isApprox(Eigen::Matrix<double, DifferentSimpleScalar::dimension,
                                  SimpleScalar::dimension>::Ones()));
  EXPECT_TRUE(
      solver.b_correlated.block(0, 0, DifferentSimpleScalar::dimension, 1)
          .isApprox(Eigen::Matrix<double, DifferentSimpleScalar::dimension,
                                  1>::Constant(2)));
  EXPECT_TRUE(solver.b_uncorrelated.getRowBlock(ssKey1).isApprox(
      Eigen::Matrix<double, SimpleScalar::dimension, 1>::Constant(1)));

  // Reset the off diagonal component of the prior.
  prior.A0.getRowMap<DifferentSimpleScalar>()
      .at(dssKey1)
      .getVariableMap<SimpleScalar>()
      .at(ssKey2)
      .setZero();

  // Zero the rhs.
  prior.b0.getRowBlock(ssKey1) =
      Eigen::Matrix<double, SimpleScalar::dimension, 1>::Constant(0);
  prior.b0.getRowBlock(dssKey1) =
      Eigen::Matrix<double, DifferentSimpleScalar::dimension, 1>::Constant(0);

  // Test that reset runs
  solver.setZero();

  // verify that the B matrices are the correct size.
  EXPECT_NE(std::get<0>(solver.B).at(ssKey1), std::get<0>(solver.B).end());
  EXPECT_NE(std::get<0>(solver.B).at(ssKey2), std::get<0>(solver.B).end());
  EXPECT_EQ(std::get<0>(solver.B).at(ssKey1)->rows(), 1);
  EXPECT_EQ(std::get<0>(solver.B).at(ssKey2)->rows(), 1);

  // verify that there are blocks in D
  EXPECT_NE(std::get<0>(solver.D).at(ssKey1), std::get<0>(solver.D).end());
  EXPECT_NE(std::get<0>(solver.D).at(ssKey2), std::get<0>(solver.D).end());

  if constexpr ((internal::Is_in_tuple<
                    SimpleScalar,
                    std::tuple<SimpleScalar, DifferentSimpleScalar>>::value)) {
    EXPECT_TRUE(true);
  } else {
    EXPECT_TRUE(false);
  }

  if constexpr (!(internal::Is_in_tuple<
                    SimpleScalar, std::tuple<DifferentSimpleScalar>>::value)) {
    EXPECT_TRUE(true);
  } else {
    EXPECT_TRUE(false);
  }

  EXPECT_EQ(solver.dimensionOfA, 1);

  // Test that linearize runs.
  solver.linearize(variableContainer, errorTermContainer);

  // The error terms should have been linearized.
  EXPECT_TRUE(errorTermContainer.getErrorTermMap<DifferenceErrorTerm>()
                  .at(errorTermKey1)
                  ->linearizationValid);
  EXPECT_TRUE(errorTermContainer.getErrorTermMap<DifferenceErrorTerm>()
                  .at(errorTermKey2)
                  ->linearizationValid);

  // Build the problem.
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);

  // Our LHS is computed as:
  // A0 = [3, 0, 0;
  //      0, 1, 0;
  //      0, 0, 2]
  // J_0'J_0 = [1, -1, 0;
  //            -1, 1, 0;
  //            0,  0, 0]
  // J_1'J_1 =  [1, 0, -1;
  //            0, 0, 0;
  //            -1,0, 1]
  // A = [5, -1, -1;
  //      -1, 2, 0;
  //      -1, 0, 3]
  // Verify that A is correct.
  EXPECT_TRUE(solver.A
                  .block(0, 0, DifferentSimpleScalar::dimension,
                         DifferentSimpleScalar::dimension)
                  .isApprox(Prior::BV::MatrixBlock<SimpleScalar>::Constant(5)));
  EXPECT_TRUE(
      (std::get<LS::DVector<SimpleScalar>>(solver.D).at(ssKey1))
          ->isApprox(Prior::BV::MatrixBlock<SimpleScalar>::Constant(2)));
  EXPECT_TRUE(
      (std::get<LS::DVector<SimpleScalar>>(solver.D).at(ssKey2))
          ->isApprox(Prior::BV::MatrixBlock<SimpleScalar>::Constant(3)));
  EXPECT_TRUE(
      (std::get<LS::BVector<SimpleScalar>>(solver.B).at(ssKey1))
          ->block(0, 0, DifferentSimpleScalar::dimension,
                  SimpleScalar::dimension)
          .isApprox(-Eigen::Matrix<double, DifferentSimpleScalar::dimension,
                                   SimpleScalar::dimension>::Ones()));
  EXPECT_TRUE(
      (std::get<LS::BVector<SimpleScalar>>(solver.B).at(ssKey2))
          ->block(0, 0, DifferentSimpleScalar::dimension,
                  SimpleScalar::dimension)
          .isApprox(-Eigen::Matrix<double, DifferentSimpleScalar::dimension,
                                   SimpleScalar::dimension>::Ones()));

  // The RHS is computed as
  // b0 = [0;0;0]
  // J_0'e = [1, -1, 0] * (5-1)
  // J_1'e = [1, 0, -1] * (5 - 2)
  // b = [7, -4, -3]'
  EXPECT_TRUE(
      solver.b_correlated.block(0, 0, DifferentSimpleScalar::dimension, 1)
          .isApprox(Eigen::Matrix<double, DifferentSimpleScalar::dimension,
                                  1>::Constant(-7)));
  EXPECT_TRUE(solver.b_uncorrelated.getRowBlock(ssKey1).isApprox(
      Eigen::Matrix<double, SimpleScalar::dimension, 1>::Constant(4)));
  EXPECT_TRUE(solver.b_uncorrelated.getRowBlock(ssKey2).isApprox(
      Eigen::Matrix<double, SimpleScalar::dimension, 1>::Constant(3)));

  // Test if the linear system is solved correctly.
  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  // Verify the perturbation is correct.
  EXPECT_TRUE(solver.dx.block(0, 0, 3, 1)
                  .isApprox(Eigen::Vector3d(-0.96000, 1.52000, 0.68000)));

  // Apply the update.
  solver.applyUpdateToVariables(variableContainer);

  // Verify that the variables are updated.
  EXPECT_NEAR(
      variableContainer.getVariableMap<SimpleScalar>().at(ssKey1)->value,
      ss1.value + 1.52, 1e-6);
  EXPECT_NEAR(
      variableContainer.getVariableMap<SimpleScalar>().at(ssKey2)->value,
      ss2.value + 0.68, 1e-6);
  EXPECT_NEAR(variableContainer.getVariableMap<DifferentSimpleScalar>()
                  .at(dssKey1)
                  ->value,
              dss1.value - 0.96, 1e-6);

  solver.applyUpdateToVariables<true>(variableContainer);

  // Verify the update was reverted.
  EXPECT_NEAR(
      variableContainer.getVariableMap<SimpleScalar>().at(ssKey1)->value,
      ss1.value, 1e-6);
  EXPECT_NEAR(
      variableContainer.getVariableMap<SimpleScalar>().at(ssKey2)->value,
      ss2.value, 1e-6);
  EXPECT_NEAR(variableContainer.getVariableMap<DifferentSimpleScalar>()
                  .at(dssKey1)
                  ->value,
              dss1.value, 1e-6);

  // Reapply the update
  solver.applyUpdateToVariables(variableContainer);

  // verify that the delta has not changed.
  EXPECT_TRUE(solver.dx.block(0, 0, 3, 1)
                  .isApprox(Eigen::Vector3d(-0.96000, 1.52000, 0.68000)));

  // Update the prior
  prior.update(solver.dxBlockVector);

  EXPECT_TRUE(prior.b0.blockExists(ssKey1));
  EXPECT_NEAR(prior.b0.getRowBlock(ssKey1)(0, 0), -1 * 1.52, 1e-6);

  EXPECT_TRUE(prior.b0.blockExists(ssKey2));
  EXPECT_NEAR(prior.b0.getRowBlock(ssKey2)(0, 0), -2 * 0.68, 1e-6);

  EXPECT_TRUE(prior.b0.blockExists(dssKey1));
  EXPECT_NEAR(prior.b0.getRowBlock(dssKey1)(0, 0), 3 * 0.96, 1e-6);

  // Solve another iteration and make sure that the delta vector is zero.
  solver.initialize(variableContainer, errorTermContainer);
  solver.linearize(variableContainer, errorTermContainer);
  solver.buildLinearSystem(prior, errorTermContainer, variableContainer);
  solver.solveLinearSystem(variableContainer, errorTermContainer, prior);

  // Verify the perturbation is correct.
  EXPECT_NEAR(solver.dx.block(0, 0, 3, 1).norm(), 0, 1e-6);
}

TEST(InverseDepth, BoundedUpdateTest) {
  Eigen::Matrix<double, 1, 1> update(-0.02);
  ArgMin::InverseDepth dinv(std::numeric_limits<double>::min() + 0.01);

  // Correct the update.
  dinv.ensureUpdateIsRevertible(update);

  // Apply the update then immediately revert it.
  dinv.update(update);
  dinv.update(-update);

  // ensure that the update was properly reverted.
  EXPECT_NEAR(dinv.value, 0.01, 1e-24);
}
