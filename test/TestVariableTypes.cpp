#include <gtest/gtest.h>

#include <random>

#include "argmin/Variables/InverseDepth.h"
#include "argmin/Variables/SE3.h"
#include "argmin/Variables/SO3.h"
#include "argmin/Variables/SimpleScalar.h"

using namespace ArgMin;

// =============================================================================
// SE3 Tests
// =============================================================================

class SE3Test : public ::testing::Test {
 protected:
  std::mt19937 gen{42};
  std::uniform_real_distribution<> smallAngle{-0.1, 0.1};
  std::uniform_real_distribution<> smallTrans{-1.0, 1.0};

  Eigen::Matrix<double, 6, 1> randomSmallUpdate() {
    Eigen::Matrix<double, 6, 1> dx;
    dx << smallAngle(gen), smallAngle(gen), smallAngle(gen),
          smallTrans(gen), smallTrans(gen), smallTrans(gen);
    return dx;
  }
};

TEST_F(SE3Test, DefaultConstructor) {
  SE3 pose;
  EXPECT_TRUE(pose.value.matrix().isApprox(Eigen::Matrix4d::Identity()));
}

TEST_F(SE3Test, UpdateAndRevert) {
  SE3 pose;
  Sophus::SE3d original = pose.value;

  auto dx = randomSmallUpdate();
  pose.update(dx);

  // Should have changed
  EXPECT_FALSE(pose.value.matrix().isApprox(original.matrix()));

  // Revert with negative update (approximate for SE3)
  pose.update(-dx);

  // Should be close to original (not exact due to non-commutativity)
  EXPECT_NEAR((pose.value.inverse() * original).log().norm(), 0, 1e-6);
}

TEST_F(SE3Test, ZeroUpdate) {
  SE3 pose;
  pose.value.translation() = Eigen::Vector3d(1, 2, 3);
  pose.value.so3() = Sophus::SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));

  Sophus::SE3d original = pose.value;

  Eigen::Matrix<double, 6, 1> zeroUpdate = Eigen::Matrix<double, 6, 1>::Zero();
  pose.update(zeroUpdate);

  EXPECT_TRUE(pose.value.matrix().isApprox(original.matrix()));
}

TEST_F(SE3Test, MultipleUpdates) {
  SE3 pose;

  // Apply many small random updates
  for (int i = 0; i < 100; ++i) {
    auto dx = randomSmallUpdate();
    pose.update(dx);
  }

  // Should still be a valid SE3 (rotation matrix is orthogonal)
  auto R = pose.value.rotationMatrix();
  EXPECT_TRUE((R * R.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-10));
  EXPECT_NEAR(R.determinant(), 1.0, 1e-10);
}

TEST_F(SE3Test, Dimension) {
  EXPECT_EQ(static_cast<size_t>(SE3::dimension), 6u);
}

// =============================================================================
// SO3 Tests
// =============================================================================

class SO3Test : public ::testing::Test {
 protected:
  std::mt19937 gen{42};
  std::uniform_real_distribution<> smallAngle{-0.1, 0.1};

  Eigen::Matrix<double, 3, 1> randomSmallUpdate() {
    Eigen::Matrix<double, 3, 1> dx;
    dx << smallAngle(gen), smallAngle(gen), smallAngle(gen);
    return dx;
  }
};

TEST_F(SO3Test, DefaultConstructor) {
  SO3 rot;
  EXPECT_TRUE(rot.value.matrix().isApprox(Eigen::Matrix3d::Identity()));
}

TEST_F(SO3Test, UpdateAndRevert) {
  SO3 rot;
  Sophus::SO3d original = rot.value;

  auto dx = randomSmallUpdate();
  rot.update(dx);

  // Should have changed
  EXPECT_FALSE(rot.value.matrix().isApprox(original.matrix()));

  // Revert
  rot.update(-dx);

  // Should be close to original
  EXPECT_NEAR((rot.value.inverse() * original).log().norm(), 0, 1e-6);
}

TEST_F(SO3Test, ZeroUpdate) {
  SO3 rot;
  rot.value = Sophus::SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));

  Sophus::SO3d original = rot.value;

  Eigen::Matrix<double, 3, 1> zeroUpdate = Eigen::Matrix<double, 3, 1>::Zero();
  rot.update(zeroUpdate);

  EXPECT_TRUE(rot.value.matrix().isApprox(original.matrix()));
}

TEST_F(SO3Test, MultipleUpdates) {
  SO3 rot;

  for (int i = 0; i < 100; ++i) {
    auto dx = randomSmallUpdate();
    rot.update(dx);
  }

  // Should still be a valid SO3
  auto R = rot.value.matrix();
  EXPECT_TRUE((R * R.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-10));
  EXPECT_NEAR(R.determinant(), 1.0, 1e-10);
}

TEST_F(SO3Test, LargeRotation) {
  SO3 rot;

  // Apply a large rotation (close to pi)
  Eigen::Matrix<double, 3, 1> largeUpdate;
  largeUpdate << M_PI * 0.9, 0, 0;
  rot.update(largeUpdate);

  // Should still be valid
  auto R = rot.value.matrix();
  EXPECT_TRUE((R * R.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-10));
}

TEST_F(SO3Test, Dimension) {
  EXPECT_EQ(static_cast<size_t>(SO3::dimension), 3u);
}

// =============================================================================
// InverseDepth Tests
// =============================================================================

class InverseDepthTest : public ::testing::Test {
 protected:
  std::mt19937 gen{42};
  std::uniform_real_distribution<> smallUpdate{-0.01, 0.01};
};

TEST_F(InverseDepthTest, DefaultConstructor) {
  InverseDepth dinv;
  // Default value is uninitialized, just check it compiles
  SUCCEED();
}

TEST_F(InverseDepthTest, ValueConstructor) {
  InverseDepth dinv(0.5);
  EXPECT_EQ(dinv.value, 0.5);
}

TEST_F(InverseDepthTest, UpdateAndRevert) {
  InverseDepth dinv(1.0);
  double original = dinv.value;

  Eigen::Matrix<double, 1, 1> dx;
  dx << 0.1;

  dinv.update(dx);
  EXPECT_NEAR(dinv.value, 1.1, 1e-10);

  dinv.update(-dx);
  EXPECT_NEAR(dinv.value, original, 1e-10);
}

TEST_F(InverseDepthTest, ZeroUpdate) {
  InverseDepth dinv(0.75);
  double original = dinv.value;

  Eigen::Matrix<double, 1, 1> zeroUpdate;
  zeroUpdate << 0.0;

  dinv.update(zeroUpdate);
  EXPECT_EQ(dinv.value, original);
}

TEST_F(InverseDepthTest, ClampToPositive) {
  InverseDepth dinv(0.01);

  // Try to make it negative
  Eigen::Matrix<double, 1, 1> negativeUpdate;
  negativeUpdate << -1.0;

  dinv.update(negativeUpdate);

  // Should be clamped to minimum positive value
  EXPECT_GE(dinv.value, std::numeric_limits<double>::min());
  EXPECT_GT(dinv.value, 0);
}

TEST_F(InverseDepthTest, EnsureUpdateIsRevertible) {
  InverseDepth dinv(0.05);

  Eigen::Matrix<double, 1, 1> largeNegative;
  largeNegative << -0.1;

  dinv.ensureUpdateIsRevertible(largeNegative);

  // Update should be clamped
  EXPECT_GE(largeNegative(0), std::numeric_limits<double>::min() - 0.05);

  double before = dinv.value;
  dinv.update(largeNegative);
  dinv.update(-largeNegative);

  // Should be able to revert
  EXPECT_NEAR(dinv.value, before, 1e-10);
}

TEST_F(InverseDepthTest, Dimension) {
  EXPECT_EQ(static_cast<size_t>(InverseDepth::dimension), 1u);
}

// =============================================================================
// SimpleScalar Tests
// =============================================================================

TEST(SimpleScalarTest, DefaultConstructor) {
  SimpleScalar ss;
  SUCCEED();  // Just check it compiles
}

TEST(SimpleScalarTest, ValueConstructor) {
  SimpleScalar ss(42.0);
  EXPECT_EQ(ss.value, 42.0);
}

TEST(SimpleScalarTest, UpdateAndRevert) {
  SimpleScalar ss(10.0);

  Eigen::Matrix<double, 1, 1> dx;
  dx << 5.0;

  ss.update(dx);
  EXPECT_EQ(ss.value, 15.0);

  ss.update(-dx);
  EXPECT_EQ(ss.value, 10.0);
}

TEST(SimpleScalarTest, ZeroUpdate) {
  SimpleScalar ss(100.0);
  double original = ss.value;

  Eigen::Matrix<double, 1, 1> zeroUpdate;
  zeroUpdate << 0.0;

  ss.update(zeroUpdate);
  EXPECT_EQ(ss.value, original);
}

TEST(SimpleScalarTest, LargeValues) {
  SimpleScalar ss(1e10);

  Eigen::Matrix<double, 1, 1> dx;
  dx << 1e10;

  ss.update(dx);
  EXPECT_EQ(ss.value, 2e10);
}

TEST(SimpleScalarTest, NegativeValues) {
  SimpleScalar ss(-100.0);

  Eigen::Matrix<double, 1, 1> dx;
  dx << -50.0;

  ss.update(dx);
  EXPECT_EQ(ss.value, -150.0);
}

TEST(SimpleScalarTest, Dimension) {
  EXPECT_EQ(static_cast<size_t>(SimpleScalar::dimension), 1u);
}
