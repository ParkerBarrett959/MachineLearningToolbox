/*
 * LinearRegressionTest.cpp
 * Author: Parker Barrett
 * Overview: Unit tests for LinearRegression class.
 */

// Include Statements
#include "SupervisedLearning/Regression/LinearRegression.h"
#include "gtest/gtest.h"

// Constructor: Default
TEST(LinearRegressionConstructor, Default) {
  // Create LinearRegression 1D Objects
  LinearRegression lr;

  // Verify 1D model size
  EXPECT_EQ(lr.getSystemDimension(), 1);
  EXPECT_EQ(lr.getNumberPredictorWeights(), 2);
}