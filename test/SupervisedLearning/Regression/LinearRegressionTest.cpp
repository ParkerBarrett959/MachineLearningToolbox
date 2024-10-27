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
  // Create LinearRegression Objects
  LinearRegression lr;
  EXPECT_EQ(1, 1);
}