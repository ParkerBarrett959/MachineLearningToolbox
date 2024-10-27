/*
 * LinearRegressionTest.cpp
 * Author: Parker Barrett
 * Overview: Unit tests for LinearRegression class.
 */

// Include Statements
#include "SupervisedLearning/Regression/LinearRegression.h"
#include "gtest/gtest.h"

// Constructor: Default
TEST(LinearRegression, DefaultConstructor) {
  // Create LinearRegression simple Objects
  LinearRegression lr;

  // Verify simple model size
  EXPECT_EQ(lr.getNumberPredictorWeights(), 2);
  EXPECT_EQ(lr.getNumberMeasurements(), 0);
}

// Add measurements
TEST(LinearRegression, AddMeasurment) {
  // Create LinearRegression simple Objects
  LinearRegression lr;

  // Verify simple model size
  EXPECT_EQ(lr.getNumberPredictorWeights(), 2);
  EXPECT_EQ(lr.getNumberMeasurements(), 0);

  // Add a bad measurement
  Eigen::Matrix<double, 1, 2> measBad;
  measBad << 1.0, 2.0;
  double y = 3.0;
  EXPECT_FALSE(lr.addSample(measBad, y));
  EXPECT_EQ(lr.getNumberMeasurements(), 0);

  // Add a valid measurmeent
  Eigen::Matrix<double, 1, 1> measGood;
  measGood << 1.0;
  EXPECT_TRUE(lr.addSample(measGood, y));
  EXPECT_EQ(lr.getNumberMeasurements(), 1);
}