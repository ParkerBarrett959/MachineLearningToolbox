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

// Constructor: n params
TEST(LinearRegression, ParamNumConstructor) {
  // Create LinearRegression simple Objects
  LinearRegression lr(3);

  // Verify simple model size
  EXPECT_EQ(lr.getNumberPredictorWeights(), 3);
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

// Add solve regression
TEST(LinearRegression, Solve1) {
  // Create LinearRegression bject
  LinearRegression lr;

  // Verify simple model size
  EXPECT_EQ(lr.getNumberPredictorWeights(), 2);
  EXPECT_EQ(lr.getNumberMeasurements(), 0);

  // Add measurements to model 1
  double y = 1.0;
  Eigen::Matrix<double, 1, 1> meas;
  meas << 1.0;
  EXPECT_TRUE(lr.addSample(meas, y));
  EXPECT_FALSE(lr.solveRegression());
  meas(0) = -1.0;
  y = -1.0;
  EXPECT_TRUE(lr.addSample(meas, y));
  EXPECT_FALSE(lr.regressionSolved());
  EXPECT_TRUE(lr.solveRegression());
  EXPECT_TRUE(lr.regressionSolved());
  Eigen::VectorXd weights = lr.getPredictorWeights();
  EXPECT_EQ(weights(0), 0.0);
  EXPECT_EQ(weights(1), 1.0);
}

// Add solve regression
TEST(LinearRegression, Solve2) {
  // Create LinearRegression bject
  LinearRegression lr(3);

  // Verify simple model size
  EXPECT_EQ(lr.getNumberPredictorWeights(), 3);
  EXPECT_EQ(lr.getNumberMeasurements(), 0);

  // Add measurements to model
  double y = 144.0;
  Eigen::Matrix<double, 1, 2> meas;
  meas << 18.0, 52.0;
  EXPECT_TRUE(lr.addSample(meas, y));
  y = 142.0;
  meas << 24.0, 40.0;
  EXPECT_TRUE(lr.addSample(meas, y));
  y = 124.0;
  meas << 12.0, 40.0;
  EXPECT_TRUE(lr.addSample(meas, y));
  y = 64.0;
  meas << 30.0, 48.0;
  EXPECT_TRUE(lr.addSample(meas, y));
  y = 96.0;
  meas << 30.0, 32.0;
  EXPECT_TRUE(lr.addSample(meas, y));
  y = 92.0;
  meas << 22.0, 16.0;
  EXPECT_TRUE(lr.addSample(meas, y));

  // Solve Regression
  EXPECT_FALSE(lr.regressionSolved());
  EXPECT_TRUE(lr.solveRegression());
  EXPECT_TRUE(lr.regressionSolved());
  Eigen::VectorXd weights = lr.getPredictorWeights();
  EXPECT_NEAR(weights(0), 150.166, 1e-3);
  EXPECT_NEAR(weights(1), -2.731, 1e-3);
  EXPECT_NEAR(weights(2), 0.581, 1e-3);
}