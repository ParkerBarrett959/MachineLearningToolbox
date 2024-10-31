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

// Add training data
TEST(LinearRegression, AddTrainingData) {
  // Create LinearRegression simple Objects
  LinearRegression lr;

  // Verify simple model size
  EXPECT_EQ(lr.getNumberPredictorWeights(), 2);
  EXPECT_EQ(lr.getNumberMeasurements(), 0);

  // Add a bad measurement
  Eigen::Matrix<double, 1, 3> measBad;
  measBad << 1.0, 2.0, 3.0;
  EXPECT_FALSE(lr.addTrainingData(measBad));
  EXPECT_EQ(lr.getNumberMeasurements(), 0);

  // Add a valid measurmeent
  Eigen::Matrix<double, 1, 2> measGood;
  measGood << 1.0, 2.0;
  EXPECT_TRUE(lr.addTrainingData(measGood));
  EXPECT_EQ(lr.getNumberMeasurements(), 1);
}

// Add solve regression and predict
TEST(LinearRegression, SolveAndPredict1) {
  // Create LinearRegression bject
  LinearRegression lr;

  // Verify simple model size
  EXPECT_EQ(lr.getNumberPredictorWeights(), 2);
  EXPECT_EQ(lr.getNumberMeasurements(), 0);

  // Add measurements to model 1
  Eigen::Matrix<double, 2, 2> meas;
  meas << 1.0, 1.0, -1.0, -1.0;
  EXPECT_TRUE(lr.addTrainingData(meas.row(0)));
  EXPECT_FALSE(lr.solveRegression());
  EXPECT_TRUE(lr.addTrainingData(meas));
  EXPECT_FALSE(lr.regressionSolved());
  EXPECT_TRUE(lr.solveRegression());
  EXPECT_TRUE(lr.regressionSolved());
  Eigen::VectorXd weights = lr.getPredictorWeights();
  EXPECT_EQ(weights(0), 0.0);
  EXPECT_EQ(weights(1), 1.0);

  // Make single value predictions
  Eigen::Vector<double, 1> predictor;
  predictor << 2.0;
  auto result = lr.predict(predictor);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 2.0);
  predictor(0) = -2.0;
  result = lr.predict(predictor);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), -2.0);
}

// Add solve regression
TEST(LinearRegression, Solve2) {
  // Create LinearRegression bject
  LinearRegression lr(3);

  // Verify simple model size
  EXPECT_EQ(lr.getNumberPredictorWeights(), 3);
  EXPECT_EQ(lr.getNumberMeasurements(), 0);

  // Add measurements to model
  Eigen::Matrix<double, 6, 3> meas;
  meas << 18.0, 52.0, 144.0, 24.0, 40.0, 142.0, 12.0, 40.0, 124.0, 30.0, 48.0,
      64.0, 30.0, 32.0, 96.0, 22.0, 16.0, 92.0;
  EXPECT_TRUE(lr.addTrainingData(meas));

  // Solve Regression
  EXPECT_FALSE(lr.regressionSolved());
  EXPECT_TRUE(lr.solveRegression());
  EXPECT_TRUE(lr.regressionSolved());
  Eigen::VectorXd weights = lr.getPredictorWeights();
  EXPECT_NEAR(weights(0), 150.166, 1e-3);
  EXPECT_NEAR(weights(1), -2.731, 1e-3);
  EXPECT_NEAR(weights(2), 0.581, 1e-3);

  // Make single prediction value
  Eigen::Vector<double, 2> predictor;
  predictor << 2.0, 20.0;
  auto result = lr.predict(predictor);
  EXPECT_TRUE(result.has_value());
  EXPECT_NEAR(result.value(), 156.323, 1e-3);
  predictor(0) = -2.0;
  predictor(1) = -20.0;
  result = lr.predict(predictor);
  EXPECT_TRUE(result.has_value());
  EXPECT_NEAR(result.value(), 144.009, 1e-3);
}