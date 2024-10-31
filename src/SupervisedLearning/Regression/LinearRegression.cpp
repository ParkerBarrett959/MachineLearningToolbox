#include "SupervisedLearning/Regression/LinearRegression.h"

// Default constructor
LinearRegression::LinearRegression()
    : numPredictorWeights_(2), numMeasurements_(0) {
  predictorWeights_.conservativeResize(numPredictorWeights_);
}

// Constructor for case with n parameters
LinearRegression::LinearRegression(const int nParams)
    : numPredictorWeights_(nParams), numMeasurements_(0) {
  predictorWeights_.conservativeResize(numPredictorWeights_);
}

// Add training data to the linear regression
bool LinearRegression::addTrainingData(const Eigen::MatrixXd &data) {
  // Verify each observation has the correct size
  if (data.cols() != numPredictorWeights_) {
    return false;
  }

  // Resize the input matrix and measurement vector
  numMeasurements_ = data.rows();
  A_.conservativeResize(numMeasurements_, numPredictorWeights_);
  Y_.conservativeResize(numMeasurements_, 1);

  // Fill in the input matrix
  A_.col(0) = Eigen::VectorXd::Ones(A_.rows());
  for (int i = 0; i < data.cols() - 1; i++) {
    A_.col(i + 1) = data.col(i);
  }

  // Fill in the measurment vector
  Y_.col(0) = data.col(data.cols() - 1);
  return true;
}

// Solve Regression
bool LinearRegression::solveRegression() {
  // Check for to few measurments - underdetermined system
  if (numMeasurements_ < numPredictorWeights_) {
    return false;
  }

  // Solve and set the predictor weights
  predictorWeights_ = (A_.transpose() * A_).ldlt().solve(A_.transpose() * Y_);
  regressionSolved_ = true;
  return regressionSolved_;
}