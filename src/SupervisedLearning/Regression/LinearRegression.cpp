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

// Add a sample to the linear regression
bool LinearRegression::addSample(const Eigen::VectorXd &xObs,
                                 const double yObs) {
  // Verify the independent variable vector is the correct size
  if (xObs.size() + 1 != numPredictorWeights_) {
    return false;
  }

  // Add another row to the input matrix
  numMeasurements_ += 1;
  A_.conservativeResize(numMeasurements_, numPredictorWeights_);

  // Fill in the new row of the input matrix
  A_(numMeasurements_ - 1, 0) = 1.0;
  for (int i = 0; i < xObs.size(); i++) {
    A_(numMeasurements_ - 1, i + 1) = xObs(i);
  }

  // Add the measurement to the measurement vector
  Y_.conservativeResize(numMeasurements_, 1);
  Y_(numMeasurements_ - 1) = yObs;
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