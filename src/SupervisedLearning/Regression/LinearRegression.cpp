#include "SupervisedLearning/Regression/LinearRegression.h"

// Default constructor
LinearRegression::LinearRegression()
    : numPredictorWeights_(2), numMeasurements_(0) {}

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