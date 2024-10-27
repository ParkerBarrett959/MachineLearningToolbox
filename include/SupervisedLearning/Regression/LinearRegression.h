#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <Eigen/Dense>
#include <iostream>

class LinearRegression {
public:
  /**
   * Default c'tor
   * Note: The default is a simple 1D linear regression of the form: Y = b0 +
   * b1*X
   */
  LinearRegression();

  /**
   * Get number of predictor weights
   */
  int getNumberPredictorWeights() const { return numPredictorWeights_; }

  /**
   * Get number of measurements in system
   */
  int getNumberMeasurements() const { return numMeasurements_; }

  /**
   * Add a sample to the regression problem. Note: This fucntion resizes the
   * underlying measurement matrix on each call, consider adding large
   * quantities of measurements in a batch operation instead for performance.
   *
   * @param xObs: The input of the independent variables (predictors) as a
   * vector, [x1, x2, ..., xn]. This vector should have one less element than
   * the number of predictor weights. The leading "1" in the solve is
   * automatically added
   * @param yObs The output of the observation, given a scalar value.
   *
   * @return True if the observation was added to the system, false otherwise.
   */
  bool addSample(const Eigen::VectorXd &xObs, const double yObs);

private:
  // Number of predictor weights, bo, b1, ..., bn
  int numPredictorWeights_;

  // Number of measurements in the system
  int numMeasurements_;

  // Input matrix
  Eigen::MatrixXd A_;

  // Observation matrix holding the measurements
  Eigen::VectorXd Y_;
};

#endif // LINEAR_REGRESSION_H