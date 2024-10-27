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
   * Get system dimension
   */
  int getSystemDimension() const { return systemDim_; }

  /**
   * Get number of predictor weights
   */
  int getNumberPredictorWeights() const { return numberPredictorWeights_; }

private:
  // Dimension of the system
  int systemDim_;

  // Number of predictor weights, bo, b1, ..., bn
  int numberPredictorWeights_;
};

#endif // LINEAR_REGRESSION_H