#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <Eigen/Dense>
#include <iostream>
#include <optional>

/**
 * Linear Regression.
 *
 * @brief: This class is for solving linear regression models of the form:
 * Y = b0 + b1*X1 + b2*X2 + ... + bn*Xn. Note that each term in the summation is
 * linear, meaning this class supports both simple and multiple linear
 * regressions.
 */
class LinearRegression {
public:
  /**
   * Default c'tor
   * Note: The default is a simple linear regression of the form: Y = b0 +
   * b1*X
   */
  LinearRegression();

  /**
   * c'tor for a regression of n paramaters
   */
  LinearRegression(const int nParams_);

  /**
   * Get number of predictor weights
   */
  int getNumberPredictorWeights() const { return numPredictorWeights_; }

  /**
   * Get number of measurements in system
   */
  int getNumberMeasurements() const { return numMeasurements_; }

  /**
   * Check if the predictor weights have been calculated
   */
  bool regressionSolved() const { return regressionSolved_; }

  /**
   * Get predictor weights. WARNING - user should check if the weights have been
   * set with the regressionSolved() function before using
   */
  Eigen::VectorXd getPredictorWeights() const { return predictorWeights_; }

  /**
   * Add training data to the regression problem.
   *
   * @param data: The input of the independent variables (predictors) as a 2D
   * Eigen matrix, where each row corresponds to a set of predictors and
   * observation, [x1, x2, ..., xn, y]. Each row should have the same number of
   * elements as the number of predictor weights. The last element in the row is
   * the observation value.
   *
   * @return True if the observation was added to the system, false otherwise.
   */
  bool addTrainingData(const Eigen::MatrixXd &data);

  /**
   * Solve regression
   *
   * @return True if the regression solved successfully, false otherwise.
   */
  bool solveRegression();

  /**
   * Make a single prediction given the input values
   *
   * @param predictors The vector of predictors. There should be
   * numPredictorWeights_-1 predictors.
   * @return std::optional prediction value. If the prediction failed, the
   * optional will remain unset.
   */
  std::optional<double> predict(const Eigen::VectorXd &predictors);

private:
  // Number of predictor weights, bo, b1, ..., bn
  const int numPredictorWeights_;

  // Number of measurements in the system
  int numMeasurements_;

  // Input matrix
  Eigen::MatrixXd A_;

  // Observation matrix holding the measurements
  Eigen::VectorXd Y_;

  // Regression solved flag - prediction will only run if the regression has
  // been solved
  bool regressionSolved_ = false;

  // Predictor Weights
  Eigen::VectorXd predictorWeights_;
};

#endif // LINEAR_REGRESSION_H