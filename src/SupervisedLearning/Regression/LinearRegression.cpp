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

// Make single prediction
std::optional<double>
LinearRegression::predict(const Eigen::VectorXd &predictors) {
  // Check for correct predictors size
  if (predictors.size() != numPredictorWeights_ - 1) {
    return {};
  }

  // Check that the regression has been solved
  if (!regressionSolved()) {
    return {};
  }

  // Update the predictor vector to have a leading 1
  Eigen::VectorXd predictorsNew(predictors.size() + 1);
  predictorsNew(0) = 1.0;
  predictorsNew.segment(1, predictors.size()) = predictors;

  // Compute prediction
  return predictorWeights_.dot(predictorsNew);
}

// Make batch of predictions
std::optional<Eigen::VectorXd>
LinearRegression::predictBatch(const Eigen::MatrixXd &predictors) {
  // Check for correct predictors size
  if (predictors.cols() != numPredictorWeights_ - 1) {
    return {};
  }

  // Check that the regression has been solved
  if (!regressionSolved()) {
    return {};
  }

  // Loop over each row and compute a prediction
  Eigen::VectorXd predictions(predictors.rows());
  for (int i = 0; i < predictors.rows(); i++) {
    Eigen::VectorXd rowCurr = predictors.row(i);
    auto maybePrediction = predict(rowCurr);
    if (!maybePrediction.has_value()) {
      return {};
    } else {
      predictions(i) = maybePrediction.value();
    }
  }
  return predictions;
}