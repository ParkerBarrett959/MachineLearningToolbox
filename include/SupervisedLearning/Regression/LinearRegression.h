#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <Eigen/Dense>
#include <iostream>

class LinearRegression {
public:
  LinearRegression();
  void print() const;

private:
  Eigen::MatrixXd matrix_;
  double a_;
};

#endif // LINEAR_REGRESSION_H