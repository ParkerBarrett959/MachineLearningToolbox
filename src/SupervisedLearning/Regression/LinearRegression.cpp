#include "SupervisedLearning/Regression/LinearRegression.h"

LinearRegression::LinearRegression() { matrix_.setZero(); }

void LinearRegression::print() const { std::cout << "a = " << a_ << std::endl; }