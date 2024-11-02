# Regression

Regression is a widely used Machine Learning technique for analyzing and modeling the relationships between a dependent variable and one or more independent variables. By establishing a mathematical equation that best fits the observed data, regression helps identify trends, make predictions, and understand the strength and nature of relationships. There are various types of regression, including linear, polynomial, and logistic regression, each suited for different types of data and relationships.

# Linear Regression

Linear regression attempts to find linear relationships between the dependent and independent variables. This form of regression is typically categorized as either simple linear regression, or multiple linear regression.

In simple linear regression, the focus is on one independent variable to predict the outcome. The model takes the form: $y = b_{0} + b_{1}x_{1}$.

In multiple linear regression, multiple independent variables are considered simultaneously, allowing for a more comprehensive analysis of how different factors influence the dependent variable. The model takes the form: $y = b_{0} + b_{1}x_{1} + b_{2}x_{2} + ... + b_{n}x_{n}$.

The linear regression module in the Machine Learning Toolbox supports both simple and multiple regressions. The code block below deomonstrates how you can run a linear regression:

```
// Create a simple linear regression object using the default constructor
LinearRegression simpleLinearRegression;

// Create a multiple linear regression object with 3 predictor weights (2 independent variables)
LinearRegression multipleLinearRegression(3);

/**
 * Initialize training data
 * Training data is loaded as an Eigen::MatrixXd with the number of columns equal to the number
 * of training weights and number of rows equal to the number of observations the model is
 * trained on. Each column except the last corresponds to the independent variables of the
 * current observation, and the final column is the dependent variable. For example, a single
 * row might take the form: [x1, x2, ..., xn, y]. You are required to have AT LEAST as many rows
 * (training data samples) as number of model weights to solve the system. This means your model
 * must have more rows than columns.
 */
Eigen::Matrix<double, 5, 2> simpleTrainingData = Eigen::Matrix<double, 5, 2>::Zero();
Eigen::Matrix<double, 5, 3> multipleTrainingData = Eigen::Matrix<double, 5, 3>::Zero();
// Set each row with the observations for the matrices...

// Add data to the models - assume the training data has been set
bool simpleTrainingDataSet = simpleLinearRegression.addTrainingData(simpleTrainingData);
bool multipleTrainingDataSet = multipleLinearRegression.addTrainingData(multipleTrainingData);

// Solve the regression
bool simpleRegressionSolved = simpleLinearRegression.solveRegression();
bool multipleRegressionSolved =multipleLinearRegression.solveRegression();

// Make a single prediction - the predictors should be stored in an Eigen::VectorXd
Eigen::VectorXd simplePredictor;
Eigen::VectorXd multiplePredictor;
// Set values here...
auto maybeSimpleResult = simpleLinearRegression.predict(simplePredictor);
auto maybeMultipleResult = multipleLinearRegression.predict(multiplePredictor);

// The result will be stored in a std::optional<double>
if (maybeSimpleResult.has_value() && maybeMultipleResult.has_value()) {
    double simpleResult = maybeSimpleResult.value();
    double multipleesult = maybeMultipleResult.value();
}

// Make a batch of predictions - the predictors are stored in an Eigen::MatrixXd where each
// row corresponds to a single prediction
Eigen::MatrixXd simplePredictors;
Eigen::MatrixXd multiplePredictors;
// Set values here...
auto maybeSimpleResults = simpleLinearRegression.predictBatch(simplePredictors);
auto maybeMultipleResults = multipleLinearRegression.predictBatch(multiplePredictors);

// The result will be stored in a std::optional<Eigen::VectorXd>, one entry for each predictor
if (maybeSimpleResults.has_value() && maybeMultipleResults.has_value()) {
    Eigen::VectorXd simpleResults = maybeSimpleResults.value();
    Eigen::VectorXd multipleesults = maybeMultipleResults.value();
}
```

# Polynomial Regression

TODO

# Logistic Regression

TODO
