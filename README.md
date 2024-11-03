![image](https://github.com/ParkerBarrett959/MachineLearningToolbox/blob/main/doc/MLTLogo.PNG)
# Machine Learning Toolbox
A C++ library for building, testing and learning machine learning models and algorithms. Each algorithm's individual directory contains examples of how to use it. Detailed algorithm write-ups can be found here: https://www.numericalprogrammingwithparker.com/.

[![AutomatedTests Actions Status](https://github.com/ParkerBarrett959/MachineLearningToolbox/workflows/MachineLearningToolbox-master/badge.svg)](https://github.com/ParkerBarrett959/MachineLearningToolbox/actions)

# Dependencies
* C++ 17 (or greater) <br />
* CMake (3.10.0 or greater) <br />
* Eigen (3.3 or greater) <br />

# Build
```
mkdir build
cd build
cmake ..
make
```
# Run Unit Tests
```
./MltTest
```

# Run Examples

This library contains examples for all the algorithms implemented. The example executables can be built by enabling them in the CMake command by using a flag. For example:

```
mkdir build
cd build
cmake .. -DBUILD_LINEAR_REGRESSION=true
make
./LinearRegressionExample
```

Depending on which flags are used, the associated binaries will be output to the build directory. The current list of supported examples are (entire CMake and run command included below for convenience):

```
# Linear Regression
cmake .. -DBUILD_LINEAR_REGRESSION=true # CMake command
./LinearRegressionExample # Run command
```
