cmake_minimum_required(VERSION 3.10)
project(MachineLearningToolbox)

# Set C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Specify the include directories
include_directories(include ${EIGEN3_INCLUDE_DIRS})

# Create the static library from the source files
add_library(Mlt STATIC src/SupervisedLearning/Regression/LinearRegression.cpp)

# Set lib output directory
set_target_properties(Mlt PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add the test executable and link
add_executable(MltTest test/testMain.cpp test/SupervisedLearning/Regression/LinearRegressionTest.cpp)
target_link_libraries(MltTest Mlt ${GTEST_LIBRARIES})

# Set test binary output directory
set_target_properties(MltTest PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable Testing
enable_testing()

# Optional Example Builds
option(BUILD_LINEAR_REGRESSION "Build Linear Regression Example" OFF)

# Optional Builds
if (BUILD_LINEAR_REGRESSION)
    add_executable(LinearRegressionExample examples/SupervisedLearning/Regression/LinearRegression.cpp)
    target_link_libraries(LinearRegressionExample Mlt)
endif()