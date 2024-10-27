cmake_minimum_required(VERSION 3.10)
project(MachineLearningToolbox)

# Set C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Specify the include directories
include_directories(include ${EIGEN3_INCLUDE_DIRS})

# Create the static library from the source files
add_library(MyLibrary STATIC src/class.cpp)

# Optionally, you can specify the output directory for the library
set_target_properties(MyLibrary PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
