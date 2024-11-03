# Script to Run Clang-Format Automatically

# Headers
clang-format -i include/SupervisedLearning/Regression/*.h

# Source Files
clang-format -i src/SupervisedLearning/Regression/*.cpp

# Test Files
clang-format -i test/*.cpp
clang-format -i test/SupervisedLearning/Regression/*.cpp

# Example Files
clang-format -i examples/SupervisedLearning/Regression/*.cpp
