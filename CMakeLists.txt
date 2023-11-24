cmake_minimum_required(VERSION 3.5)
project(linear_algebra_library VERSION 1.0)

# Define C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the library
add_library(linear_algebra INTERFACE)

# Add the include directory
target_include_directories(linear_algebra INTERFACE include)

# Find Google Test package (assuming installed via Homebrew)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Create the example executable
add_executable(gradients examples/gradients.cpp)
add_executable(tensor examples/tensor.cpp)
add_executable(linear examples/linear.cpp)

# Link the library to the executable
target_link_libraries(gradients PRIVATE linear_algebra)
target_link_libraries(tensor PRIVATE linear_algebra)
target_link_libraries(linear PRIVATE linear_algebra)

# Add the test executable
add_executable(test_gradients tests/test_gradients.cpp)
add_executable(test_tensors tests/test_tensors.cpp)

# Link the library and Google Test to the test executable
target_link_libraries(test_gradients PRIVATE linear_algebra GTest::GTest GTest::Main)
target_link_libraries(test_tensors PRIVATE linear_algebra GTest::GTest GTest::Main)

# Add the test to CTest
add_test(
    NAME test_gradients
    COMMAND test_gradients
)
add_test(
    NAME test_tensors
    COMMAND test_tensors
)
