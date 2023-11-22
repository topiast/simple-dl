cmake_minimum_required(VERSION 3.5)
project(linear_algebra_library VERSION 1.0)

# Define C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the library
add_library(linear_algebra INTERFACE)

# Add the include directory
target_include_directories(linear_algebra INTERFACE include)

# Create the example executable
add_executable(example examples/gradients.cpp)

# Link the library to the executable
target_link_libraries(example PRIVATE linear_algebra)