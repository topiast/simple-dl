cmake_minimum_required(VERSION 3.5)
project(math_library VERSION 1.0)

# Define C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the library
add_library(math INTERFACE)

# Add the include directory
target_include_directories(math INTERFACE include)

# Find Google Test package (assuming installed via Homebrew)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Create the example executable
add_executable(gradients examples/gradients.cpp)
add_executable(tensor examples/tensor.cpp)
# add_executable(linear_model examples/linear_model.cpp)
add_executable(simple_network examples/simple_network.cpp)
add_executable(mnist_example examples/mnist_example.cpp)
add_executable(simple_network_xor examples/simple_network_xor.cpp)
add_executable(simple_network_class examples/simple_network_class.cpp)

# Link the library to the executable
target_link_libraries(gradients PRIVATE math)
target_link_libraries(tensor PRIVATE math)
# target_link_libraries(linear_model PRIVATE math)
target_link_libraries(simple_network PRIVATE math)
target_link_libraries(mnist_example PRIVATE math)
target_link_libraries(simple_network_xor PRIVATE math)
target_link_libraries(simple_network_class PRIVATE math)

# ---TESTS---
# # Add the test executable
add_executable(test_gradients tests/test_gradients.cpp)
add_executable(test_tensors tests/test_tensors.cpp)
# add_executable(test_linear_model tests/test_linear_model.cpp)

# Link the library and Google Test to the test executable
target_link_libraries(test_gradients PRIVATE math GTest::GTest GTest::Main)
target_link_libraries(test_tensors PRIVATE math GTest::GTest GTest::Main)
# target_link_libraries(test_linear_model PRIVATE math GTest::GTest GTest::Main)

# --- OpenMP Setup ---
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    # Apply OpenMP flags to all executables that require it
    set_target_properties(gradients tensor simple_network mnist_example simple_network_xor simple_network_class
        PROPERTIES COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_LIBRARIES}")
    
    # If you have test executables, apply OpenMP to them as well
    set_target_properties(test_gradients test_tensors
        PROPERTIES COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_LIBRARIES}")
endif()

# Add the test to CTest
add_test(
    NAME test_gradients
    COMMAND test_gradients
)
add_test(
    NAME test_tensors
    COMMAND test_tensors
)
# add_test(
#     NAME test_linear_model
#     COMMAND test_linear_model
# )
# set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "" FORCE)
# set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
