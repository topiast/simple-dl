#include <gtest/gtest.h>
#include "math/tensor.h"

// Test case for matmul operation
TEST(TensorTest, MatMul) {
    sdlm::Tensor<int> tensor1, tensor2, result;

    // Create two tensors
    tensor1.ones({2, 3});  // Example tensor with shape 2x3 filled with ones
    tensor2.ones({3, 2});  // Example tensor with shape 3x2 filled with ones

    tensor1.set_values(0,{1, 2, 3});
    tensor1.set_values(1,{4, 5, 6});

    tensor2.set_values(0,{1, 2});
    tensor2.set_values(1,{3, 4});
    tensor2.set_values(2,{5, 6});

    // Perform matmul
    result = tensor1.matmul(tensor2);

    // Check if the resulting shape is as expected
    std::vector<int> expectedShape = {2, 2};
    ASSERT_EQ(result.shape(), expectedShape);

    // Check if the resulting values are as expected
//   [[22 28]
//    [49 64]]
    std::vector<int> expectedValues = {22, 28, 49, 64};
    ASSERT_EQ(result.values(), expectedValues);
}

// Test case for tensor transpose operation
TEST(TensorTest, TensorTranspose) {
    sdlm::Tensor<int> tensor1, result;

    // Create a tensor
    tensor1.ones({2, 3});  // Example tensor with shape 2x3 filled with ones

    tensor1.set_values(0,{1, 2, 3});
    tensor1.set_values(1,{4, 5, 6});

    // Perform tensor transpose
    result = tensor1.transpose();

    // Check if the resulting shape is as expected
    std::vector<int> expectedShape = {3, 2};
    ASSERT_EQ(result.shape(), expectedShape);

    // Check if the resulting values are as expected
//   [[1 4]
//    [2 5]
//    [3 6]]
    std::vector<int> expectedValues = {1, 4, 2, 5, 3, 6};
    ASSERT_EQ(result.values(), expectedValues);
}

// Test case for tensor addition operation
TEST(TensorTest, TensorAddition) {
    sdlm::Tensor<int> tensor1, tensor2, result;

    // Create two tensors
    tensor1.ones({2, 3});  // Example tensor with shape 2x3 filled with ones
    tensor2.ones({2, 3});  // Example tensor with shape 2x3 filled with ones

    tensor1.set_values(0,{1, 2, 3});
    tensor1.set_values(1,{4, 5, 6});

    tensor2.set_values(0,{1, 2, 3});
    tensor2.set_values(1,{4, 5, 6});

    // Perform tensor addition
    result = tensor1 + tensor2;

    // Check if the resulting shape is as expected
    std::vector<int> expectedShape = {2, 3};
    ASSERT_EQ(result.shape(), expectedShape);

    // Check if the resulting values are as expected
//   [[2 4 6]
//    [8 10 12]]
    std::vector<int> expectedValues = {2, 4, 6, 8, 10, 12};
    ASSERT_EQ(result.values(), expectedValues);
}

// Test case for addition with broadcasting
TEST(TensorTest, TensorAdditionBroadcasting) {
    sdlm::Tensor<int> tensor1, tensor2, result;

    // Create two tensors
    tensor1.ones({2, 3});  // Example tensor with shape 2x3 filled with ones
    tensor2.ones({3});  // Example tensor with shape 3 filled with ones

    tensor1.set_values(0,{1, 2, 3});
    tensor1.set_values(1,{4, 5, 6});

    tensor2.set_values(0,{1, 2, 3});

    // Perform tensor addition with broadcasting
    result = tensor1 + tensor2;

    // Check if the resulting shape is as expected
    std::vector<int> expectedShape = {2, 3};
    ASSERT_EQ(result.shape(), expectedShape);

    // Check if the resulting values are as expected
//   [[2 4 6]
//    [5 7 9]]
    std::vector<int> expectedValues = {2, 4, 6, 5, 7, 9};
    ASSERT_EQ(result.values(), expectedValues);
}

// Test case for reduce sum operation along axis
TEST(TensorTest, ReduceSum) {
    sdlm::Tensor<int> tensor1, result;

    // Create a tensor
    tensor1.ones({2, 3, 4});  // Example tensor with shape 2x3x4 filled with ones

    tensor1.set_values({0, 0}, {1, 2, 3, 4});
    tensor1.set_values({0, 1}, {5, 6, 7, 8});
    tensor1.set_values({0, 2}, {9, 10, 11, 12});

    tensor1.set_values({1, 0}, {13, 14, 15, 16});
    tensor1.set_values({1, 1}, {17, 18, 19, 20});
    tensor1.set_values({1, 2}, {21, 22, 23, 24});

    // Perform reduce sum along axis
    result = tensor1.reduce_sum(1);

    // Check if the resulting shape is as expected
    std::vector<int> expectedShape = {2, 4};
    ASSERT_EQ(result.shape(), expectedShape);

    // Check if the resulting values are as expected
//   [[15 18 21 24]
//    [51 54 57 60]]
    std::vector<int> expectedValues = {15, 18, 21, 24, 51, 54, 57, 60};
    ASSERT_EQ(result.values(), expectedValues);

    // Perform reduce sum along axis
    result = result.reduce_sum(0);

    // Check if the resulting shape is as expected
    expectedShape = {4};
    ASSERT_EQ(result.shape(), expectedShape);

    // Check if the resulting values are as expected
//   [66 72 78 84]
    expectedValues = {66, 72, 78, 84};
    ASSERT_EQ(result.values(), expectedValues);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}