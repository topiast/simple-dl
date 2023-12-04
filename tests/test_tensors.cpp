#include <gtest/gtest.h>
#include "math/tensor.h"

// Test case for tensor product operation
TEST(TensorTest, TensorProduct) {
    sdlm::Tensor<int> tensor1, tensor2, result;

    // Create two tensors
    tensor1.ones({2, 3});  // Example tensor with shape 2x3 filled with ones
    tensor2.ones({3, 2});  // Example tensor with shape 3x2 filled with ones

    tensor1.set_values(0,{1, 2, 3});
    tensor1.set_values(1,{4, 5, 6});

    tensor2.set_values(0,{1, 2});
    tensor2.set_values(1,{3, 4});
    tensor2.set_values(2,{5, 6});

    // Perform tensor product
    result = tensor1.matmul(tensor2);

    // Check if the resulting shape is as expected
    std::vector<int> expectedShape = {2, 2};
    ASSERT_EQ(result.get_shape(), expectedShape);

    // Check if the resulting values are as expected
//   [[22 28]
//    [49 64]]
    std::vector<int> expectedValues = {22, 28, 49, 64};
    ASSERT_EQ(result.get_values(), expectedValues);
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
    ASSERT_EQ(result.get_shape(), expectedShape);

    // Check if the resulting values are as expected
//   [[1 4]
//    [2 5]
//    [3 6]]
    std::vector<int> expectedValues = {1, 4, 2, 5, 3, 6};
    ASSERT_EQ(result.get_values(), expectedValues);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}