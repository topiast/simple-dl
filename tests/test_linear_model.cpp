#include "math/number.h"
#include "math/tensor.h"
#include "ml/linear_model.h"

#include <gtest/gtest.h>

using Number = sdlm::Number<float>;
using Tensor = sdlm::Tensor<Number>;
using Linear = sdl::Linear;

// create some linear function
Tensor some_linear_function(Number y, int x_size) {
    y.set_count_gradient(false);

    Tensor result;
    result.ones({x_size});

    for (int i = 0; i < x_size; i++) {
        result.set_values(i, {y * (i + 1)});
    }

    return result;
   
}


TEST(LinearTest, TestLinearFit) {
    Tensor X, Y;
    X.ones({10, 3});  // Example tensor with shape 4x3 filled with ones
    Y.ones({10, 1});  // Example tensor with shape 4x3 filled with ones

    for(int i = 0; i < X.get_shape()[0]; i++) {
        X.set_values(i, some_linear_function(i, 3).get_values());
        Y.set_values(i, {i});
    }

    Linear linear(3, 1);

    linear.fit(X, Y, 1000, 0.01, false);

    Tensor output = linear.forward(X);

    // check that loss is below 0.1
    Number loss = (output - Y).pow(2).sum() / X.get_shape()[0];
    EXPECT_LT(loss.value(), 0.1);

}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
