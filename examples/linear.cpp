#include "math/number.h"
#include "math/tensor.h"
#include "ml/linear.h"

#include <iostream>

using Number = sdlm::Number<float>;
using Tensor = sdlm::Tensor<Number>;
using Linear = sdl::Linear;

// create some linear function
// returns a vector of 3 numbers
Tensor some_linear_function(Number y) {
    y.set_count_gradient(false);

    Tensor result;
    result.ones({3});

    result.set_values(0, {y * 2});
    result.set_values(1, {y/3});
    result.set_values(2, {y * 7});


    return result;
    
}


int main() {
    //create mock data tensor
    Tensor X, Y;
    X.ones({10, 3});  // Example tensor with shape 4x3 filled with ones
    Y.ones({10, 1});  // Example tensor with shape 4x3 filled with ones

    for(int i = 0; i < X.get_shape()[0]; i++) {
        X.set_values(i, some_linear_function(i).get_values());
        Y.set_values(i, {i});
    }

    // print X
    std::cout << "Data set: " << std::endl;
    X.print();

    // print Y
    std::cout << "Target: " << std::endl;
    Y.print();

    Linear linear(3, 1);

    // print weights
    std::cout << "Weights: " << std::endl;
    linear.get_weights().print();

    // print bias
    std::cout << "Bias: " << std::endl;
    linear.get_bias().print();

    // print output
    std::cout << "Output: " << std::endl;
    linear.forward(X).print();

    // fit the model
    linear.fit(X, Y, 1000, 0.01);

    // print weights
    std::cout << "Weights: " << std::endl;
    linear.get_weights().print();

    // print bias
    std::cout << "Bias: " << std::endl;
    linear.get_bias().print();

    // print output

    std::cout << "Output: " << std::endl;
    linear.forward(X).print();

    return 0;



}
