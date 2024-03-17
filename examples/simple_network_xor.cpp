#include "math/number.h"
#include "math/tensor.h"
#include "math/function.h"

#include "ml/linear.h"
#include "ml/activation_functions.h"
#include "ml/sequential.h"
#include "ml/sdg.h"


#include <iostream>

using Number = sdlm::Number<double>;
using Tensor = sdlm::Tensor<Number>;
using Linear = sdl::Linear<double>;
using Sigmoid = sdl::Sigmoid<double>;
using ReLU = sdl::ReLU<double>;
using Sequential = sdl::Sequential<double>;
using SDG = sdl::SDG<double>;
using Function = sdlm::Function<double>;

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
    X.ones({4, 2});  // Example tensor with shape 4x2 filled with ones
    Y.ones({4, 1});  // Example tensor with shape 4x1 filled with ones

    // set values of X and Y, so that it represents the XOR function
    X.set_values(0, {0, 0});
    Y.set_values(0, {0});

    X.set_values(1, {0, 1});
    Y.set_values(1, {1});

    X.set_values(2, {1, 0});
    Y.set_values(2, {1});

    X.set_values(3, {1, 1});
    Y.set_values(3, {0});



    // print X
    std::cout << "Data set: " << std::endl;
    X.print();


    // create a simple network
    Linear* linear1 = new Linear(2, 10);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(10, 1);
    ReLU* act2 = new ReLU();
    
    Sequential simple_network({linear1, act1, linear2});

    // print weights
    std::cout << "Weights: " << std::endl;
    linear1->get_weights().print();
    linear2->get_weights().print();

    // print bias
    std::cout << "Bias: " << std::endl;
    linear1->get_bias().print();
    linear2->get_bias().print();

    simple_network.print();

    // output of the model
    std::cout << "Output: " << std::endl;
    simple_network.forward(X).print();

    // ouput of the first layer
    std::cout << "Output of the first layer: " << std::endl;
    linear1->forward(X).print();

    // ouput of the second layer
    std::cout << "Output of the second layer: " << std::endl;
    linear2->forward(linear1->forward(X)).print();


    std::vector<Number*> parameters = simple_network.get_parameters();

    Function loss_func(parameters, [&simple_network, &X, &Y]() {
        // mean squared error
        return (simple_network.forward(X) - Y).pow(2).sum() / X.get_shape()[0];
    });


    SDG sdg(parameters, loss_func, 0.001, 0.9);


    sdg.fit_until_convergence(0.00001);
    // sdg.fit(1000, true);

    // print Y
    std::cout << "Target: " << std::endl;
    Y.print();

    // output of the model
    std::cout << "Output: " << std::endl;
    simple_network.forward(X).print();

    // print weights
    std::cout << "Weights: " << std::endl;
    linear1->get_weights().print();
    linear2->get_weights().print();

    // print bias
    std::cout << "Bias: " << std::endl;
    linear1->get_bias().print();
    linear2->get_bias().print();

    // print loss
    std::cout << "Loss: " << std::endl;
    Number loss = loss_func.compute();

    std::cout << loss << std::endl;
    

    delete linear1;
    delete linear2;
    delete act1;
    delete act2;

    return 0;



}
