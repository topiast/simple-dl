#include "math/number.h"
#include "math/tensor.h"
#include "math/function.h"

#include "ml/linear.h"
#include "ml/activation_functions.h"
#include "ml/sequential.h"
#include "ml/sdg.h"


#include <iostream>

using Number = sdlm::Number<float>;
using Tensor = sdlm::Tensor<Number>;
using Linear = sdl::Linear<float>;
using Sigmoid = sdl::Sigmoid<float>;
using ReLU = sdl::ReLU<float>;
using Sequential = sdl::Sequential<float>;
using SDG = sdl::SDG<float>;
using Function = sdlm::Function<float>;

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


    // create a simple network
    Linear* linear1 = new Linear(3, 5);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(5, 1);
    Sigmoid* act2 = new Sigmoid();
    
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


    SDG sdg(parameters, loss_func, 0.001, 0.5);


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
