#include "math/tensor.h"

#include "ml/linear.h"
#include "ml/activation_functions.h"
#include "ml/sequential.h"
#include "ml/sdg.h"
#include "ml/loss_functions.h"


#include <iostream>

using Tensor = sdlm::Tensor<float>;
using Linear = sdl::Linear<float>;
using Sigmoid = sdl::Sigmoid<float>;
using ReLU = sdl::ReLU<float>;
using Tanh = sdl::Tanh<float>;
using Sequential = sdl::Sequential<float>;
using SDG = sdl::SDG<float>;



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

    // print Y
    std::cout << "Target: " << std::endl;
    Y.print();

    // create a simple network
    Linear* linear1 = new Linear(2, 15);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(15, 1);
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


    std::vector<Tensor*> parameters = simple_network.get_parameters();

    std::function<Tensor()> loss_func = [&simple_network, &X, &Y]() { return sdl::mse(simple_network.forward(X), Y); };

    SDG sdg(parameters, 0.01f);


    sdg.fit_until_convergence(loss_func, 0.0001, true);

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
    Tensor loss = loss_func();

    std::cout << loss << std::endl;
    

    delete linear1;
    delete linear2;
    delete act1;

    return 0;



}
