#include "math/number.h"
#include "math/tensor.h"
// #include "math/function.h"

#include "ml/linear.h"
#include "ml/activation_functions.h"
#include "ml/sequential.h"
#include "ml/sdg.h"
#include "ml/loss_functions.h"


#include <iostream>
#include <functional>

using Number = sdlm::Number<double>;
using Tensor = sdlm::Tensor<Number>;
using Linear = sdl::Linear<double>;
using Sigmoid = sdl::Sigmoid<double>;
using ReLU = sdl::ReLU<double>;
using Sequential = sdl::Sequential<double>;
using SDG = sdl::SDG<double>;
using Softmax = sdl::Softmax<double>;
// using Function = sdlm::Function<double>;

// create some linear function
// returns a vector of 3 numbers
Tensor some_linear_function(Number y) {
    Tensor result;
    result.ones({3});

    result.set_values(0, {y * 2});
    result.set_values(1, {y/3});
    result.set_values(2, {y * 7});


    return result;
    
}


int main() {
Tensor input({1, 3});
    input.set_values(0, {1, 2, 3});
    // create tensor with 1,0,0 values
    Tensor target({1, 3});
    target.set_values(0, {1, 0, 0});

    std::vector<Number*> variables;

    for (auto& v : input.get_values()) {
        variables.push_back(&v);
        v.set_count_gradient(true);
    }
    // input.print();

    // Number& x = input[0];
    // (-( (std::exp(input[0]) / (std::exp(input[0]) + std::exp(input[1] + std::exp(input[2])))).log() )).backward();
    // x.debug_print();
    // std::cout << "softmax: " << std::endl;
    // Number sum_exp = (std::exp(input[0]) + std::exp(input[1] + std::exp(input[2])));
    // ???

    // Number f = -(target[0] * ( (std::exp(input[0]) / sum_exp).log() ) + target[1] * ( (std::exp(input[1]) / sum_exp).log() ) + target[2] * ( (std::exp(input[2]) / sum_exp).log() ));
    // auto softmax = input.softmax();
    // Number f = sdl::cross_entropy(softmax, target);
    // f.backward();
    // f.debug_print();

    // variables[0]->debug_print();
    // variables[1]->debug_print();
    // variables[2]->debug_print();

    // softmax[0].debug_print();
    // softmax[1].debug_print();
    // softmax[2].debug_print();

    // NOTE: Tensor operations break the gradient tracking since the values are copied and the pointers are different


    //create mock data tensor
    Tensor X, Y;
    Y.ones({3, 3});  // Example tensor with shape 4x3 filled with ones
    X.ones({3, 3});  // Example tensor with shape 4x3 filled with ones

    std::cout << "X shape: " << X.get_shape()[0] << "x" << X.get_shape()[1] << std::endl;



    for(int i = 0; i < X.get_shape()[0]; i++) {
        X.set_values(i, some_linear_function(i).get_values());
        // Set the appropriate one-hot vector for each sample
        if (i == 0) {
            Y.set_values(i, {1, 0, 0});
        } else if (i == 1) {
            Y.set_values(i, {0, 1, 0});
        } else if (i == 2) {
            Y.set_values(i, {0, 0, 1});
        } else {
            Y.set_values(i, {0, 0, 0});
        }
    }


    std::cout << "Data set: " << std::endl;
    X.print();


    // create a simple network
    Linear* linear1 = new Linear(3, 50);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(50, 50);
    ReLU* act2 = new ReLU();
    Linear* linear3 = new Linear(50, 3);  // Update the output shape to match the number of classes
    Softmax* act3 = new Softmax();

    
    Sequential simple_network({linear1, act1, linear2, act2, linear3, act3});
    
    std::vector<Number*> parameters = simple_network.get_parameters();

    for (auto& p : parameters) {
        p->set_count_gradient(true);
    }
    std::cout << "Parameters size: " << parameters.size() << std::endl;

    // std::function<Number()> loss_func = [&simple_network, &X, &Y]() { return sdl::mse(simple_network.forward(X), Y); };
    std::function<Number()> loss_func = [&simple_network, &X, &Y]() { return sdl::cross_entropy(simple_network.forward(X), Y); };

    SDG sdg(parameters, 0.015);

    sdg.fit_until_convergence(loss_func, 0.0001, true);

    // print Y
    std::cout << "Target: " << std::endl;
    Y.print();

    // output of the model
    std::cout << "Output: " << std::endl;
    simple_network.forward(X).print();

    // print the parameters
    linear1->get_bias().print();
    linear3->get_bias().print();

    delete linear1;
    delete linear2;
    delete act1;
    delete act2;

    return 0;
}
