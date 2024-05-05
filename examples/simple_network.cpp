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

using Number = sdlm::Number<float>;
using Tensor = sdlm::Tensor<Number>;
using Linear = sdl::Linear<float>;
using Sigmoid = sdl::Sigmoid<float>;
using ReLU = sdl::ReLU<float>;
using Sequential = sdl::Sequential<float>;
using SDG = sdl::SDG<float>;
// using Function = sdlm::Function<float>;

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
    X.ones({4, 3});  // Example tensor with shape 4x3 filled with ones
    Y.ones({4, 1});  // Example tensor with shape 4x3 filled with ones

    std::cout << "X shape: " << X.get_shape()[0] << "x" << X.get_shape()[1] << std::endl;

    // set the count gradient to false
    X.foreach([](Number& x) { x.set_count_gradient(false); });
    Y.foreach([](Number& y) { y.set_count_gradient(false); });


    for(int i = 0; i < X.get_shape()[0]; i++) {
        X.set_values(i, some_linear_function(i).get_values());
        Y.set_values(i, {i});
    }

    // print X
    std::cout << "Data set: " << std::endl;
    X.print();


    // create a simple network
    Linear* linear1 = new Linear(3, 50);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(50, 50);
    ReLU* act2 = new ReLU();
    Linear* linear3 = new Linear(50, 1);
    
    Sequential simple_network({linear1, act1, linear2, act2, linear3});


    // // print weights
    // std::cout << "Weights: " << std::endl;
    // linear1->get_weights().print();
    // linear2->get_weights().print();

    // // print bias
    // std::cout << "Bias: " << std::endl;
    // linear1->get_bias().print();
    // linear2->get_bias().print();

    // simple_network.print();

    // // output of the model
    // std::cout << "Output: " << std::endl;
    // simple_network.forward(X).print();

    // // ouput of the first layer
    // std::cout << "Output of the first layer: " << std::endl;
    // linear1->forward(X).print();

    // // ouput of the second layer
    // std::cout << "Output of the second layer: " << std::endl;
    // linear2->forward(linear1->forward(X)).print();


    std::vector<Number*> parameters = simple_network.get_parameters();

    for (auto& p : parameters) {
        p->debug_print();
    }
    std::cout << "Parameters size: " << parameters.size() << std::endl;

    std::function<Number()> loss_func = [&simple_network, &X, &Y]() { return sdl::mse(simple_network.forward(X), Y); };

    SDG sdg(parameters, 0.001, 0.5);

    float total_loss = Number::max().value();
    float prev_loss = 0;
    float threshold = 0.0001;
    int i = 0;

    while (std::abs(total_loss - prev_loss) > threshold) {
        // std::cout << "forward" << std::endl;
        Number loss = sdl::mse(simple_network.forward(X), Y);
        prev_loss = total_loss;
        total_loss = loss.value();

        std::cout << "Epoch " << i << " loss: " << loss.value() << std::endl;
        loss.backward();
        // std::cout << "step" << std::endl;
        sdg.step();
        // std::cout << "zero grad" << std::endl;
        for (auto& p : parameters) {
            p->set_gradient(0);
        }
        i++;
    }

    // Number loss = loss_func();
    // std::cout << "-----Initial loss: " << std::endl;
    // loss.debug_print();
    // loss.backward();

    // std::cout << "-----Gradients: " << std::endl;

    // for (auto& p : parameters) {
    //     p->debug_print();
    // }

    // sdg.step();

    // std::cout << "-----After step: " << std::endl;

    // for (auto& p : parameters) {
    //     p->debug_print();
    // }
    // std::cout << "-----Loss after step: " << std::endl;
    // loss = loss_func();
    // loss.debug_print();

    // loss.zero_grad();

    // std::cout << "-----After zero grad: " << std::endl;

    // for (auto& p : parameters) {
    //     p->debug_print();
    // }



    // sdg.fit_until_convergence(loss_func, 0.0001, true);
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
    // std::cout << "Loss: " << std::endl;
    // Number loss = loss_func();

    // std::cout << loss << std::endl;
    

    delete linear1;
    delete linear2;
    delete act1;
    delete act2;

    return 0;



}
