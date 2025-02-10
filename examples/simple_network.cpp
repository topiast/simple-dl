#include "math/tensor.h"

#include "ml/linear.h"
#include "ml/activation_functions.h"
#include "ml/sequential.h"
#include "ml/sdg.h"
#include "ml/loss_functions.h"


#include <iostream>
#include <functional>

using Tensor = sdlm::Tensor<float>;
using Linear = sdl::Linear<float>;
using Sigmoid = sdl::Sigmoid<float>;
using ReLU = sdl::ReLU<float>;
using Sequential = sdl::Sequential<float>;
using SDG = sdl::SDG<float>;
// using Function = sdlm::Function<float>;

// create some linear function
// returns a vector of 3 floats
Tensor some_linear_function(float y) {
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

    std::cout << "X shape: " << X.shape()[0] << "x" << X.shape()[1] << std::endl;



    for(int i = 0; i < X.shape()[0]; i++) {
        X.set_values(i, some_linear_function(i).values());
        Y.set_values(i, {(float)i});
    }

    std::cout << "Data set: " << std::endl;
    X.print();


    // create a simple network
    Linear* linear1 = new Linear(3, 10, Linear::Initializer::Xavier);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(10, 3, Linear::Initializer::Xavier);
    ReLU* act2 = new ReLU();
    Linear* linear3 = new Linear(3, 1, Linear::Initializer::Xavier);
    // Linear* linear3 = new Linear(100, 50, Linear::Initializer::Xavier);
    // ReLU* act3 = new ReLU();
    // Linear* linear4 = new Linear(50, 1, Linear::Initializer::Xavier);

    Sequential simple_network({linear1, act1, linear2, act2, linear3});
    // Sequential simple_network({linear1, act1, linear2, act2, linear3, act3, linear4});
    // // create a smaller network
    // Linear* linear1 = new Linear(3, 1);
    // ReLU* act1 = new ReLU();
    // Linear* linear2 = new Linear(2, 1);


    // Sequential simple_network({linear1});


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


    std::vector<Tensor*> parameters = simple_network.get_parameters();

    for (auto& p : parameters) {
        p->set_requires_gradient(true);
        // p->debug_print();
    }
    std::cout << "Parameters size: " << parameters.size() << std::endl;

    std::function<Tensor()> loss_func = [&simple_network, &X, &Y]() { return sdl::mse(simple_network.forward(X), Y); };

    SDG sdg(parameters, 0.001f, 0.9f);

    // float total_loss = float::max().value();
    // float prev_loss = 0;
    // float threshold = 0.0001;
    // int i = 0;

    // while (std::abs(total_loss - prev_loss) > threshold) {
    //     // std::cout << "forward" << std::endl;
    //     float loss = loss_func();
    //     prev_loss = total_loss;
    //     total_loss = loss.value();

    //     std::cout << "Epoch " << i << " loss: " << loss.value() << std::endl;
    //     loss.backward();
    //     // std::cout << "step" << std::endl;
    //     sdg.step();
    //     // std::cout << "zero grad" << std::endl;
    //     for (auto& p : parameters) {
    //         p->set_gradient(0);
    //     }
    //     i++;
    // }

    sdg.fit_until_convergence(loss_func, 0.0001, true);
    // sdg.fit(1000, true);

    // print Y
    std::cout << "Target: " << std::endl;
    Y.print();

    // output of the model
    std::cout << "Output: " << std::endl;
    simple_network.forward(X).print();

    // // print weights
    // std::cout << "Weights: " << std::endl;
    // linear1->get_weights().print();
    // linear2->get_weights().print();

    // // print bias
    // std::cout << "Bias: " << std::endl;
    // linear1->get_bias().print();
    // linear2->get_bias().print();

    // print loss
    // std::cout << "Loss: " << std::endl;
    // float loss = loss_func();

    // std::cout << loss << std::endl;


    delete linear1;
    delete linear2;
    delete act1;
    delete act2;
    delete linear3;


    return 0;



}
