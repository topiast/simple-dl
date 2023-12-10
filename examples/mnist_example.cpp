#include "math/number.h"
#include "math/tensor.h"
#include "math/function.h"

#include "ml/linear.h"
#include "ml/activation_functions.h"
#include "ml/loss_functions.h"
#include "ml/sequential.h"
#include "ml/sdg.h"
#include "ml/utils.h"


#include <iostream>

using Number = sdlm::Number<float>;
using Tensor = sdlm::Tensor<Number>;
using Linear = sdl::Linear<float>;
using Sigmoid = sdl::Sigmoid<float>;
using Softmax = sdl::Softmax<float>;
using ReLU = sdl::ReLU<float>;
using Flatten = sdl::Flatten<float>;
using Sequential = sdl::Sequential<float>;
using SDG = sdl::SDG<float>;
using Function = sdlm::Function<float>;

// take the path to the mnist dataset as command line argument
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path to mnist dataset>" << std::endl;
        exit(1);
    }

    std::string path = argv[1];
    std::string train_images = path + "/train-images-idx3-ubyte";
    std::string train_labels = path + "/train-labels-idx1-ubyte";

    Tensor X, Y;

    sdl::utils::load_mnist_data(train_images, train_labels, X, Y);

    std::cout << "X shape: " << std::endl;
    X.print_shape();
    std::cout << "Y shape: " << std::endl;
    Y.print_shape();

    // convert Y to one-hot encoding
    Y = Y.one_hot(10, [](const Number& x) { return static_cast<int>(x.value()); });

    // to visualize the data, we write the first 10 images to file
    for (int i = 0; i < 10; i++) {
        Tensor X_0 = X.get_values({i});
        // std::cout << "X_0 shape: " << std::endl;
        // X_0.print_shape();
        // X_0.print();

        // std::cout << "X.get_values({" << i << "})" << std::endl;

        Number y_0 = Y.get_values()[i];
        // std::cout << "y_0: " << y_0 << std::endl;

        // write X_0 to file
        std::string filename = "X_0_" + std::to_string((int)(y_0.value())) + ".tga";

        sdl::utils::write_tga_image(filename, X_0);
    }

    // use only 1 image for training
    X = X.get_values({0});
    Y = Y.get_values({0});
    std::cout << "X shape: " << std::endl;

    // create a simple network

    Flatten* flatten = new Flatten();
    Linear* linear1 = new Linear(784, 128);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(128, 10);
    Softmax* act2 = new Softmax();

    Sequential simple_network({flatten, linear1, act1, linear2, act2});
    
    // get parameters
    std::vector<Number*> parameters = simple_network.get_parameters();

    Function loss_func(parameters, [&simple_network, &X, &Y]() {
        Tensor output = simple_network.forward(X);
        Number loss = sdl::cross_entropy(output, Y);
        return loss;
    });

    // create optimizer
    SDG sdg(parameters, loss_func, 0.001, 0.9);

    // train the model
    std::cout << "Training..." << std::endl;

    sdg.fit_until_convergence(0.0001);


    return 0;
}