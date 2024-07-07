#include "math/number.h"
#include "math/tensor.h"

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
    // for (int i = 0; i < 10; i++) {
    //     Tensor X_0 = X.get_values({i});
    //     // std::cout << "X_0 shape: " << std::endl;
    //     // X_0.print_shape();
    //     // X_0.print();

    //     // std::cout << "X.get_values({" << i << "})" << std::endl;

    //     Number y_0 = Y.get_values()[i];
    //     // std::cout << "y_0: " << y_0 << std::endl;

    //     // write X_0 to file
    //     std::string filename = "X_0_" + std::to_string((int)(y_0.value())) + ".tga";

    //     sdl::utils::write_tga_image(filename, X_0);
    // }

    // use only 1 image for training
    X = X.head(10).reshape({10, 28, 28}).normalize(0, 1);
    Y = Y.head(10).reshape({10, 10});

    // create a simple network

    Flatten* flatten = new Flatten();
    Linear* linear1 = new Linear(784, 32);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(32, 16);
    ReLU* act2 = new ReLU();
    Linear* linear3 = new Linear(16, 10);
    Softmax* output = new Softmax();

    Sequential simple_network({flatten, linear1, act1, linear2, act2, linear3, output});

    simple_network.print();

    // // print data   
    // std::cout << "Data set: " << std::endl;
    // X.print();

    // // forward each layer and print the output shape
    // std::cout << "Forward flatten layer: " << std::endl;
    // Tensor out = flatten->forward(X);
    // out.print();

    // std::cout << "Forward linear1 layer: " << std::endl;
    // out = linear1->forward(out);
    // out.print();

    // std::cout << "Forward act1 layer: " << std::endl;
    // out = act1->forward(out);
    // out.print();

    // std::cout << "Forward linear2 layer: " << std::endl;
    // out = linear2->forward(out);
    // out.print();

    // std::cout << "Forward act2 layer: " << std::endl;
    // out = act2->forward(out);
    // out.print();

    // std::cout << "Forward output layer: " << std::endl;
    // out = output->forward(out);
    // out.print();

    // get parameters
    std::vector<Number*> parameters = simple_network.get_parameters();

    std::cout << "Number of parameters: " << parameters.size() << std::endl;

    // create loss function
    std::function<Number()> loss_func = [&simple_network, &X, &Y]() { return sdl::cross_entropy(simple_network.forward(X), Y); };

    // create optimizer
    SDG sdg(parameters, 0.01, 0.9);

    // train the model
    std::cout << "Training..." << std::endl;

    sdg.fit_until_convergence(loss_func, 0.0001, true);

    // print the output of the model
    std::cout << "Output: " << std::endl;
    simple_network.forward(X).print();

    // print the target
    std::cout << "Target: " << std::endl;
    Y.print();

    // loss after training
    std::cout << "Loss after training: " << std::endl;
    std::cout << loss_func() << std::endl;


    return 0;
}