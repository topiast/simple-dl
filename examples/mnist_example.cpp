#include "math/number.h"
#include "math/tensor.h"
#include "math/function.h"

#include "ml/linear.h"
#include "ml/activation_functions.h"
#include "ml/sequential.h"
#include "ml/sdg.h"
#include "ml/utils.h"


#include <iostream>

using Number = sdlm::Number<float>;
using Tensor = sdlm::Tensor<Number>;
using Linear = sdl::Linear<float>;
using Sigmoid = sdl::Sigmoid<float>;
using ReLU = sdl::ReLU<float>;
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

    // to visualize the data, we write the first 10 images to file
    for (int i = 0; i < 10; i++) {
        Tensor X_0 = X.get_values({i});
        std::cout << "X_0 shape: " << std::endl;
        X_0.print_shape();
        // X_0.print();

        std::cout << "X.get_values({" << i << "})" << std::endl;

        Number y_0 = Y.get_values()[i];
        std::cout << "y_0: " << y_0 << std::endl;

        // write X_0 to file
        std::string filename = "X_0_" + std::to_string((int)(y_0.value())) + ".tga";

        sdl::utils::write_tga_image(filename, X_0);
    }

    return 0;
}