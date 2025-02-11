#include "math/tensor.h"

#include "ml/linear.h"
#include "ml/activation_functions.h"
#include "ml/loss_functions.h"
#include "ml/sequential.h"
#include "ml/sdg.h"
#include "ml/utils.h"


#include <iostream>
#include <omp.h>
#include <chrono>


using Tensor = sdlm::Tensor<double>;
using Linear = sdl::Linear<double>;
using Sigmoid = sdl::Sigmoid<double>;
using Softmax = sdl::Softmax<double>;
using ReLU = sdl::ReLU<double>;
using Flatten = sdl::Flatten<double>;
using Sequential = sdl::Sequential<double>;
using SDG = sdl::SDG<double>;

// take the path to the mnist dataset as command line argument
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path to mnist dataset>" << std::endl;
        exit(1);
    }

    // get available cpu cores from openmp
    int num_cores = omp_get_max_threads();

    std::cout << "Number of cores: " << num_cores << std::endl;

    std::string path = argv[1];
    std::string train_images = path + "/train-images-idx3-ubyte";
    std::string train_labels = path + "/train-labels-idx1-ubyte";

    Tensor X, Y;

    sdl::utils::load_mnist_data(train_images, train_labels, X, Y);

    std::cout << "X shape: " << std::endl;
    for (int i = 0; i < X.shape().size(); i++) {
        std::cout << X.shape()[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Y shape: " << std::endl;
    for (int i = 0; i < Y.shape().size(); i++) {
        std::cout << Y.shape()[i] << " ";
    }
    std::cout << std::endl;
    // convert Y to one-hot encoding
    std::cout << "Converting Y to one-hot encoding..." << std::endl;
    Y = Y.one_hot(10);


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
    std::cout << "Normalizing X..." << std::endl;
    X = X.reshape({60000, 28, 28}).normalize(0, 1);
    Y = Y.reshape({60000, 10});
    std::cout << "Shuffling data..." << std::endl;
    std::vector<int> indices = X.shuffle_indices();
    X = X.select_indices(indices, 0);
    Y = Y.select_indices(indices, 0);
    std::cout << "Splitting data into training and test set..." << std::endl;
    Tensor X_train = X.head(50000);
    Tensor y_train = Y.head(50000);

    Tensor X_test = X.tail(10000);
    Tensor y_test = Y.tail(10000);

    // // use only 1 image for training
    // Tensor X_train = X.head(100).reshape({100, 28, 28}).normalize(0, 1);
    // Tensor y_train = Y.head(100).reshape({100, 10});

    // Tensor X_test = X.tail(100).reshape({100, 28, 28}).normalize(0, 1);
    // Tensor y_test = Y.tail(100).reshape({100, 10});

    // create a simple network

    Flatten* flatten = new Flatten();
    Linear* linear1 = new Linear(784, 128);
    ReLU* act1 = new ReLU();
    Linear* linear2 = new Linear(128, 64);
    ReLU* act2 = new ReLU();
    Linear* linear3 = new Linear(64, 10);
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
    std::vector<Tensor*> parameters = simple_network.get_parameters();

    std::cout << "Number of parameters: " << parameters.size() << std::endl;

    // create loss function
    // std::function<Tensor()> loss_func = [&simple_network, &X_train, &y_train]() { return sdl::cross_entropy(simple_network.forward(X_train), y_train); };

    // create optimizer
    SDG sdg(parameters, 0.001, 0.9);

    int epochs = 20;
    int batch_size = 50;
    int number_of_batches = X_train.number_of_batches(batch_size);
    Tensor loss;
    double avg_loss = 0.0;
    std::cout << "Number of batches: " << number_of_batches << std::endl;
    std::cout << "Training for " << epochs << " epochs..." << std::endl;
    // train the model

    for (int i = 0; i < epochs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < number_of_batches; j++) {
            Tensor batch = X_train.batch(32, j);
            Tensor target = y_train.batch(32, j);
            loss = sdl::cross_entropy(simple_network.forward(batch), target);
            avg_loss += loss.value();
            loss.backward();
            sdg.step();

            // zero gradients
            simple_network.zero_grad();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        avg_loss /= number_of_batches;
        std::cout << "Epoch: " << i << " Loss: " << avg_loss << " Time: " << elapsed.count() << " seconds" << std::endl;
        avg_loss = 0.0;
        // shuffle the data
        indices = X_train.shuffle_indices();
        X_train = X_train.select_indices(indices, 0);
        y_train = y_train.select_indices(indices, 0);
    }
    std::cout << "Training done." << std::endl;
    // train the model
    // std::cout << "Training..." << std::endl;

    // sdg.fit_until_convergence(loss_func, 0.001, true);

    // std::cout << "Training prediction: " << std::endl;
    // // print the output of the model
    // std::cout << "Output: " << std::endl;
    // simple_network.forward(X_train.head(20)).print();

    // // print the target
    // std::cout << "Target: " << std::endl;
    // y_train.head(20).print();

    // // loss after training
    // std::cout << "Loss after training: " << std::endl;
    // std::cout << loss_func() << std::endl;

    Tensor y_pred_train = simple_network.forward(X_train);

    // loss after training
    std::cout << "Loss for training data: " << std::endl;
    std::cout << sdl::cross_entropy(y_pred_train, y_train) << std::endl;

    // accuracy
    std::cout << "Accuracy: " << std::endl;
    std::cout << sdl::accuracy(y_pred_train, y_train) << std::endl;

    std::cout << "Testing..." << std::endl;

    // test the model
    Tensor y_pred = simple_network.forward(X_test);

    // // print the output of the model
    // std::cout << "Output: " << std::endl;
    // y_pred.print();

    // // print the target
    // std::cout << "Target: " << std::endl;
    // y_test.head(20).print();

    // loss after testing
    std::cout << "Loss for test data: " << std::endl;
    std::cout << sdl::cross_entropy(y_pred, y_test) << std::endl;

    // accuracy
    std::cout << "Accuracy: " << std::endl;
    std::cout << sdl::accuracy(y_pred, y_test) << std::endl;

    return 0;
}