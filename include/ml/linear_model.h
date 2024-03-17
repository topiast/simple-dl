#pragma once

#include "math/number.h"
#include "math/function.h"
#include "math/tensor.h"

#include <iostream>

using Number = sdlm::Number<float>;
using Tensor = sdlm::Tensor<Number>;
using Function = sdlm::Function<float>;

namespace sdl {

class Linear {
private:
    Tensor weights;
    Tensor bias;

public:
    Linear(int in_features, int out_features) {
        weights.zeros({in_features, out_features});
        bias.zeros({1, out_features});
    }

    Tensor forward(const Tensor& input) {
        return input.matmul(weights).row_add(bias);
    }

    Tensor get_weights() {
        return weights;
    }

    Tensor get_bias() {
        return bias;
    }

    void fit(const Tensor& input, const Tensor& target, int epochs, float learning_rate, bool print_loss = true) {

        // combine weights and bias into one vector of pointers
        std::vector<Number*> parameters;
        parameters.reserve(weights.get_values().size() + bias.get_values().size());
        for (auto& w : weights.get_values()) {
            parameters.push_back(&w);
        }
        for (auto& b : bias.get_values()) {
            parameters.push_back(&b);
        }

        // // print parameters
        // for (auto& p : parameters) {
        //     std::cout << p << std::endl;
        // }



        Function loss_func(parameters, [this, &input, &target]() {
            // mean squared error
            // (1 / n) * (y - y_hat)^2
            return (forward(input) - target).pow(2).sum() / input.get_shape()[0];
        });
        
        for (int e = 0; e < epochs; e++) {
            Number loss(0);

            std::vector<Number> gradients_weights;
            for (int i = 0; i < weights.get_size(); i++) {

                Number result = loss_func.derivative(parameters[i]);
                gradients_weights.push_back(result.gradient());
                // printf("Gradient for parameter %d: %f\n", i, result.gradient());
                loss += result;
            }
            //for biases
            std::vector<Number> gradients_bias;
            for (int i = weights.get_size(); i < parameters.size(); i++) {
                Number result = loss_func.derivative(parameters[i]);
                gradients_bias.push_back(result.gradient());
            }

            Tensor gradients_weights_tensor(gradients_weights, weights.get_shape());
            Tensor gradients_bias_tensor(gradients_bias, bias.get_shape());

            // clip gradients to prevent exploding gradients
            gradients_weights_tensor = gradients_weights_tensor.clip(-1, 1);
            gradients_bias_tensor = gradients_bias_tensor.clip(-1, 1);


            weights -= gradients_weights_tensor * learning_rate;
            bias -= gradients_bias_tensor * learning_rate;

            if (print_loss) {
                std::cout << "Epoch: " << e << " Loss: " << loss / input.get_shape()[0] << std::endl;
            }
        }
    }
};

}  // namespace sdl