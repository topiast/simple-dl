#pragma once

#include "linear_algebra/number.h"
#include "linear_algebra/function.h"
#include "linear_algebra/tensor.h"

#include <iostream>

using Number = ln::Number<float>;
using Tensor = ln::Tensor<Number>;
using Function = ln::Function<float>;

class Linear {
private:
    Tensor weights;
    Tensor bias;

public:
    Linear(int in_features, int out_features) {
        weights.ones({in_features, out_features});
        bias.ones({out_features});
    }

    Tensor forward(const Tensor& input) {
        return input.matmul(weights) + bias;
    }

    Tensor get_weights() {
        return weights;
    }

    Tensor get_bias() {
        return bias;
    }

    void fit(const Tensor& input, const Tensor& target, int epochs, float learning_rate) {
        for (int i = 0; i < epochs; i++) {
            Tensor output = forward(input);
            Tensor loss = output - target;

            // combine weights and bias into one vector of pointers
            std::vector<Number*> parameters;
            parameters.reserve(weights.get_values().size() + bias.get_values().size());
            for (auto& w : weights.get_values()) {
                parameters.push_back(&w);
            }
            for (auto& b : bias.get_values()) {
                parameters.push_back(&b);
            }
            

            Function loss_func(parameters, [&output, &target]() {
                return (output - target).pow(2).sum();
            });

            std::vector<Number> gradients_weights;
            for (int i = 0; i < weights.get_size(); i++) {
                Number result = loss_func.derivative(parameters[i]);
                gradients_weights.push_back(result.gradient());
            }
            //for biases
            std::vector<Number> gradients_bias;
            for (int i = bias.get_size(); i < parameters.size(); i++) {
                Number result = loss_func.derivative(parameters[i]);
                gradients_bias.push_back(result.gradient());
            }
            Tensor gradients_weights_tensor(gradients_weights, weights.get_shape());
            Tensor gradients_bias_tensor(gradients_bias, bias.get_shape());

            weights = weights - gradients_weights_tensor * learning_rate;
            bias = bias - gradients_bias_tensor * learning_rate;

            std::cout << "Epoch: " << i << " Loss: " << loss.sum() / loss.get_size() << std::endl;
        }
    }
};