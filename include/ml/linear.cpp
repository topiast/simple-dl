#pragma once

#include "linear_algebra/number.h"
#include "linear_algebra/function.h"
#include "linear_algebra/tensor.h"

#include <iostream>

using Number = ln::Number<float>;
using Tensor = ln::Tensor<Number>;
using Function = ln::Function<Number>;

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

            // combine weights and bias into one vector 
            std::vector<Number> parameters;
            parameters.reserve(weights.get_values().size() + bias.get_values().size());
            parameters.insert(parameters.end(), weights.get_values().begin(), weights.get_values().end());
            parameters.insert(parameters.end(), bias.get_values().begin(), bias.get_values().end());

            Function loss_func({weights.get_values(), bias.get_values()}, [&output, &target]() {
                return (output - target).pow(2).sum();
            });

            Tensor gradient = loss / input.get_shape()[0];
            weights = weights - input.transpose().matmul(gradient) * learning_rate;
            bias = bias - gradient.sum(0) * learning_rate;
            std::cout << "Epoch: " << i << " Loss: " << loss.sum() << std::endl;
        }
    }
};