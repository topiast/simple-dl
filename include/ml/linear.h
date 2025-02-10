#pragma once

#include <vector>
#include <iostream>

#include "math/tensor.h"
#include "ml/module.h"

namespace sdl {



template <typename T>
class Linear : public Module<T> {
private:
    sdlm::Tensor<T> weights;
    sdlm::Tensor<T> bias;

public:

    enum class Initializer {
        Zeros,
        Xavier,
        He
    };

    Linear(int in_features, int out_features, Initializer initializer = Initializer::Xavier) {
        bias.zeros({out_features});
        // bias = bias.transpose();

        if (initializer == Initializer::Zeros) {
            weights.zeros({in_features, out_features});
        } else if (initializer == Initializer::Xavier) {
            T std = std::sqrt(2.0f / (in_features + out_features));
            weights.normal({in_features, out_features}, 0.f, std);

        } else if (initializer == Initializer::He) {
            T std = std::sqrt(2.0f / in_features);
            weights.normal({in_features, out_features}, 0.f, std);
        }
    }


    sdlm::Tensor<T> forward(sdlm::Tensor<T>& input) override {
        return input.matmul(weights) + bias;
    }

    sdlm::Tensor<T> forward(sdlm::Tensor<T>&& input) override {
        return input.matmul(weights) + bias;
    }

    sdlm::Tensor<T> get_weights() {
        return weights;
    }

    sdlm::Tensor<T> get_bias() {
        return bias;
    }

    std::vector<sdlm::Tensor<T>*> get_parameters() override {
        // combine weights and bias into one vector of pointers
        std::vector<sdlm::Tensor<T>*> parameters;
        parameters.push_back(&weights);
        parameters.push_back(&bias);

        return parameters;
    }

    std::string get_name() override {
        return "Linear(" + std::to_string(weights.shape()[0]) + ", " + std::to_string(weights.shape()[1]) + ")";
    }


};
    
} // namespace sdl