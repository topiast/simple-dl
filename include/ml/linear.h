#pragma once

#include <vector>
#include <iostream>

#include "math/number.h"
#include "ml/module.h"

namespace sdl {



template <typename T>
class Linear : public Module<T> {
private:
    sdlm::Tensor<sdlm::Number<T>> weights;
    sdlm::Tensor<sdlm::Number<T>> bias;

public:

    enum class Initializer {
        Zeros,
        Xavier,
        He
    };

    Linear(int in_features, int out_features, Initializer initializer = Initializer::Xavier) {
        bias.zeros({out_features});

        if (initializer == Initializer::Zeros) {
            weights.zeros({in_features, out_features});
        } else if (initializer == Initializer::Xavier) {
            sdlm::Number<T> std = sqrt(2.0 / (in_features + out_features));
            weights.random({in_features, out_features}, 0.f, std.value());

        } else if (initializer == Initializer::He) {
            sdlm::Number<T> std = sqrt(2.0 / in_features);
            weights.random({in_features, out_features}, 0.f, std.value());
        }
    }


    sdlm::Tensor<sdlm::Number<T>> forward(const sdlm::Tensor<sdlm::Number<T>>& input) override {
        return input.matmul(weights) + bias;
    }

    sdlm::Tensor<sdlm::Number<T>> get_weights() {
        return weights;
    }

    sdlm::Tensor<sdlm::Number<T>> get_bias() {
        return bias;
    }

    std::vector<sdlm::Number<T>*> get_parameters() override {
        // combine weights and bias into one vector of pointers
        std::vector<sdlm::Number<T>*> parameters;
        parameters.reserve(weights.get_values().size() + bias.get_values().size());
        for (auto& w : weights.get_values()) {
            parameters.push_back(&w);
        }
        for (auto& b : bias.get_values()) {
            parameters.push_back(&b);
        }

        return parameters;
    }

    std::string get_name() override {
        return "Linear";
    }


};
    
} // namespace sdl