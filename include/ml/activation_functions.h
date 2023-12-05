#pragma once

#include <vector>
#include <iostream>

#include "math/number.h"
#include "ml/module.h"

namespace sdl {

template <typename T>
class Sigmoid : public Module<T> {
public:

    Sigmoid() {}

    sdlm::Tensor<sdlm::Number<T>> forward(const sdlm::Tensor<sdlm::Number<T>>& input) override {
        return input.sigmoid();
    }

    std::vector<sdlm::Number<T>*> get_parameters() override {
        return {};
    }

    std::string get_name() override {
        return "Sigmoid";
    }

};

template <typename T>
class ReLU : public Module<T> {
public:

    ReLU() {}

    sdlm::Tensor<sdlm::Number<T>> forward(const sdlm::Tensor<sdlm::Number<T>>& input) override {
        return input.relu();
    }

    std::vector<sdlm::Number<T>*> get_parameters() override {
        return {};
    }

    std::string get_name() override {
        return "ReLU";
    }

};

} // namespace sdl