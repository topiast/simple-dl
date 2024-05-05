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

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>& input) override {
        return input.sigmoid();
    }

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>&& input) override {
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

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>& input) override {
        return input.relu();
    }

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>&& input) override {
        return input.relu();
    }

    std::vector<sdlm::Number<T>*> get_parameters() override {
        return {};
    }

    std::string get_name() override {
        return "ReLU";
    }

};

//softmax
template <typename T>
class Softmax : public Module<T> {
public:

    Softmax() {}

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>& input) override {
        return input.softmax();
    }
    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>&& input) override {
        return input.softmax();
    }

    std::vector<sdlm::Number<T>*> get_parameters() override {
        return {};
    }

    std::string get_name() override {
        return "Softmax";
    }

};

// tanh
template <typename T>
class Tanh : public Module<T> {
public:

    Tanh() {}

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>& input) override {
        return input.tanh();
    }

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>&& input) override {
        return input.tanh();
    }

    std::vector<sdlm::Number<T>*> get_parameters() override {
        return {};
    }

    std::string get_name() override {
        return "Tanh";
    }

};

template <typename T>
class Flatten : public Module<T> {
public:

    Flatten() {}

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>& input) override {
        return input.flatten();
    }

    sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>&& input) override {
        return input.flatten();
    }

    std::vector<sdlm::Number<T>*> get_parameters() override {
        return {};
    }

    std::string get_name() override {
        return "Flatten";
    }

};

} // namespace sdl