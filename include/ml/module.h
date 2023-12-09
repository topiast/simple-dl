#pragma once

#include <vector>
#include <iostream>
#include <string>

#include "math/number.h"
#include "math/tensor.h"

// using namespace sdlm;

namespace sdl {

template <typename T>
class Module {
public:
    virtual ~Module() {}

    virtual sdlm::Tensor<sdlm::Number<T>> forward(const sdlm::Tensor<sdlm::Number<T>>& input) = 0;
    
    virtual std::vector<sdlm::Number<T>*> get_parameters() = 0;

    virtual std::string get_name() = 0;

    sdlm::Tensor<sdlm::Number<T>> operator()(const sdlm::Tensor<sdlm::Number<T>>& input) {
        return forward(input);
    }

    sdlm::Tensor<sdlm::Number<T>> flatten() {
        auto parameters = get_parameters();
        return sdlm::Tensor<sdlm::Number<T>>(parameters, {parameters.size(), 1});
    }

};
    
} // namespace sdl
