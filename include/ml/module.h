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

};
    
} // namespace sdl
