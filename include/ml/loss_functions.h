#pragma once

#include <vector>
#include <iostream>

#include "math/number.h"
#include "math/tensor.h"
#include "ml/module.h"

namespace sdl {

// cross entropy loss
// input is the output of the model and each row is a vector of probabilities
template <typename T>
sdlm::Number<T> cross_entropy(sdlm::Tensor<sdlm::Number<T>>& input, sdlm::Tensor<sdlm::Number<T>>& target) {
    return -(target * (input.log())).sum() / input.get_shape()[0];
}
template <typename T>
sdlm::Number<T> cross_entropy(sdlm::Tensor<sdlm::Number<T>>&& input, sdlm::Tensor<sdlm::Number<T>>&& target) {
    return cross_entropy(input, target);
}
template <typename T>
sdlm::Number<T> cross_entropy(sdlm::Tensor<sdlm::Number<T>>& input, sdlm::Tensor<sdlm::Number<T>>&& target) {
    return cross_entropy(input, target);
}
template <typename T>
sdlm::Number<T> cross_entropy(sdlm::Tensor<sdlm::Number<T>>&& input, sdlm::Tensor<sdlm::Number<T>>& target) {
    return cross_entropy(input, target);
}


// mean squared error loss
template <typename T>
sdlm::Number<T> mse(sdlm::Tensor<sdlm::Number<T>>& input, sdlm::Tensor<sdlm::Number<T>>& target) {
    return (input - target).pow(2).sum() / input.get_shape()[0];
}
template <typename T>
sdlm::Number<T> mse(sdlm::Tensor<sdlm::Number<T>>&& input, sdlm::Tensor<sdlm::Number<T>>&& target) {
    return mse(input, target);
}
template <typename T>
sdlm::Number<T> mse(sdlm::Tensor<sdlm::Number<T>>& input, sdlm::Tensor<sdlm::Number<T>>&& target) {
    return mse(input, target);
}
template <typename T>
sdlm::Number<T> mse(sdlm::Tensor<sdlm::Number<T>>&& input, sdlm::Tensor<sdlm::Number<T>>& target) {
    return mse(input, target);
}


} // namespace sdl
