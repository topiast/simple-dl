#pragma once

#include <vector>
#include <iostream>

#include "math/tensor.h"
#include "ml/module.h"

namespace sdl {

// cross entropy loss
// input is the output of the model and each row is a vector of probabilities
template <typename T>
sdlm::Tensor<T> cross_entropy(sdlm::Tensor<T>& input, sdlm::Tensor<T>& target) {
    return -(target * (input.log())).sum();// / input.shape()[0];
}
template <typename T>
sdlm::Tensor<T> cross_entropy(sdlm::Tensor<T>&& input, sdlm::Tensor<T>&& target) {
    return cross_entropy(input, target);
}
template <typename T>
sdlm::Tensor<T> cross_entropy(sdlm::Tensor<T>& input, sdlm::Tensor<T>&& target) {
    return cross_entropy(input, target);
}
template <typename T>
sdlm::Tensor<T> cross_entropy(sdlm::Tensor<T>&& input, sdlm::Tensor<T>& target) {
    return cross_entropy(input, target);
}

// accuracy
template <typename T>
sdlm::Tensor<T> accuracy(sdlm::Tensor<T>& input, sdlm::Tensor<T>& target) {
    sdlm::Tensor<T> max_vals = input.argmax(1);
    sdlm::Tensor<T> target_vals = target.argmax(1);
    return (max_vals == target_vals).sum() / input.shape()[0];
}




// mean squared error loss
template <typename T>
sdlm::Tensor<T> mse(sdlm::Tensor<T>& input, sdlm::Tensor<T>& target) {
    sdlm::Tensor<T> diff = input - target;
    return (diff^2).sum() / input.shape()[0];
}
template <typename T>
sdlm::Tensor<T> mse(sdlm::Tensor<T>&& input, sdlm::Tensor<T>&& target) {
    return mse(input, target);
}
template <typename T>
sdlm::Tensor<T> mse(sdlm::Tensor<T>& input, sdlm::Tensor<T>&& target) {
    return mse(input, target);
}
template <typename T>
sdlm::Tensor<T> mse(sdlm::Tensor<T>&& input, sdlm::Tensor<T>& target) {
    return mse(input, target);
}


} // namespace sdl
