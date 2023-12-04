#pragma once

#include <vector>
#include <iostream>

#include "math/number.h"
#include "math/tensor.h"
#include "math/function.h"


namespace sdl {

template <typename T>
class SDG {
private:
    std::vector<sdlm::Number<T>*> parameters;
    sdlm::Function<T> loss_func;
    T learning_rate;
    bool clip_gradients = true;

public:

    SDG(std::vector<sdlm::Number<T>*> parameters, sdlm::Function<T> loss_func, T learning_rate) : parameters(parameters), loss_func(loss_func), learning_rate(learning_rate) {}

    SDG(std::vector<sdlm::Number<T>*> parameters, sdlm::Function<T> loss_func, T learning_rate, bool clip_gradients) : parameters(parameters), loss_func(loss_func), learning_rate(learning_rate), clip_gradients(clip_gradients) {}
    
    sdlm::Number<T> step() {
        sdlm::Number<T> total_loss = loss_func.compute();
        // Calculate gradients
        std::vector<sdlm::Number<T>> gradients;
        gradients.reserve(parameters.size());
        for (auto& p : parameters) {
            gradients.push_back(loss_func.derivative(p).gradient());
        }


        // Update parameters
        for (int i = 0; i < parameters.size(); i++) {
            if (clip_gradients) {
                *parameters[i] -= gradients[i].clip(-1, 1) * learning_rate;
            } else {
                *parameters[i] -= gradients[i] * learning_rate;
            }
        }

        return total_loss;

    }

    void fit(int epochs, bool print_loss = true) {
        for (int i = 0; i < epochs; i++) {
            sdlm::Number<T> total_loss = step();
            if (print_loss) {
                std::cout << "Epoch " << i << " loss: " << total_loss << std::endl;
            }
        }
    }

    void fit_until_convergence(const sdlm::Number<T>& threshold, bool print_loss = true) {
        sdlm::Number<T> total_loss = sdlm::Number<T>::max();
        sdlm::Number<T> prev_loss = step();

        int i = 0;
        while (std::abs(total_loss - prev_loss) > threshold) {
            prev_loss = total_loss;
            total_loss = step();
            if (print_loss) {
                std::cout << "Epoch " << i << " loss: " << total_loss << std::endl;
            }
            i++;
        }
    }

};

} // namespace sdl