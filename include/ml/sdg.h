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
    T momentum_factor;
    bool clip_gradients;
    std::vector<sdlm::Number<T>> momentums; 

public:
    SDG(std::vector<sdlm::Number<T>*> parameters, sdlm::Function<T> loss_func, T learning_rate, T momentum_factor = 0.9, bool clip_gradients = false)
        : parameters(parameters), loss_func(loss_func), learning_rate(learning_rate), momentum_factor(momentum_factor) {
        momentums.reserve(parameters.size());
        for (auto& p : parameters) {
            momentums.push_back(sdlm::Number<T>(0)); 
        }
    }    

    // TODO: implenment adaptive learning rate
    sdlm::Number<T> step() {
        sdlm::Number<T> total_loss = loss_func.compute();
        // Calculate gradients
        std::vector<sdlm::Number<T>> gradients;
        gradients.reserve(parameters.size());
        for (auto& p : parameters) {
            gradients.push_back(loss_func.derivative(p).gradient());
        }


        // Update momentums and parameters using momentum
        for (int i = 0; i < parameters.size(); i++) {
            momentums[i] = momentums[i] * momentum_factor + gradients[i];
            // std::cout << "parameter " << parameters[i] << " gradient: " << gradients[i] << std::endl;
            if (clip_gradients) {
                *parameters[i] -= momentums[i].clip(-1, 1) * learning_rate;
            } else {
                *parameters[i] -= momentums[i] * learning_rate;
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