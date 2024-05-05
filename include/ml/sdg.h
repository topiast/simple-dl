#pragma once

#include <vector>
#include <iostream>
#include <functional>

#include "math/number.h"
#include "math/tensor.h"
#include "math/function.h"


namespace sdl {

template <typename T>
class SDG {
private:
    std::vector<sdlm::Number<T>*> parameters;
    T learning_rate;
    T momentum_factor;
    bool clip_gradients;
    std::vector<T> momentums; 

public:
    SDG(std::vector<sdlm::Number<T>*> parameters, T learning_rate, T momentum_factor = 0.9, bool clip_gradients = false)
        : parameters(parameters), learning_rate(learning_rate), momentum_factor(momentum_factor), clip_gradients(clip_gradients), momentums(parameters.size(), 0){
    }    

    void step() {

        // Update momentums and parameters using momentum
        for (int i = 0; i < parameters.size(); i++) {
            momentums[i] = momentums[i] * momentum_factor + parameters[i]->gradient();
            // std::cout << "parameter " << parameters[i] << " gradient: " << gradients[i] << std::endl;
            if (clip_gradients) {
                T clipped_momentum = momentums[i] > 1 ? 1 : (momentums[i] < -1 ? -1 : momentums[i]);
                parameters[i]->no_grad() -= clipped_momentum * learning_rate;
            } else {
                parameters[i]->no_grad() -= momentums[i] * learning_rate;
            }
        }

    }


    // void fit(int epochs, bool print_loss = true) {
    //     for (int i = 0; i < epochs; i++) {
    //         sdlm::Number<T> total_loss = step();
    //         if (print_loss) {
    //             std::cout << "Epoch " << i << " loss: " << total_loss << std::endl;
    //         }
    //     }
    // }

    void fit_until_convergence(const std::function<sdlm::Number<T>()>& loss_function, const T& threshold, bool print_loss = true) {
        if (print_loss) {
            std::cout << "Training until convergence with threshold " << threshold << std::endl;
        }
        T total_loss = sdlm::Number<T>::max().value();
        sdlm::Number<T> loss = loss_function();
        T prev_loss = loss.value();

        // loss.get_grad_fn()->visualize();

        if (print_loss){
            std::cout << "Initial loss: " << loss << std::endl;
        }
        
        int i = 0;
        while (std::abs(total_loss - prev_loss) > threshold) {
            prev_loss = total_loss;
            std::cout << "here" << std::endl;
            // loss.zero_grad();
            std::cout << "here 2" << std::endl;
            loss.backward();
            std::cout << "here 3" << std::endl;
            step();
            
            //zero gradients
            for (auto& p : parameters) {
                p->set_gradient(0);
            }

            std::cout << "here 4" << std::endl;
            loss = loss_function();
            std::cout << "here 5" << std::endl;
            total_loss = loss.value();

            if (print_loss) {
                std::cout << "Epoch " << i << " loss: " << total_loss << std::endl;
            }
            i++;
        }
    }

};

} // namespace sdl