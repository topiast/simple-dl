#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <limits>

#include "math/tensor.h"



namespace sdl {

template <typename T>
class SDG {
private:
    std::vector<sdlm::Tensor<T>*> parameters;
    T learning_rate;
    T momentum_factor;
    bool clip_gradients;
    std::vector<T> momentums; 

public:
    SDG(std::vector<sdlm::Tensor<T>*> parameters, T learning_rate, T momentum_factor = 0.9, bool clip_gradients = false)
        : parameters(parameters), learning_rate(learning_rate), momentum_factor(momentum_factor), clip_gradients(clip_gradients) {
            // initialize momentums
            int size = 0;
            for (auto& p : parameters) {
                size += p->size();
            }
            momentums = std::vector<T>(size, 0);
            // set all parameters to count gradients
            for (auto& p : parameters) {
                p->set_requires_gradient(true);
            }
    }    

    void step() {

        // Update momentums and parameters using momentum
        for (int i = 0; i < parameters.size(); i++) {
            auto& p = parameters[i];
            auto grads = p->gradients();
            for (int j = 0; j < p->size(); j++) {
                momentums[i] = momentums[i] * momentum_factor + grads[j];
                if (clip_gradients) {
                    T clipped_momentum = momentums[i] > 1 ? 1 : (momentums[i] < -1 ? -1 : momentums[i]);
                    (*p)[j] -= clipped_momentum * learning_rate;
                } else {
                    (*p)[j] -= momentums[i] * learning_rate;
                }
            }
        }







        //     momentums[i] = momentums[i] * momentum_factor + parameters[i]->gradient();
        //     // std::cout << "parameter " << parameters[i] << " gradient: " << gradients[i] << std::endl;
        //     if (clip_gradients) {
        //         T clipped_momentum = momentums[i] > 1 ? 1 : (momentums[i] < -1 ? -1 : momentums[i]);
        //         for (int j = 0; j < parameters[i]->gradients().size(); j++) {
                    
        //         parameters[i]->gradients() -= clipped_momentum * learning_rate;
        //     } else { 
        //         parameters[i]->gradients() -= momentums[i] * learning_rate;
        //     }
        // }

    }


    // void fit(int epochs, bool print_loss = true) {
    //     for (int i = 0; i < epochs; i++) {
    //         sdlm::Number<T> total_loss = step();
    //         if (print_loss) {
    //             std::cout << "Epoch " << i << " loss: " << total_loss << std::endl;
    //         }
    //     }
    // }

    void fit_until_convergence(const std::function<sdlm::Tensor<T>()>& loss_function, const T& threshold, bool print_loss = true) {
        if (print_loss) {
            std::cout << "Training until convergence with threshold " << threshold << std::endl;
        }
        T total_loss = std::numeric_limits<T>::max();
        sdlm::Tensor<T> loss = loss_function();
        T prev_loss = loss.value();

        if (print_loss){
            std::cout << "Initial loss: " << loss << std::endl;
        }
        
        int i = 0;
        while (std::abs(total_loss - prev_loss) > threshold) {
            prev_loss = total_loss;
            loss.backward();
            step();
            
            //zero gradients
            for (auto& p : parameters) {
                p->zero_grad();
            }

            loss = loss_function();
            total_loss = loss.value();

            if (print_loss) {
                std::cout << "Epoch " << i << " loss: " << total_loss << std::endl;
            }
            i++;
        }
    }

};

} // namespace sdl