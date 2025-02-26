#pragma once

#include <vector>
#include <iostream>
#include <string>

#include "math/tensor.h"

// using namespace sdlm;

namespace sdl {

/**
 * @brief The base class for all modules in the Simple Deep Learning library.
 * 
 * This class provides an interface for defining and using modules in the Simple Deep Learning library.
 * Modules are the building blocks of neural networks and encapsulate operations that can be applied to input data.
 * 
 * @tparam T The data type of the module's parameters and input data.
 */
template <typename T>
class Module {
public:
    /**
     * @brief Destructor for the Module class.
     */
    virtual ~Module() {}

    /**
     * @brief Performs a forward pass through the module.
     * 
     * This function takes an input tensor and applies the module's operations to produce an output tensor.
     * 
     * @param input The input tensor to the module.
     * @return The output tensor produced by the module.
     */
    virtual sdlm::Tensor<T> forward(sdlm::Tensor<T>& input) = 0;

    virtual sdlm::Tensor<T> forward(sdlm::Tensor<T>&& input) = 0;

    
    /**
     * @brief Retrieves the parameters of the module.
     * 
     * This function returns a vector of pointers to the parameters of the module.
     * Parameters are variables that are learned during the training process.
     * 
     * @return A vector of pointers to the parameters of the module.
     */
    virtual std::vector<sdlm::Tensor<T>*> get_parameters() = 0;

    /**
     * @brief Retrieves the name of the module.
     * 
     * This function returns the name of the module as a string.
     * 
     * @return The name of the module.
     */
    virtual std::string get_name() = 0;

    /**
     * @brief Operator overload for calling the module as a function.
     * 
     * This operator allows the module to be called as a function, which is equivalent to calling the forward() function.
     * 
     * @param input The input tensor to the module.
     * @return The output tensor produced by the module.
     */
    sdlm::Tensor<T> operator()(sdlm::Tensor<T>& input) {
        return forward(input);
    }

    /**
     * @brief Flattens the parameters of the module into a 2D tensor.
     * 
     * This function returns a 2D tensor that contains the parameters of the module.
     * The parameters are flattened into a single column vector.
     * 
     * @return A 2D tensor containing the flattened parameters of the module.
     */
    // sdlm::Tensor<T> flatten() { // TODO: figure out if this is needed
    //     auto parameters = get_parameters();
    //     return sdlm::Tensor<T>(parameters, {parameters.size(), 1});
    // }

    /**
     * @brief Set gradient to zero for all parameters of the module.
     * 
     * This function sets the gradient of all parameters of the module to zero.
     */
    void zero_grad() {
        auto parameters = get_parameters();
        for (auto& p : parameters) {
            p->zero_grad();
        }
    }
};
    
} // namespace sdl
