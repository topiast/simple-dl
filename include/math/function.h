#pragma once 
#include "math/number.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <functional>

namespace sdlm {

template <typename T>
class Function {
private:
    std::vector<Number<T>*> variables; // Changed to pointers

    std::function<Number<T>()> func;

public:
    Function(std::vector<Number<T>*>& vars, const std::function<Number<T>()>& f) : variables(vars), func(f) {} // Changed parameter type

    Number<T> compute() const {
        return func();
    }

    Number<T> derivative(const Number<T>* variable) {
        for (auto& var : variables) {
            if (var == variable) {
                var->set_gradient(1); // Dereference the pointer to access the object
                // std::cout << "Variable " << var << " gradient set to 1" << std::endl;
            } else {
                var->set_gradient(0); // Dereference the pointer to access the object
            }
        }

        Number<T> result = func();
        return result;
    }

    void set_variables(const std::vector<Number<T>*>& vars) { // Changed parameter type
        variables = vars;
    }

    void print_pointers() {
        for (auto& var : variables) {
            std::cout << var << std::endl;
        }
    }
};

} // namespace sdlm