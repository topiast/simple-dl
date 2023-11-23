#pragma once 
#include "linear_algebra/number.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <functional>

namespace ln {

template <typename T>
class Function {
private:
    std::vector<Number<T>>& variables;
    std::function<Number<T>()> func;

public:
    Function(std::vector<Number<T>>& vars, const std::function<Number<T>()>& f) : variables(vars), func(f) {}

    Number<T> compute() const {
        return func();
    }

    Number<T> derivative(const Number<T>& variable) {
        for (auto& var : variables) {
            if (&var == &variable) {
                var.set_gradient(1);
            } else {
                var.set_gradient(0);
            }
        }

        Number<T> result = func();
        return result;
    }

    void set_variables(const std::vector<Number<T>>& vars) {
        variables = vars;
    }
};

} // namespace ln