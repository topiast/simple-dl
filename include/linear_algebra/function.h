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
    std::vector<Number<T>> m_variables;

    Number<T> m_value;

    public:
    Function() = default;

    Function(const std::vector<Number<T>>& variables) : m_variables(variables) {}

    Number<T> value() const {
        return m_value;
    }

    void set_value(const Number<T>& value) {
        m_value = value;
    }

    void zero_gradient() {
        for (auto& variable : m_variables) {
            variable.set_gradient(0);
        }
    }


    Number<T> forward(Number<T>& variable, std::function<Number<T>()>& function) {
        for (auto& variable2 : m_variables) {
            if (variable.get_id() == variable2.get_id()) {
                variable2.set_gradient(1);
            } else {
                variable2.set_gradient(0);
            }
        }
        // prit gradients 
        for (auto& variable2 : m_variables) {
            std::cout << "variable.grad " << variable2.gradient() << " id: " << variable2.get_id() << std::endl;
        }

        m_value = function();
        return m_value.gradient();
    }
};

} // namespace ln