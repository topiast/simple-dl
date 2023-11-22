#include "linear_algebra/number.h"
#include "linear_algebra/function.h"
#include <iostream>
#include <vector>

using Number = ln::Number<float>;

Number forward(Number& variable, std::vector<Number>& variables, const std::function<Number()>& function) {
    for (auto& variable2 : variables) {
        if (variable.get_id() == variable2.get_id()) {
            variable2.set_gradient(1);
        } else {
            variable2.set_gradient(0);
        }
    }


    Number value = function();
    return value;
}


int main() {
    std::vector<Number> variables;
    variables.push_back(Number(1));
    variables.push_back(Number(2));
    variables.push_back(Number(3));


    // print variable ids
    for (int i = 0; i < variables.size(); i++) {
        std::cout << "variable " << i << " id: " << variables[i].get_id() << std::endl;
    }

    for (int i = 0; i < variables.size(); i++) {
        Number result = forward(variables[i], variables, [&variables]() {
            return variables[0] * variables[1] + variables[2] * 2; // 1 * 2 + 3 = 5
        });

        std::cout << "result: " << result.value() << std::endl;
        std::cout << "gradient: " << result.gradient() << std::endl;
    }
}