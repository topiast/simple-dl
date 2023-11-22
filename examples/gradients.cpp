#include "linear_algebra/number.h"
#include "linear_algebra/function.h"
#include <iostream>
#include <vector>

using Number = ln::Number<float>;




int main() {
    std::vector<Number> variables;
    variables.push_back(1);
    variables.push_back(2);
    variables.push_back(3);

    ln::Function<float> func(variables, [&variables]() {
        return variables[0] * variables[1] + variables[2] * 2; // 1 * 2 + 3 * 2 = 8
    });

    for (int i = 0; i < variables.size(); i++) {
        Number result = func.derivative(variables[i]);
        std::cout << "Result for variable " << i << ": " << result.value() << std::endl;
        std::cout << "Gradient for variable " << i << ": " << result.gradient() << std::endl;
    }

    return 0;
}