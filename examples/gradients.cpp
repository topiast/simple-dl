#include "math/number.h"
#include "math/function.h"
#include <iostream>
#include <vector>

using Number = sdlm::Number<float>;




int main() {
    std::cout << "Testing gradients: " << std::endl;
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    sdlm::Function<float> func(variables, [&variables]() {
        return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
    });

    for (int i = 0; i < variables.size(); i++) {
        Number *var = variables[i];
        Number result = func.derivative(var);
        std::cout << "Result for variable " << i << ": " << result.value() << std::endl;
        std::cout << "Gradient for variable " << i << ": " << result.gradient() << std::endl;
    }

    return 0;
}