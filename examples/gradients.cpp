#include "math/number.h"
#include "math/function.h"
#include <iostream>
#include <vector>

using Number = sdlm::Number<float>;




int main() {
    std::cout << "----Testing gradients:----" << std::endl;
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;
    Number d = 4;
    Number e = 5;

    

    a = d + (e * c) / b; // 4 + (5 * 3) / 2 = 4 + 7.5 = 11.5

    // gradients:
    // da = 1
    // db = -5 * 3 / 2^2 = -3.75
    // dc = 5 / 2 = 2.5
    // dd = 1
    // de = 3 / 2 = 1.5


    // a = b * c + d;


    c.debug_print();
    b.debug_print();
    a.debug_print();
    d.debug_print();
    e.debug_print();

    std::cout << "Result: " << a.value() << std::endl;

    std::cout << "----Backward pass:----" << std::endl;
    a.backward();

    a.debug_print();
    b.debug_print();
    c.debug_print();
    d.debug_print();
    e.debug_print();



    // variables.push_back(&a);
    // variables.push_back(&b);
    // variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
    // });

    // for (int i = 0; i < variables.size(); i++) {
    //     Number *var = variables[i];
    //     Number result = func.derivative(var);
    //     std::cout << "Result for variable " << i << ": " << result.value() << std::endl;
    //     std::cout << "Gradient for variable " << i << ": " << result.gradient() << std::endl;
    // }

    return 0;
}