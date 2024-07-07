// #include "math/Tensor.h"
// #include "math/function.h"
#include "math/new_tensor.h"
#include <iostream>
#include <vector>

using Tensor = sdlm::Tensor<float>;




int main() {
    std::cout << "----Testing gradients:----\n" << std::endl;
    std::vector<Tensor*> variables;
    Tensor a = 1;
    Tensor b = 2;
    Tensor c = 3;
    Tensor d = 4;
    Tensor e = 5;

    for(auto& var : {&a, &b, &c, &d, &e}) {
        var->set_requires_gradient(true);
    }

    b.set_requires_gradient(false);

    

    a = d + (e * c) - b; // 4 + (5 * 3) / 2 = 4 + 7.5 = 11.5

    // gradients:
    // da = 1
    // db = -5 * 3 / 2^2 = -3.75
    // dc = 5 / 2 = 2.5
    // dd = 1
    // de = 3 / 2 = 1.5


    // a = b * c + d;

    std::cout << "----Forward pass:----\n" << std::endl;

    c.print();
    b.print();
    a.print();
    d.print();
    e.print();

    std::cout << "----Gradients:----\n" << std::endl;

    c.gradient_tensor().print();
    b.gradient_tensor().print();
    a.gradient_tensor().print();
    d.gradient_tensor().print();
    e.gradient_tensor().print();


    std::cout << "Result: " << a.value() << std::endl;

    std::cout << "----Backward pass:----\n" << std::endl;
    a.backward();

    a.print();
    b.print();
    c.print();
    d.print();
    e.print();

    std::cout << "----Gradients:----\n" << std::endl;

    c.gradient_tensor().print();
    b.gradient_tensor().print();
    a.gradient_tensor().print();
    d.gradient_tensor().print();
    e.gradient_tensor().print();




    // variables.push_back(&a);
    // variables.push_back(&b);
    // variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
    // });

    // for (int i = 0; i < variables.size(); i++) {
    //     Tensor *var = variables[i];
    //     Tensor result = func.derivative(var);
    //     std::cout << "Result for variable " << i << ": " << result.value() << std::endl;
    //     std::cout << "Gradient for variable " << i << ": " << result.gradient() << std::endl;
    // }

    return 0;
}