#include "math/tensor.h"
#include <iostream>
#include <vector>
#include "ml/loss_functions.h"

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

    // b.set_requires_gradient(false);

    

    a = d + (e * c) - b; // 4 + (5 * 3) - 2 = 17

    // gradients:
    // da = 0
    // db = -1
    // dc = 5
    // dd = 1
    // de = 3



    // a = b * c + d;

    std::cout << "----Forward pass:----\n" << std::endl;

    a.print();
    b.print();
    c.print();
    d.print();
    e.print();

    std::cout << "----Gradients:----\n" << std::endl;

    a.gradient_tensor().print();
    b.gradient_tensor().print();
    c.gradient_tensor().print();
    d.gradient_tensor().print();
    e.gradient_tensor().print();


    std::cout << "Result: " << a.value() << std::endl;

    std::cout << "\n----Backward pass:----\n" << std::endl;
    a.backward();

    a.print();
    b.print();
    c.print();
    d.print();
    e.print();

    std::cout << "----Gradients:----\n" << std::endl;

    a.gradient_tensor().print();
    b.gradient_tensor().print();
    c.gradient_tensor().print();
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
    // create tensor with 1,2,3 values
    Tensor input({1, 3});
    input.set_values(0, {1, 2, 3});
    // create tensor with 1,0,0 values
    Tensor target({1, 3});
    target.set_values(0, {1, 0, 0});


    input.set_requires_gradient(true);
    // input.print();

    // Tensor& x = input[0];
    // (-( (std::exp(input[0]) / (std::exp(input[0]) + std::exp(input[1] + std::exp(input[2])))).log() )).backward();
    // x.debug_print();
    // std::cout << "softmax: " << std::endl;
    // Tensor sum_exp = (std::exp(input[0]) + std::exp(input[1] + std::exp(input[2])));
    // ???

    // Tensor f = -(target[0] * ( (std::exp(input[0]) / sum_exp).log() ) + target[1] * ( (std::exp(input[1]) / sum_exp).log() ) + target[2] * ( (std::exp(input[2]) / sum_exp).log() ));
    Tensor f = sdl::cross_entropy(input.softmax(), target);
    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {-0.90996945, 0.24472848, 0.66524094};
    // Test values
    if (f.value() != 2.4076059f) {
        std::cout << "Expected value: 2.4076059" << std::endl;
        std::cout << "Actual value: " << f.value() << std::endl;
    } else {
        std::cout << "Value test passed" << std::endl;
    }

    for (int i = 0; i < input.size(); i++) {
        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        if (input.gradients()[i] != expected_gradients[i]) {
            std::cout << "Expected gradient: " << expected_gradients[i] << std::endl;
            std::cout << "Actual gradient: " << input.gradients()[i] << std::endl;
        } else {
            std::cout << "Gradient test passed" << std::endl;
        }
    }

    return 0;
} 