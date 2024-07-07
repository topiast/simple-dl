// #include "math/number.h"
// #include "math/tensor.h"
#include "math/new_tensor.h"

// #include "ml/loss_functions.h"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// using Number = sdlm::Number<float>;
// using Tensor = sdlm::Tensor<Number>;
using Tensor = sdlm::Tensor<float>;

TEST(TestGradients, TestFunctionValuesAndGradients) {
    std::vector<Tensor*> variables;
    Tensor a = 1;
    Tensor b = 2;
    Tensor c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // set cout gradient to true
    for (int i = 0; i < variables.size(); i++) {
        variables[i]->set_requires_gradient(true);
    }

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
    // });

    Tensor f = a * b + c;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {2, 1, 1};


    for (int i = 0; i < variables.size(); i++) {;

        // Test values
        EXPECT_FLOAT_EQ(f.value(), 5);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}

// test addition and negative operator
TEST(TestGradients, TestFunctionValuesAndGradientsAdditionNegative) {
    std::vector<Tensor*> variables;
    Tensor a = 1;
    Tensor b = 2;
    Tensor c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);
    // set cout gradient to true
    for (int i = 0; i < variables.size(); i++) {
        variables[i]->set_requires_gradient(true);
    }

    Tensor f = -(a + b + c * 2);

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {-1, -1, -2};

    for (int i = 0; i < variables.size(); i++) {
        // Test values
        EXPECT_FLOAT_EQ(f.value(), -9);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}
// test division
TEST(TestGradients, TestFunctionValuesAndGradientsDivision) {
    std::vector<Tensor*> variables;

    Tensor a = 1;
    Tensor b = 2;
    Tensor c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);
    // set cout gradient to true
    for (int i = 0; i < variables.size(); i++) {
        variables[i]->set_requires_gradient(true);
    }
    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) / (*variables[1]) + (*variables[2]) * 2; // 1 / 2 + 3 * 2 = 6.5
    // });

    Tensor f = a / b + c * 2;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {0.5, -0.25, 2};

    for (int i = 0; i < variables.size(); i++) {

        // Test values
        EXPECT_FLOAT_EQ(f.value(), 6.5);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}
// // test power
// TEST(TestGradients, TestFunctionValuesAndGradientsPower) {
//     // variables
//     float x1, x2, x3;
//     // create random integers
//     x1 = rand() % 10;
//     x2 = rand() % 10;
//     x3 = rand() % 10;

//     std::vector<Number*> variables;
    
//     Number a = x1;
//     Number b = x2;
//     Number c = x3;

//     variables.push_back(&a);
//     variables.push_back(&b);
//     variables.push_back(&c);
//         // set cout gradient to true
//     for (int i = 0; i < variables.size(); i++) {
//         variables[i]->set_requires_gradient(true);
//     }

//     // sdlm::Function<float> func(variables, [&variables]() {
//     //     return sdlm::pow((*variables[0]), (*variables[1])) + (*variables[2]) * 2; 
//     // });

//     Number f = sdlm::pow(a, b) + c * 2;

//     f.backward();

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {
//         std::pow(x1, x2 -1) * x2,
//         std::pow(x1, x2) * std::log(x1),
//         2
//         };

//     for (int i = 0; i < variables.size(); i++) {
//         // std::cout << "i: " << i << " gradient: " << variables[i]->gradient() << " expected: " << expected_gradients[i] << std::endl;

//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), 
//         std::pow(x1, x2) + x3 * 2
//         );

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);

//     }
// }
// // test sqrt
// TEST(TestGradients, TestFunctionValuesAndGradientsSqrt) {
//     std::vector<Number*> variables;
//     Number a = 1;
//     Number b = 2;
//     Number c = 3;

//     variables.push_back(&a);
//     variables.push_back(&b);
//     variables.push_back(&c);
//     // set cout gradient to true
//     for (int i = 0; i < variables.size(); i++) {
//         variables[i]->set_requires_gradient(true);
//     }
//     // sdlm::Function<float> func(variables, [&variables]() {
//     //     return sdlm::sqrt((*variables[0])) + (*variables[2]) * 2; // sqrt(1) + 3 * 2 = 7
//     // });

//     Number f = sdlm::sqrt(a) + c * 2;

//     f.backward();

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {0.5, 0, 2};

//     for (int i = 0; i < variables.size(); i++) {
//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), 7);

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
//     }
// }

// // test abs
// TEST(TestGradients, TestFunctionValuesAndGradientsAbs) {
//     std::vector<Number*> variables;
//     Number a = 1;
//     Number b = 2;
//     Number c = 3;

//     variables.push_back(&a);
//     variables.push_back(&b);
//     variables.push_back(&c);
//     // set cout gradient to true
//     for (int i = 0; i < variables.size(); i++) {
//         variables[i]->set_requires_gradient(true);
//     }
//     // sdlm::Function<float> func(variables, [&variables]() {
//     //     return sdlm::abs((*variables[0])) + (*variables[2]) * 2; // abs(1) + 3 * 2 = 7
//     // });

//     Number f = sdlm::abs(a) + c * 2;

//     f.backward();

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {1, 0, 2};

//     for (int i = 0; i < variables.size(); i++) {
//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), 7);

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
//     }
// }

// // test sdlm
// TEST(TestGradients, TestFunctionValuesAndGradientssdlm) {
//     // variables
//     float x1, x2, x3;
//     // create random integers
//     x1 = rand() % 10;
//     x2 = rand() % 10;
//     x3 = rand() % 10;

//     std::vector<Number*> variables;
    
//     Number a = x1;
//     Number b = x2;
//     Number c = x3;

//     variables.push_back(&a);
//     variables.push_back(&b);
//     variables.push_back(&c);
//     // set cout gradient to true
//     for (int i = 0; i < variables.size(); i++) {
//         variables[i]->set_requires_gradient(true);
//     }
//     // sdlm::Function<float> func(variables, [&variables]() {
//     //     return sdlm::log((*variables[0])) + (*variables[2]) * 2; // sdlm(1) + 3 * 2 = 7
//     // });

//     Number f = sdlm::log(a) + c * 2;

//     f.backward();

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {1/x1, 0, 2};

//     for (int i = 0; i < variables.size(); i++) {
//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), std::log(x1) + x3 * 2);

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
//     }
// }
// // a more complicated function
// TEST(TestGradients, TestFunctionValuesAndGradientsComplicated) {
//     // variables
//     float x1, x2, x3;
//     // create random integers
//     x1 = (rand() % 10);
//     x2 = rand() % 10;
//     x3 = rand() % 10;

//     // print out the random integers
//     // std::cout << "x1: " << x1 << std::endl;
//     // std::cout << "x2: " << x2 << std::endl;
//     // std::cout << "x3: " << x3 << std::endl;

//     std::vector<Number*> variables;
    
//     Number a = x1;
//     Number b = x2;
//     Number c = x3;

//     variables.push_back(&a);
//     variables.push_back(&b);
//     variables.push_back(&c);
//     // set cout gradient to true
//     for (int i = 0; i < variables.size(); i++) {
//         variables[i]->set_requires_gradient(true);
//     }
//     // sdlm::Function<float> func(variables, [&variables]() {

//     //     return sdlm::pow((*variables[0]), (*variables[1])) + sdlm::sqrt((*variables[2])) * 2; // 2^3 + sqrt(4) * 2 = 14
//     // });

//     Number f = sdlm::pow(a, b) + sdlm::sqrt(c) * 2;

//     f.backward();

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {
//         std::pow(x1, x2 -1) * x2,
//         std::pow(x1, x2) * std::log(x1),
//         2 / (2 * std::sqrt(x3))
//         };

//     for (int i = 0; i < variables.size(); i++) {
//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), 
//         std::pow(x1, x2) + std::sqrt(x3) * 2
//         );

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);

//     }
// }

// // test count_gradient false for some variables
// TEST(TestGradients, TestFunctionValuesAndGradientsCountGradient) {
//     std::vector<Number*> variables;
//     Number a = 1;
//     Number b = 2;
//     Number c = 3;


//     variables.push_back(&a);
//     variables.push_back(&b);
//     variables.push_back(&c);
//     // set cout gradient to true
//     for (int i = 0; i < variables.size(); i++) {
//         variables[i]->set_requires_gradient(true);
//     }
//     a.set_requires_gradient(false);
//     // sdlm::Function<float> func(variables, [&variables]() {
//     //     return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
//     // });

//     Number f = a * b + c * 2;

//     f.backward();

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {0, 1, 2};

//     for (int i = 0; i < variables.size(); i++) {
//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), 8);

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
//     }
// }
// // test sigmoid
// TEST(TestGradients, TestFunctionValuesAndGradientsSigmoid) {
//     std::vector<Number*> variables;
//     Number a = rand() % 10;


//     variables.push_back(&a);

//     a.set_requires_gradient(true);

//     // sdlm::Function<float> func(variables, [&variables]() {
//     //     return Number(1) / (Number(1) + sdlm::exp(-(*variables[0]))); 
//     // });

//     Number f = Number(1) / (Number(1) + sdlm::exp(-a));
//     f.backward();

//     float value = 1 / (1 + std::exp(-a.value()));

//     for (int i = 0; i < variables.size(); i++) {

//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), value );

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), (value) * (1 - value));
//     }

// }

// // test same variable multiple times in function
// TEST(TestGradients, TestFunctionValuesAndGradientsSameVariable) {
//     std::vector<Number*> variables;
//     Number a = 1;
//     Number b = 2;
//     Number c = 3;

//     variables.push_back(&a);
//     variables.push_back(&b);
//     variables.push_back(&c);
//     // set cout gradient to true
//     for (int i = 0; i < variables.size(); i++) {
//         variables[i]->set_requires_gradient(true);
//     }
//     // sdlm::Function<float> func(variables, [&variables]() {
//     //     return (*variables[0]) * (*variables[0]) + (*variables[0]) * 2; // 1 * 1 + 1 * 2 = 3
//     // });

//     Number f = a * a + a * 2;
    
//     f.backward();


//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {4, 0, 0};

//     for (int i = 0; i < variables.size(); i++) {
//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), 3);

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
//     }
// }

// // test multiple iterations of backward and zero_grad
// TEST(TestGradients, TestFunctionValuesAndGradientsMultipleIterations) {
//     std::vector<Number*> variables;
//     Number a = 1;
//     Number b = 2;
//     Number c = 3;

//     variables.push_back(&a);
//     variables.push_back(&b);
//     variables.push_back(&c);
//     // set cout gradient to true
//     for (int i = 0; i < variables.size(); i++) {
//         variables[i]->set_requires_gradient(true);
//     }
//     // sdlm::Function<float> func(variables, [&variables]() {
//     //     return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
//     // });

//     Number f = sdlm::pow(a * b + c * 2, b) + c * 2;
//     float value = f.value();
//     for (auto& p : variables) {
//         p->zero_grad();
//     }
//     f.backward();
//     f = sdlm::pow(a * b + c * 2, b) + c * 2;
//     for (auto& p : variables) {
//         p->zero_grad();
//     }
//     f.backward();

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {32, 149.08426, 34};

//     for (int i = 0; i < variables.size(); i++) {
//         // Test values
//         EXPECT_FLOAT_EQ(f.value(), value);

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
//     }
// }
// // test cross entropy loss function
// TEST(TestGradients, TestFunctionValuesAndGradientsCrossEntropy) {
//     // create tensor with 1,2,3 values
//     Tensor input({1, 3});
//     input.set_values(0, {1, 2, 3});
//     // create tensor with 1,0,0 values
//     Tensor target({1, 3});
//     target.set_values(0, {1, 0, 0});

//     std::vector<Number*> variables;

//     for (auto& v : input.get_values()) {
//         variables.push_back(&v);
//         v.set_requires_gradient(true);
//     }
//     // input.print();

//     // Number& x = input[0];
//     // (-( (std::exp(input[0]) / (std::exp(input[0]) + std::exp(input[1] + std::exp(input[2])))).log() )).backward();
//     // x.debug_print();
//     // std::cout << "softmax: " << std::endl;
//     // Number sum_exp = (std::exp(input[0]) + std::exp(input[1] + std::exp(input[2])));
//     // ???

//     // Number f = -(target[0] * ( (std::exp(input[0]) / sum_exp).log() ) + target[1] * ( (std::exp(input[1]) / sum_exp).log() ) + target[2] * ( (std::exp(input[2]) / sum_exp).log() ));
//     Number f = sdl::cross_entropy(input.softmax(), target);
//     f.backward();

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {-0.90996945, 0.24472848, 0.66524094};
//     // Test values
//     EXPECT_FLOAT_EQ(f.value(), 2.4076059);

//     for (int i = 0; i < variables.size(); i++) {

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
//     }
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
