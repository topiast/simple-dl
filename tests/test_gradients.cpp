#include "math/number.h"
#include "math/function.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using Number = sdlm::Number<float>;

TEST(TestGradients, TestFunctionValuesAndGradients) {
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
    // });

    Number f = a * b + c * 2;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {2, 1, 2};


    for (int i = 0; i < variables.size(); i++) {;

        // Test values
        EXPECT_FLOAT_EQ(f.value(), 8);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}
// test division
TEST(TestGradients, TestFunctionValuesAndGradientsDivision) {
    std::vector<Number*> variables;

    Number a = 1;
    Number b = 2;
    Number c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) / (*variables[1]) + (*variables[2]) * 2; // 1 / 2 + 3 * 2 = 6.5
    // });

    Number f = a / b + c * 2;

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
// test power
TEST(TestGradients, TestFunctionValuesAndGradientsPower) {
    // variables
    float x1, x2, x3;
    // create random integers
    x1 = rand() % 10;
    x2 = rand() % 10;
    x3 = rand() % 10;

    std::vector<Number*> variables;
    
    Number a = x1;
    Number b = x2;
    Number c = x3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);
    

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return sdlm::pow((*variables[0]), (*variables[1])) + (*variables[2]) * 2; 
    // });

    Number f = sdlm::pow(a, b) + c * 2;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {
        std::powf(x1, x2 -1) * x2,
        std::powf(x1, x2) * std::logf(x1),
        2
        };

    for (int i = 0; i < variables.size(); i++) {
        // std::cout << "i: " << i << " gradient: " << variables[i]->gradient() << " expected: " << expected_gradients[i] << std::endl;

        // Test values
        EXPECT_FLOAT_EQ(f.value(), 
        std::powf(x1, x2) + x3 * 2
        );

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);

    }
}
// test sqrt
TEST(TestGradients, TestFunctionValuesAndGradientsSqrt) {
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return sdlm::sqrt((*variables[0])) + (*variables[2]) * 2; // sqrt(1) + 3 * 2 = 7
    // });

    Number f = sdlm::sqrt(a) + c * 2;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {0.5, 0, 2};

    for (int i = 0; i < variables.size(); i++) {
        // Test values
        EXPECT_FLOAT_EQ(f.value(), 7);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}

// test abs
TEST(TestGradients, TestFunctionValuesAndGradientsAbs) {
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return sdlm::abs((*variables[0])) + (*variables[2]) * 2; // abs(1) + 3 * 2 = 7
    // });

    Number f = sdlm::abs(a) + c * 2;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {1, 0, 2};

    for (int i = 0; i < variables.size(); i++) {
        // Test values
        EXPECT_FLOAT_EQ(f.value(), 7);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}

// test sdlm
TEST(TestGradients, TestFunctionValuesAndGradientssdlm) {
    // variables
    float x1, x2, x3;
    // create random integers
    x1 = rand() % 10;
    x2 = rand() % 10;
    x3 = rand() % 10;

    std::vector<Number*> variables;
    
    Number a = x1;
    Number b = x2;
    Number c = x3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return sdlm::log((*variables[0])) + (*variables[2]) * 2; // sdlm(1) + 3 * 2 = 7
    // });

    Number f = sdlm::log(a) + c * 2;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {1/x1, 0, 2};

    for (int i = 0; i < variables.size(); i++) {
        // Test values
        EXPECT_FLOAT_EQ(f.value(), std::log(x1) + x3 * 2);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}
// a more complicated function
TEST(TestGradients, TestFunctionValuesAndGradientsComplicated) {
    // variables
    float x1, x2, x3;
    // create random integers
    x1 = (rand() % 10);
    x2 = rand() % 10;
    x3 = rand() % 10;

    // print out the random integers
    // std::cout << "x1: " << x1 << std::endl;
    // std::cout << "x2: " << x2 << std::endl;
    // std::cout << "x3: " << x3 << std::endl;

    std::vector<Number*> variables;
    
    Number a = x1;
    Number b = x2;
    Number c = x3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {

    //     return sdlm::pow((*variables[0]), (*variables[1])) + sdlm::sqrt((*variables[2])) * 2; // 2^3 + sqrt(4) * 2 = 14
    // });

    Number f = sdlm::pow(a, b) + sdlm::sqrt(c) * 2;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {
        std::powf(x1, x2 -1) * x2,
        std::powf(x1, x2) * std::logf(x1),
        2 / (2 * std::sqrtf(x3))
        };

    for (int i = 0; i < variables.size(); i++) {
        // Test values
        EXPECT_FLOAT_EQ(f.value(), 
        std::powf(x1, x2) + std::sqrtf(x3) * 2
        );

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);

    }
}

// test count_gradient false for some variables
TEST(TestGradients, TestFunctionValuesAndGradientsCountGradient) {
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;

    a.set_count_gradient(false);

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
    // });

    Number f = a * b + c * 2;

    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {0, 1, 2};

    for (int i = 0; i < variables.size(); i++) {
        // Test values
        EXPECT_FLOAT_EQ(f.value(), 8);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}
// // test sigmoid
// TEST(TestGradients, TestFunctionValuesAndGradientsSigmoid) {
//     std::vector<Number*> variables;
//     Number a = rand() % 10;

//     std::cout << "a: " << a << std::endl;

//     variables.push_back(&a);



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

// test same variable multiple times in function
TEST(TestGradients, TestFunctionValuesAndGradientsSameVariable) {
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) * (*variables[0]) + (*variables[0]) * 2; // 1 * 1 + 1 * 2 = 3
    // });

    Number f = a * b + c * 2;

    f.zero_grad();
    // a.debug_print();
    f.backward();
    // a.debug_print();

    // f.get_grad_fn()->visualize();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {2, 1, 2};

    for (int i = 0; i < variables.size(); i++) {
        // Test values
        EXPECT_FLOAT_EQ(f.value(), 8);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}

// test multiple iterations of backward and zero_grad
TEST(TestGradients, TestFunctionValuesAndGradientsMultipleIterations) {
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    // sdlm::Function<float> func(variables, [&variables]() {
    //     return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
    // });

    Number f = sdlm::pow(a * b + c * 2, b) + c * 2;
    float value = f.value();
    f.zero_grad();
    f.backward();
    f = sdlm::pow(a * b + c * 2, b) + c * 2;
    f.zero_grad();
    f.backward();

    // vector of expected function gradients
    std::vector<float> expected_gradients = {32, 149.08426, 34};

    for (int i = 0; i < variables.size(); i++) {
        // Test values
        EXPECT_FLOAT_EQ(f.value(), value);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(variables[i]->gradient(), expected_gradients[i]);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
