#include "linear_algebra/number.h"
#include "linear_algebra/function.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using Number = ln::Number<float>;

TEST(TestGradients, TestFunctionValuesAndGradients) {
    std::vector<Number*> variables;
    Number a = 1;
    Number b = 2;
    Number c = 3;

    variables.push_back(&a);
    variables.push_back(&b);
    variables.push_back(&c);

    ln::Function<float> func(variables, [&variables]() {
        return (*variables[0]) * (*variables[1]) + (*variables[2]) * 2; // 1 * 2 + 3 * 2 = 8
    });

    // vector of expected function gradients
    std::vector<float> expected_gradients = {2, 1, 2};


    for (int i = 0; i < variables.size(); i++) {
        Number result = func.derivative(variables[i]);

        // Test values
        EXPECT_FLOAT_EQ(result.value(), 8);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);
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

    ln::Function<float> func(variables, [&variables]() {
        return (*variables[0]) / (*variables[1]) + (*variables[2]) * 2; // 1 / 2 + 3 * 2 = 6.5
    });

    // vector of expected function gradients
    std::vector<float> expected_gradients = {0.5, -0.25, 2};

    for (int i = 0; i < variables.size(); i++) {
        Number result = func.derivative(variables[i]);

        // Test values
        EXPECT_FLOAT_EQ(result.value(), 6.5);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);
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
    

    ln::Function<float> func(variables, [&variables]() {
        return ln::pow((*variables[0]), (*variables[1])) + (*variables[2]) * 2; 
    });

    // vector of expected function gradients
    std::vector<float> expected_gradients = {
        std::powf(x1, x2 -1) * x2,
        std::powf(x1, x2) * std::logf(x2),
        2
        };

    for (int i = 0; i < variables.size(); i++) {
        Number result = func.derivative(variables[i]);

        // Test values
        EXPECT_FLOAT_EQ(result.value(), 
        std::powf(x1, x2) + x3 * 2
        );

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);

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

    ln::Function<float> func(variables, [&variables]() {
        return ln::sqrt((*variables[0])) + (*variables[2]) * 2; // sqrt(1) + 3 * 2 = 7
    });

    // vector of expected function gradients
    std::vector<float> expected_gradients = {0.5, 0, 2};

    for (int i = 0; i < variables.size(); i++) {
        Number result = func.derivative(variables[i]);

        // Test values
        EXPECT_FLOAT_EQ(result.value(), 7);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);
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

    ln::Function<float> func(variables, [&variables]() {
        return ln::abs((*variables[0])) + (*variables[2]) * 2; // abs(1) + 3 * 2 = 7
    });

    // vector of expected function gradients
    std::vector<float> expected_gradients = {1, 0, 2};

    for (int i = 0; i < variables.size(); i++) {
        Number result = func.derivative(variables[i]);

        // Test values
        EXPECT_FLOAT_EQ(result.value(), 7);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);
    }
}

// test ln
TEST(TestGradients, TestFunctionValuesAndGradientsLn) {
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

    ln::Function<float> func(variables, [&variables]() {
        return ln::log((*variables[0])) + (*variables[2]) * 2; // ln(1) + 3 * 2 = 7
    });

    // vector of expected function gradients
    std::vector<float> expected_gradients = {1/x1, 0, 2};

    for (int i = 0; i < variables.size(); i++) {
        Number result = func.derivative(variables[i]);

        // Test values
        EXPECT_FLOAT_EQ(result.value(), std::log(x1) + x3 * 2);

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);
    }
}
// a more complicated function
TEST(TestGradients, TestFunctionValuesAndGradientsComplicated) {
    // variables
    float x1, x2, x3;
    // create random integers
    x1 = -(rand() % 10);
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

    ln::Function<float> func(variables, [&variables]() {

        return ln::pow((*variables[0]), (*variables[1])) + ln::sqrt((*variables[2])) * 2; // 2^3 + sqrt(4) * 2 = 14
    });

    // vector of expected function gradients
    std::vector<float> expected_gradients = {
        std::powf(x1, x2 -1) * x2,
        std::powf(x1, x2) * std::logf(x2),
        2 / (2 * std::sqrtf(x3))
        };

    for (int i = 0; i < variables.size(); i++) {
        Number result = func.derivative(variables[i]);

        // Test values
        EXPECT_FLOAT_EQ(result.value(), 
        std::powf(x1, x2) + std::sqrtf(x3) * 2
        );

        // Test gradients
        // Adjust these assertions based on the expected gradient values for your function
        EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);

    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
