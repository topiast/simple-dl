#include "linear_algebra/number.h"
#include "linear_algebra/function.h"
#include <gtest/gtest.h>
#include <vector>

using Number = ln::Number<float>;

TEST(TestGradients, TestFunctionValuesAndGradients) {
    std::vector<Number> variables;
    variables.push_back(1);
    variables.push_back(2);
    variables.push_back(3);

    ln::Function<float> func(variables, [&variables]() {
        return variables[0] * variables[1] + variables[2] * 2; // 1 * 2 + 3 * 2 = 8
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
TEST(TestGradients, TestFunctionValuesAndGradients2) {
    std::vector<Number> variables;
    variables.push_back(1);
    variables.push_back(2);
    variables.push_back(3);

    ln::Function<float> func(variables, [&variables]() {
        return variables[0] / variables[1] + variables[2] * 2; // 1 / 2 + 3 * 2 = 6.5
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
// // test power
// TEST(TestGradients, TestFunctionValuesAndGradients3) {
//     std::vector<Number> variables;
//     variables.push_back(1);
//     variables.push_back(2);
//     variables.push_back(3);

//     ln::Function<float> func(variables, [&variables]() {
//         return ln::pow(variables[0], variables[1]) + variables[2] * 2; // 1 ^ 2 + 3 * 2 = 7
//     });

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {2, 0.693147, 2};

//     for (int i = 0; i < variables.size(); i++) {
//         Number result = func.derivative(variables[i]);

//         // Test values
//         EXPECT_FLOAT_EQ(result.value(), 7);

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);
//     }
// }
// test sqrt
TEST(TestGradients, TestFunctionValuesAndGradients4) {
    std::vector<Number> variables;
    variables.push_back(1);
    variables.push_back(2);
    variables.push_back(3);

    ln::Function<float> func(variables, [&variables]() {
        return ln::sqrt(variables[0]) + variables[2] * 2; // sqrt(1) + 3 * 2 = 7
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
TEST(TestGradients, TestFunctionValuesAndGradients5) {
    std::vector<Number> variables;
    variables.push_back(1);
    variables.push_back(2);
    variables.push_back(3);

    ln::Function<float> func(variables, [&variables]() {
        return ln::abs(variables[0]) + variables[2] * 2; // abs(1) + 3 * 2 = 7
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

// // test ln
// TEST(TestGradients, TestFunctionValuesAndGradients6) {
//     std::vector<Number> variables;
//     variables.push_back(1);
//     variables.push_back(2);
//     variables.push_back(3);

//     ln::Function<float> func(variables, [&variables]() {
//         return ln::log(variables[0]) + variables[2] * 2; // ln(1) + 3 * 2 = 7
//     });

//     // vector of expected function gradients
//     std::vector<float> expected_gradients = {1, 0, 2};

//     for (int i = 0; i < variables.size(); i++) {
//         Number result = func.derivative(variables[i]);

//         // Test values
//         EXPECT_FLOAT_EQ(result.value(), 7);

//         // Test gradients
//         // Adjust these assertions based on the expected gradient values for your function
//         EXPECT_FLOAT_EQ(result.gradient(), expected_gradients[i]);
//     }
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
