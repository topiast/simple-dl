#include "math/number.h"
#include "math/function.h"
#include "math/tensor.h"
#include <iostream>
#include <vector>

using Number = sdlm::Number<float>;


int main() {
    //create tensor
    sdlm::Tensor<int> tensor1, tensor2, result;
    tensor1.ones({2, 3});  // Example tensor with shape 2x3 filled with ones
    tensor2.ones({3, 2});  // Example tensor with shape 3x2 filled with ones

    // [[2, 5, 3]
    // [[3, 6, 2]
    
    // [[3, 5]
    // [[1, 4]
    // [[3, 1]]

    // result should be 
    //[[20 33]
    //[21 41]]

    // assign values to tensor1
    tensor1.set_values(0,{1, 2, 3});
    tensor1.set_values(1,{4, 5, 6});

    // assign values to tensor2
    tensor2.set_values(0,{1, 2});
    tensor2.set_values(1,{3, 4});
    tensor2.set_values(2,{5, 6});

    // print tensor1
    std::cout << "Tensor1: " << std::endl;
    tensor1.print();

    // print tensor2
    std::cout << "Tensor2: " << std::endl;
    tensor2.print();



    result = tensor2.matmul(tensor1);
    std::cout << "Result for tensor matmul t1 * t2: " << std::endl;
    result.print();

    tensor2 = tensor2.transpose();
    std::cout << "Tensor2.T: " << std::endl;
    tensor2.print();

    std::cout << "Testing 3d tensors: " << std::endl;

    // now test with 3d tensors
    sdlm::Tensor<int> tensor3, tensor4, result2;
    tensor3.ones({2, 3, 2});  // Example tensor with shape 2x3x2 filled with ones
    tensor4.ones({2, 2, 3});  // Example tensor with shape 2x2x3 filled with ones

    // assign values to tensor3
// Assign values to tensor3 using set_values method
    tensor3.set_values({0, 0}, {1, 2}); 
    tensor3.set_values({0, 1}, {3, 4}); 
    tensor3.set_values({0, 2}, {5, 6}); 

    tensor3.set_values({1, 0}, {7, 8});   
    tensor3.set_values({1, 1}, {9, 10});  
    tensor3.set_values({1, 2}, {11, 12}); 

    // Assign values to tensor4 using set_values method
    tensor4.set_values({0, 0}, {1, 2, 3});
    tensor4.set_values({0, 1}, {4, 5, 6});

    tensor4.set_values({1, 0}, {7, 8, 9});
    tensor4.set_values({1, 1}, {10, 11, 12});
    

    // print tensor3
    std::cout << "Tensor3: " << std::endl;
    tensor3.print();
    tensor3.print_data();

    // print tensor4
    std::cout << "Tensor4: " << std::endl;
    tensor4.print();
    tensor4.print_data();

    result2 = tensor3 * tensor3;
    std::cout << "Result for tensor element-wise product t3 * t3: " << std::endl;
    result2.print();


    return 0;
}