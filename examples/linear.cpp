#include "linear_algebra/number.h"
#include "linear_algebra/tensor.h"
#include "ml/linear.h"

#include <iostream>


int main() {
    //create tensor
    ln::Tensor<int> tensor1, tensor2, result;
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

}
