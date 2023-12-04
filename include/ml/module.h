#pragma once

#include <vector>
#include <iostream>
#include "math/number.h"


template <typename T>
class Module {
public:
    virtual ~Module() {}

    virtual void forward() = 0;
    
    virtual std::vector<T*> get_parameters() = 0;
};
