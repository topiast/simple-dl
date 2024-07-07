#pragma once

#include "math/new_tensor.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <memory>

namespace sdlm {

template <typename T>
class Tensor;

template <typename T>
class Operator {
public:
    virtual void evaluate(const Tensor<T>& gradient) const = 0;
    virtual void print() const = 0;
    virtual ~Operator() { }

    void debug_evalutate(const T& x) const {
        // check if member variables are set
        std::cout << "m_saved_values.size(): " << m_saved_values.size() << std::endl;
        std::cout << "m_next_operators.size(): " << m_next_operators.size() << std::endl;
        evaluate(x);
    }

    void next_evaluate(const int i, const Tensor<T>& x) const {
        if (i >= m_next_operators.size()) {
            std::cout << "Error: next operator index out of range" << std::endl;
            return;
        }
        // auto next_operator = m_next_operators[i];
        if (m_next_operators[i] != nullptr) {
            m_next_operators[i]->evaluate(x);
        } 
    }

    Operator<T>& operator=(const Operator<T>& rhs) {
        m_saved_values = rhs.m_saved_values;
        m_next_operators = rhs.m_next_operators;

        std::cout << "Operator copy called" << std::endl;

        return *this;
    }

protected:
    std::vector<Tensor<T>> m_saved_values;
    std::vector<std::shared_ptr<Operator<T>>> m_next_operators;
    
    Operator(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : m_saved_values(saved_values), m_next_operators(next_operators) { }
private:

     bool verify_input() {
        for (const Tensor<T>& t : m_saved_values) {
            if (t.requires_grad()) {
                return false;
            }
        }
        return true;
     }
};

template <typename T>
class AccumulateGrad : public Operator<T> {
private:
    Tensor<T>* m_variable;

public:
    AccumulateGrad(Tensor<T>* variable)
        : Operator<T>(std::vector<Tensor<T>>(), std::vector<std::shared_ptr<Operator<T>>>()), m_variable(variable) { }


    void evaluate(const Tensor<T>& x) const override {
        m_variable->set_gradient(m_variable->gradient_tensor() + x);
    }

    void print() const override {
        std::cout << "AccumulateGrad" << std::endl;
    }
};

template <typename T>
class AddBack : public Operator<T> {
public:
    AddBack(const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(std::vector<Tensor<T>>(), next_operators) { }
    ~AddBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        // evaluate the next operators
        this->next_evaluate(0, gradient);
        this->next_evaluate(1, gradient);
    }

    void print() const override {
        std::cout << "AddBack" << std::endl;
    }
};

template <typename T>
class SubBack : public Operator<T> {
public:
    SubBack(const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(std::vector<Tensor<T>>(), next_operators) { }
    ~SubBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        // evaluate the next operators
        this->next_evaluate(0, gradient);
        this->next_evaluate(1, -gradient);
    }

    void print() const override {
        std::cout << "SubBack" << std::endl;
    }
};

template <typename T>
class NegBack : public Operator<T> {
public:
    NegBack(const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(std::vector<Tensor<T>>(), next_operators) { }
    ~NegBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        // evaluate the next operators
        this->next_evaluate(0, -gradient);
    }

    void print() const override {
        std::cout << "NegBack" << std::endl;
    }
};

// element-wise multiplication
template <typename T>
class MulBack : public Operator<T> {
public:
    MulBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~MulBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];
        const Tensor<T>& b = saved_values[1];

        // calculate the gradients
        Tensor<T> grad_a = gradient * b; // TODO: check if it matters if these have requires_grad set to true
        Tensor<T> grad_b = gradient * a;

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
        this->next_evaluate(1, grad_b);
    }

    void print() const override {
        std::cout << "MulBack" << std::endl;
    }
};

// element-wise division
template <typename T>
class DivBack : public Operator<T> {
public:
    DivBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~DivBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];
        const Tensor<T>& b = saved_values[1];

        // calculate the gradients
        Tensor<T> grad_a = gradient / b; 
        Tensor<T> grad_b = -gradient * a / (b * b);

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
        this->next_evaluate(1, grad_b);
    }

    void print() const override {
        std::cout << "DivBack" << std::endl;
    }
};

// element-wise power
template <typename T>
class PowBack : public Operator<T> {
public:
    PowBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~PowBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];
        const Tensor<T>& b = saved_values[1];

        // calculate the gradients
        Tensor<T> grad_a = gradient * b * a.pow(b - 1); 
        Tensor<T> grad_b = gradient * a.pow(b) * a.log();

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
        this->next_evaluate(1, grad_b);
    }

    void print() const override {
        std::cout << "PowBack" << std::endl;
    }
};

// element-wise square root
template <typename T>
class SqrtBack : public Operator<T> {
public:
    SqrtBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~SqrtBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = gradient / (Tensor<T>(2.f) * a.sqrt());

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "SqrtBack" << std::endl;
    }
};

// element-wise log
template <typename T>
class LogBack : public Operator<T> {
public:
    LogBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~LogBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = gradient / a;

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "LogBack" << std::endl;
    }
};

// element-wise exp
template <typename T>
class ExpBack : public Operator<T> {
public:
    ExpBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~ExpBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = gradient * a.exp();

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "ExpBack" << std::endl;
    }
};

// element-wise abs
template <typename T>
class AbsBack : public Operator<T> {
public:
    AbsBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~AbsBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = gradient * (a > 0.f ? Tensor<T>(1.f) : Tensor<T>(-1.f));

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "AbsBack" << std::endl;
    }
};

// element-wise sin
template <typename T>
class SinBack : public Operator<T> {
public:
    SinBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~SinBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = gradient * a.cos();

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "SinBack" << std::endl;
    }
};

// element-wise cos
template <typename T>
class CosBack : public Operator<T> {
public:
    CosBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~CosBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = -gradient * a.sin();

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "CosBack" << std::endl;
    }
};

// element-wise tan
template <typename T>
class TanBack : public Operator<T> {
public:
    TanBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~TanBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = gradient / a.cos().pow(2.f);

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "TanBack" << std::endl;
    }
};


} // namespace sdlm