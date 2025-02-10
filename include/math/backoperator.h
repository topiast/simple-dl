#pragma once

#include "math/tensor.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <functional>

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

    bool requires_grad(const int i) const {
        if (i >= m_next_operators.size()) {
            std::cout << "Error: next operator index out of range" << std::endl;
            return false;
        }
        return (m_next_operators[i] != nullptr);
    }

    void next_evaluate(const int i, const Tensor<T>& x) const {
        if (i >= m_next_operators.size()) {
            std::cout << "Error: next operator index out of range" << std::endl;
            return;
        }
        // auto next_operator = m_next_operators[i];
        if (requires_grad(i)) {
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
        if (x.shape() != m_variable->shape()) {
            auto reduced = x.reduce_sum(0);
            // reduce sum the gradient
            if (reduced.shape() != m_variable->shape()) {
                std::cout << "Error: AccumulateGrad: reduced gradient shape does not match variable shape" << std::endl;
                return;
            }
            m_variable->set_gradient(m_variable->gradient_tensor() + reduced);
        } else {
            m_variable->set_gradient(m_variable->gradient_tensor() + x);
        }
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
        if (this->requires_grad(0)) {
            Tensor<T> grad_a = gradient * b;
            this->next_evaluate(0, grad_a);
        }

        if (this->requires_grad(1)) {
            Tensor<T> grad_b = gradient * a;
            this->next_evaluate(1, grad_b);
        }
    }

    void print() const override {
        std::cout << "MulBack" << std::endl;
    }
};

// tensor multiplication with single value tensor
template <typename T>
class MulBackScalar : public Operator<T> {
public:
    MulBackScalar(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}

    ~MulBackScalar() = default;

    void evaluate(const Tensor<T>& gradient) const override {
        const Tensor<T>& a = this->m_saved_values[0];
        const Tensor<T>& b = this->m_saved_values[1];

        if (this->requires_grad(0)) {
            Tensor<T> grad_a = gradient * b;
            this->next_evaluate(0, grad_a);
        }

        if (this->requires_grad(1)) {
            Tensor<T> grad_b = (gradient * a).reduce_sum(0);
            this->next_evaluate(1, grad_b);
        }
    }

    void print() const override {
        std::cout << "MulBackScalar" << std::endl;
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

        if (this->requires_grad(0)) {
            Tensor<T> grad_a = gradient / b;
            this->next_evaluate(0, grad_a);
        }

        if (this->requires_grad(1)) {
            Tensor<T> grad_b = -gradient * a / (b * b);
            this->next_evaluate(1, grad_b);
        }
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
        if (this->requires_grad(0)) {
            Tensor<T> grad_a = gradient * b * pow(a, b - 1);
            this->next_evaluate(0, grad_a);
        }
        if (this->requires_grad(1)) {
            Tensor<T> grad_b = gradient * pow(a, b) * log(a);
            this->next_evaluate(1, grad_b);
        }
    }

    void print() const override {
        std::cout << "PowBack" << std::endl;
    }
};

// tensor power with single value tensor
template <typename T>
class PowBackScalar : public Operator<T> {
public:
    PowBackScalar(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}

    ~PowBackScalar() = default;

    void evaluate(const Tensor<T>& gradient) const override {
        const Tensor<T>& a = this->m_saved_values[0];
        const Tensor<T>& b = this->m_saved_values[1];
        if (this->requires_grad(0)) {
            Tensor<T> grad_a = gradient * b * pow(a, b - 1);
            this->next_evaluate(0, grad_a);
        }
        if (this->requires_grad(1)) {
            Tensor<T> grad_b = (gradient * pow(a, b) * log(a)).reduce_sum(0);
            this->next_evaluate(1, grad_b);
        }
    }

    void print() const override {
        std::cout << "PowBackScalar" << std::endl;
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
        if (!this->requires_grad(0)) {
            return;
        }
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = gradient / (Tensor<T>(2.f) * (a.sqrt()));

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
        if (!this->requires_grad(0)) {
            return;
        }
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
        if (!this->requires_grad(0)) {
            return;
        }
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
        if (!this->requires_grad(0)) {
            return;
        }
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        std::function<T(T)> f = [](T x) { return x > 0 ? 1 : -1; };
        // calculate the gradients
        Tensor<T> grad_a = gradient * (a.map(f));

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
        if (!this->requires_grad(0)) {
            return;
        }

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
        if (!this->requires_grad(0)) {
            return;
        }
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
        if (!this->requires_grad(0)) {
            return;
        }
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        // calculate the gradients
        Tensor<T> grad_a = gradient / pow(a.cos(), Tensor<T>(2.f));

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "TanBack" << std::endl;
    }
};

// matmul
template <typename T>
class MatmulBack : public Operator<T> {
public:
    MatmulBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~MatmulBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];
        const Tensor<T>& b = saved_values[1];

        // calculate the gradients
        if (this->requires_grad(0)) {
            Tensor<T> grad_a = gradient.matmul(b.transpose());
            this->next_evaluate(0, grad_a);
        }
        if (this->requires_grad(1)) {
            Tensor<T> grad_b = a.transpose().matmul(gradient);
            this->next_evaluate(1, grad_b);
        }
    }

    void print() const override {
        std::cout << "MatmulBack" << std::endl;
    }
};

// sum back
template <typename T>
class SumBack : public Operator<T> {
public:
    SumBack(const std::vector<std::shared_ptr<Operator<T>>>& next_operators, const std::vector<int>& shape) 
        : Operator<T>(std::vector<Tensor<T>>(), next_operators), m_shape(shape) { }
    ~SumBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        if (!this->requires_grad(0)) {
            return;
        }
        // create a tensor with the same shape as the input
        Tensor<T> grad(m_shape, gradient.value());

        // evaluate the next operators
        this->next_evaluate(0, grad);
    }

    void print() const override {
        std::cout << "SumBack" << std::endl;
    }
private:
    std::vector<int> m_shape;
};

// ACTIVATION FUNCTIONS

// relu
template <typename T>
class ReluBack : public Operator<T> {
public:
    ReluBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~ReluBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        if (!this->requires_grad(0)) {
            return;
        }
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        std::function<T(T)> f = [](T x) { return x > 0 ? 1 : 0; };
        // calculate the gradients
        Tensor<T> grad_a = gradient * (a.map(f));

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "ReluBack" << std::endl;
    }
};

// sigmoid
template <typename T>
class SigmoidBack : public Operator<T> {
public:
    SigmoidBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~SigmoidBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        if (!this->requires_grad(0)) {
            return;
        }
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];
        
        std::function<T(T)> f = [](T x) { return 1 - x; };
        // calculate the gradients
        // grad_a = gradient * a.sigmoid() * (1 - a.sigmoid())
        Tensor<T> grad_a = gradient * a.sigmoid() * (a.sigmoid().map(f));

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "SigmoidBack" << std::endl;
    }
};

// softmax along the last dimension
template <typename T>
class SoftmaxBack : public Operator<T> {
public:
    SoftmaxBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~SoftmaxBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        if (!this->requires_grad(0)) {
            return;
        }
        auto& saved_values = this->m_saved_values;
        const Tensor<T>& a = saved_values[0];

        // Compute softmax of the input
        Tensor<T> softmax_a = a.softmax();

        // Compute gradient * softmax (element-wise)
        Tensor<T> s_times_grad = gradient * softmax_a;

        // // Sum along the last dimension (softmax axis) keeping dimensions for broadcasting
        // Tensor<T> sum_s_times_grad = s_times_grad.reduce_sum(-1, /*keepdims=*/true);

        // Compute final gradient: softmax * (gradient - sum(gradient * softmax))
        Tensor<T> grad_a = softmax_a + s_times_grad;

        // Propagate gradient to previous operation
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "SoftmaxBack" << std::endl;
    }
};

// tanh
template <typename T>
class TanhBack : public Operator<T> {
public:
    TanhBack(const std::vector<Tensor<T>>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators) 
        : Operator<T>(saved_values, next_operators) { }
    ~TanhBack() { }

    void evaluate(const Tensor<T>& gradient) const override {
        if (!this->requires_grad(0)) {
            return;
        }
        auto& saved_values = this->m_saved_values;
        // get the saved values
        const Tensor<T>& a = saved_values[0];

        std::function<T(T)> f = [](T x) { return 1 - x * x; };

        // calculate the gradients
        Tensor<T> tanh_a = a.tanh();
        // grad_a = gradient * a.tanh() * (1 - a.tanh()^2)
        Tensor<T> grad_a = gradient * tanh_a * tanh_a.map(f);

        // evaluate the next operators
        this->next_evaluate(0, grad_a);
    }

    void print() const override {
        std::cout << "TanhBack" << std::endl;
    }
};

} // namespace sdlm