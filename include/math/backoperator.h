#pragma once 

#include "math/number.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <memory>

namespace sdlm {

template <typename T>
class Number;

template <typename T>
class Operator {
public:
    virtual void evaluate(const T& x) const = 0;
    virtual void print() const = 0;
    virtual ~Operator() { }

    void debug_evalutate(const T& x) const {
        // check if member variables are set
        std::cout << "m_saved_values.size(): " << m_saved_values.size() << std::endl;
        std::cout << "m_next_operators.size(): " << m_next_operators.size() << std::endl;
        evaluate(x);
    }

    void next_evaluate(const int i, const T& x) const {
        if (this != nullptr && i >= m_next_operators.size()) {
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

    void visualize() const {
        std::cout << "Operator ";
        print();
        for (int i = 0; i < m_saved_values.size(); i++) {
            std::cout << "  " << m_saved_values[i];
        }
        std::cout << "--" << std::endl;
        for (const std::shared_ptr<Operator<T>>& next_operator : m_next_operators) {
            // std::cout << " -> ";
            if (next_operator != nullptr) {
                next_operator->visualize();
            } else {
                std::cout << "nullptr" << std::endl;
            }
        }
    }

protected:
    std::vector<T> m_saved_values;
    std::vector<std::shared_ptr<Operator<T>>> m_next_operators;

    Operator(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : m_saved_values(saved_values), m_next_operators(next_operators) {
            // std::cout << "Operator constructor: " << this << std::endl;
        }

    // Operator(const Operator<T>& other) : m_saved_values(other.m_saved_values), m_next_operators(other.m_next_operators) {}
};

template <typename T>
class AccumulateGrad : public Operator<T> {
private:
    Number<T>* m_variable;

public:
    AccumulateGrad(Number<T>* variable)
        : Operator<T>(std::vector<T>(), std::vector<std::shared_ptr<Operator<T>>>()), m_variable(variable) {}
    ~AccumulateGrad() { }


    void evaluate(const T& x) const override {
        m_variable->set_gradient(m_variable->gradient() + x);
    }

    void print() const override {
        std::cout << "AccumulateGrad" << std::endl;
    }
};

template <typename T>
class MulBack : public Operator<T> {
public:
    MulBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~MulBack() {}


    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        auto n0 = saved_values[0];
        auto n1 = saved_values[1];

        this->next_evaluate(0, n1 * x);
        this->next_evaluate(1, n0 * x);
    }

    void print() const override {
        std::cout << "MulBack" << std::endl;
    }
};

template <typename T>
class DivBack : public Operator<T> {
public:
    DivBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~DivBack() {}


    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        auto n0 = saved_values[0]; // numerator
        auto n1 = saved_values[1]; // denominator

        this->next_evaluate(0, x / n1);
        this->next_evaluate(1, -x * n0 / (n1 * n1));
    }

    void print() const override {
        std::cout << "DivBack" << std::endl;
    }
};
template <typename T>
class AddBack: public Operator<T> { 
public:
    // AddBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
    //     : Operator<T>(saved_values, next_operators) {}
    // without saved_values
    AddBack(const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(std::vector<T>(), next_operators) {}
    ~AddBack() {}

    void evaluate(const T& x) const override {
        this->next_evaluate(0, x);
        this->next_evaluate(1, x);
    }

    void print() const override {
        std::cout << "AddBack" << std::endl;
    }

};

template <typename T>
class SubBack: public Operator<T> { 
public:
    // SubBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
    //     : Operator<T>(saved_values, next_operators) {}
    // without saved_values
    SubBack(const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(std::vector<T>(), next_operators) {}
    ~SubBack() {}

    void evaluate(const T& x) const override {
        this->next_evaluate(0, x);
        this->next_evaluate(1, -x);
    }

    void print() const override {
        std::cout << "SubBack" << std::endl;
    }

};
template <typename T>
class NegBack: public Operator<T> { 
public:
    // NegBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
    //     : Operator<T>(saved_values, next_operators) {}
    // without saved_values
    NegBack(const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(std::vector<T>(), next_operators) {}
    ~NegBack() {}

    void evaluate(const T& x) const override {
        this->next_evaluate(0, -x);
    }

    void print() const override {
        std::cout << "NegBack" << std::endl;
    }

};

template <typename T>
class PowBack: public Operator<T> { 
public:
    PowBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~PowBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values; 

        auto n0 = saved_values[0]; // base
        auto n1 = saved_values[1]; // exponent

        this->next_evaluate(0, x * n1 * std::pow(n0, n1 - 1));
        this->next_evaluate(1, x * std::pow(n0, n1) * std::log(n0));
    }

    void print() const override {
        std::cout << "PowBack" << std::endl;
    }

};

template <typename T>
class ExpBack: public Operator<T> { 
public:
    ExpBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~ExpBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        this->next_evaluate(0, x * std::exp(saved_values[0]));
    }

    void print() const override {
        std::cout << "ExpBack" << std::endl;
    }

};
template <typename T>
class LogBack: public Operator<T> { 
public:
    LogBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~LogBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        this->next_evaluate(0, x / saved_values[0]);
    }

    void print() const override {
        std::cout << "LogBack" << std::endl;
    }

};

template <typename T>
class SqrtBack: public Operator<T> { 
public:
    SqrtBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~SqrtBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        this->next_evaluate(0, x / (2 * std::sqrt(saved_values[0])));
    }

    void print() const override {
        std::cout << "SqrtBack" << std::endl;
    }

};
template <typename T>
class SinBack: public Operator<T> { 
public:
    SinBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~SinBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        this->next_evaluate(0, x * std::cos(saved_values[0]));
    }

    void print() const override {
        std::cout << "SinBack" << std::endl;
    }

};

template <typename T>
class CosBack: public Operator<T> { 
public:
    CosBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~CosBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        this->next_evaluate(0, -x * std::sin(saved_values[0]));
    }

    void print() const override {
        std::cout << "CosBack" << std::endl;
    }

};

template <typename T>
class TanBack: public Operator<T> { 
public:
    TanBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~TanBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        this->next_evaluate(0, x / (std::cos(saved_values[0]) * std::cos(saved_values[0])));
    }

    void print() const override {
        std::cout << "TanBack" << std::endl;
    }

};
template <typename T>
class AbsBack: public Operator<T> { 
public:
    AbsBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~AbsBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values;

        this->next_evaluate(0, x * (saved_values[0] > 0 ? 1 : -1));
    }

    void print() const override {
        std::cout << "AbsBack" << std::endl;
    }

};

template <typename T>
class ReluBack: public Operator<T> { 
public:
    ReluBack(const std::vector<T>& saved_values, const std::vector<std::shared_ptr<Operator<T>>>& next_operators)
        : Operator<T>(saved_values, next_operators) {}
    ~ReluBack() {}

    void evaluate(const T& x) const override {
        auto& saved_values = this->m_saved_values; 

        this->next_evaluate(0, x * (saved_values[0] > 0 ? 1 : 0));
    }

    void print() const override {
        std::cout << "ReluBack" << std::endl;
    }

};



} // namespace sdlm

