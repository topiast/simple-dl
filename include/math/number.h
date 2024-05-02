#pragma once

#include "math/backoperator.h"

#include <cmath>
#include <iostream>
#include <random>
#include <limits>
#include <typeinfo>
#include <vector>

namespace sdlm {

template <typename T>
class Operator;

template <typename T>
class Number {
private:
    Operator<T>* grad_fn = nullptr;
    bool count_gradient = true;
    bool is_leaf = true;
    T m_value;
    T m_gradient;

    // Operator<T>* set_up_operator(const Number<T>& x) {

    //     std::vector<Number<T>*> saved_numbers = {this, &x};
    //     std::vector<Operator<T>*> next_operators = {grad_fn, x.grad_fn};

    //     return new Operator<T>(saved_numbers, next_operators);
    // }

public:
    Number() : m_value(0), m_gradient(0) {
        // grad_fn = new AccumulateGrad<T>({this});
    }
    // Number(const T& value) : m_value(value) {}
    Number(const T& value, const T& gradient = 0) : m_value(value), m_gradient(gradient) {
        // grad_fn = new AccumulateGrad<T>({this});
    }

    Number(const T& value, bool count_gradient, bool is_leaf) : m_value(value), count_gradient(count_gradient), is_leaf(is_leaf) {}

    ~Number() {
        // delete grad_fn;
    }

    static Number<T> max() {
        return Number<T>(std::numeric_limits<T>::max());
    }

    static Number<T> min() {
        return Number<T>(std::numeric_limits<T>::min());
    }

    static Number<T> epsilon() {
        return Number<T>(std::numeric_limits<T>::epsilon());
    }

    void randomize(const T& mean, const T& std) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(mean, std);
        m_value = dis(gen);
    }

    T value() const { return m_value; }

    // set count_gradient to true to count gradient
    void set_count_gradient(bool count_gradient) {
        this->count_gradient = count_gradient;
    }

    void set_gradient(const T& gradient) {
        m_gradient = gradient;
    }

    void set_leaf(bool is_leaf) {
        this->is_leaf = is_leaf;
    }

    void set_grad_fn(Operator<T>* grad_fn) {
        if (this->grad_fn != nullptr) {
            delete this->grad_fn;
        }
        this->grad_fn = grad_fn;
    }

    T gradient() const {
        return m_gradient;
    }

    Number<T> operator+(Number<T>& rhs) {
        Number<T> result(m_value + rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        AddBack<T>* op = new AddBack<T>(next_operators);
        result.set_grad_fn(op);
        
        return result;
    }

    Number<T> operator+(Number<T>&& rhs) {
        Number<T> result(m_value + rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        AddBack<T>* op = new AddBack<T>(next_operators);
        result.set_grad_fn(op);
        
        return result;
    }

    // OPERATOR * 
    Number<T> operator*(Number<T>& rhs) {
        Number<T> result(m_value * rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        MulBack<T>* op = new MulBack<T>(saved_values, next_operators);
        result.set_grad_fn(op);

        return result;
    }
    Number<T> operator*(Number<T>&& rhs) {
        Number<T> result(m_value * rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }
        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        MulBack<T>* op = new MulBack<T>(saved_values, next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // OPERATOR *

    // OPERATOR /
    Number<T> operator/(Number<T>& rhs) {
        Number<T> result(m_value / rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        DivBack<T>* op = new DivBack<T>(saved_values, next_operators);
        result.set_grad_fn(op);

        return result;
    }

    Number<T> operator/(Number<T>&& rhs) {
        Number<T> result(m_value / rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        DivBack<T>* op = new DivBack<T>(saved_values, next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // OPERATOR /

    // OPERATOR -
    Number<T> operator-(Number<T>& rhs) {
        Number<T> result(m_value - rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        SubBack<T>* op = new SubBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }

    Number<T> operator-(Number<T>&& rhs) {
        Number<T> result(m_value - rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        SubBack<T>* op = new SubBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // OPERATOR -

    // OPERATOR unary -
    Number<T> operator-() {
        Number<T> result(-m_value);

        if (count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn};

        NegBack<T>* op = new NegBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // OPERATOR unary -

    // POWER
    Number<T> pow(const Number<T>& rhs) {
        Number<T> result(std::pow(m_value, rhs.m_value));

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        PowBack<T>* op = new PowBack<T>(saved_values, next_operators);
        result.set_grad_fn(op);

        return result;
    }

    Number<T> pow(const Number<T>&& rhs) {
        Number<T> result(std::pow(m_value, rhs.m_value));

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        if (rhs.is_leaf && rhs.grad_fn == nullptr) {
            rhs.set_grad_fn(new AccumulateGrad<T>(&rhs));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn, rhs.grad_fn};

        PowBack<T>* op = new PowBack<T>(saved_values, next_operators);
        result.set_grad_fn(op);

        return result;
    }

    Number<T> pow(const T& rhs) {
        Number<T> result(std::pow(m_value, rhs));

        if (count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs};

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn};

        PowBack<T>* op = new PowBack<T>(saved_values, next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // POWER

    // EXP
    Number<T> exp() {
        Number<T> result(std::exp(m_value));

        if (count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn};

        ExpBack<T>* op = new ExpBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // EXP

    // LOG
    Number<T> log() {
        Number<T> result(std::log(m_value));

        if (count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn};

        LogBack<T>* op = new LogBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // LOG

    // SQRT
    Number<T> sqrt() {
        Number<T> result(std::sqrt(m_value));

        if (count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn};

        SqrtBack<T>* op = new SqrtBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // SQRT

    // SIN
    Number<T> sin() {
        Number<T> result(std::sin(m_value));

        if (count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn};

        SinBack<T>* op = new SinBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // SIN

    // COS
    Number<T> cos() {
        Number<T> result(std::cos(m_value));

        if (count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn};

        CosBack<T>* op = new CosBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // COS

    // TAN
    Number<T> tan() {
        Number<T> result(std::tan(m_value));

        if (count_gradient){
            result.set_leaf(false);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && grad_fn == nullptr) {
            set_grad_fn(new AccumulateGrad<T>(this));
        }
        std::vector<Operator<T>*> next_operators = {grad_fn};

        TanBack<T>* op = new TanBack<T>(next_operators);
        result.set_grad_fn(op);

        return result;
    }
    // TAN
    





    void debug_print() {
        std::cout << "Value: " << m_value << std::endl;
        std::cout << "Gradient: " << m_gradient << std::endl;
        std::cout << "Is leaf: " << is_leaf << std::endl;
        std::cout << "Count gradient: " << count_gradient << std::endl;
        std::cout << "Grad fn: ";
        if (grad_fn != nullptr) {
            grad_fn->print();
        } else {
            std::cout << "None" << std::endl;
        }
    }

    Number<T>& operator=(const Number<T>& rhs) {
        m_value = rhs.m_value;
        m_gradient = rhs.m_gradient;
        grad_fn = rhs.grad_fn->clone();
        count_gradient = rhs.count_gradient;
        is_leaf = rhs.is_leaf;

        return *this;
    }

    Number<T> operator=(const T& rhs) {
        m_value = rhs;
        m_gradient = 0;
        grad_fn = nullptr;
        count_gradient = true;
        is_leaf = true;

        return *this;
    }

    void backward(T gradient = 1) {
        if (grad_fn == nullptr) {
            return;
        }
        m_gradient = gradient;


        grad_fn->evaluate(m_gradient);
    }
};

} // namespace sdlm