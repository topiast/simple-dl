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
#include <memory>

class Number {
private:
    std::shared_ptr<Operator<T>> m_grad_fn = std::shared_ptr<Operator<T>>(nullptr);
    bool count_gradient = false;
    bool is_leaf = true;
    T m_value;
    T m_gradient;

    // Operator<T>* set_up_operator(Number<T>& x) {

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

    Number(const T& value, bool count_gradient, bool is_leaf) : m_value(value), count_gradient(count_gradient), is_leaf(is_leaf), m_gradient(0) {}
    Number(const T& value, bool is_leaf) : m_value(value), count_gradient(true), is_leaf(is_leaf), m_gradient(0) {}

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

    T& no_grad() { return m_value; }

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

    void set_grad_fn(std::shared_ptr<Operator<T>> grad_fn) {
        // check if grad_fn is type of AccumulateGrad
        if (typeid(this->m_grad_fn) == typeid(AccumulateGrad<T>)) {
            std::cout << "AccumulateGrad is dereferenced" << std::endl;
            if (typeid(grad_fn) == typeid(AccumulateGrad<T>)) {
                std::cout << "and AccumulateGrad is assigned" << std::endl;
            }
        }
        this->m_grad_fn = grad_fn;
    }

    std::shared_ptr<Operator<T>> get_grad_fn() const {
        return m_grad_fn;
    }

    T gradient() const {
        return m_gradient;
    }

    // OPERATOR +
    Number<T> operator+(Number<T>& rhs) {
        Number<T> result(m_value + rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};

            result.set_grad_fn(std::make_shared<AddBack<T>>(next_operators));
        }
        
        return result;
    }

    Number<T> operator+(Number<T>&& rhs) {
        return *this + rhs;
    }
    // OPERATOR +

    // OPERATOR * 
    Number<T> operator*(Number<T>& rhs) {
        Number<T> result(m_value * rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};

            result.set_grad_fn(std::make_shared<MulBack<T>>(saved_values, next_operators));
        }

        return result;
    }
    Number<T> operator*(Number<T>&& rhs) {
        return *this * rhs;
    }
    // OPERATOR *

    // OPERATOR /
    Number<T> operator/(Number<T>& rhs) {
        Number<T> result(m_value / rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};

            result.set_grad_fn(std::make_shared<DivBack<T>>(saved_values, next_operators));
        }

        return result;
    }

    Number<T> operator/(Number<T>&& rhs) {
        return *this / rhs;
    }
    // OPERATOR /

    // OPERATOR -
    Number<T> operator-(Number<T>& rhs) {
        Number<T> result(m_value - rhs.m_value);

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};

            result.set_grad_fn(std::make_shared<SubBack<T>>(next_operators));
        }

        return result;
    }

    Number<T> operator-(Number<T>&& rhs) {
        return *this - rhs;
    }
    // OPERATOR -

    // OPERATOR unary -
    Number<T> operator-() {
        Number<T> result(-m_value);

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            result.set_grad_fn(std::make_shared<NegBack<T>>(next_operators));
        }

        return result;
    }
    // OPERATOR unary -

    // OPERATOR ++
    Number<T>& operator++() {
        Number<T> result(m_value + 1);

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }

if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            result.set_grad_fn(std::make_shared<AddBack<T>>(next_operators));
}

        return result;
    }
    // OPERATOR ++

    // OPERATOR --
    Number<T>& operator--() {
        Number<T> result(m_value - 1);

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }

        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            result.set_grad_fn(std::make_shared<SubBack<T>>(next_operators));
        }

        return result;
    }
    // OPERATOR --
    Number<T> pow(Number<T>& rhs) {
        Number<T> result(std::pow(m_value, rhs.m_value));

        if (count_gradient || rhs.count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};
            std::vector<T> saved_values = {m_value, rhs.m_value};

            result.set_grad_fn(std::make_shared<PowBack<T>>(saved_values, next_operators));
        }

        return result;
    }

    Number<T> pow(Number<T>&& rhs) {
        return pow(rhs);
    }

    Number<T> pow(const T& rhs) {
        Number<T> result(std::pow(m_value, rhs));

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};
            std::vector<T> saved_values = {m_value, rhs};

            result.set_grad_fn(std::make_shared<PowBack<T>>(saved_values, next_operators));
        }

        return result;
    }
    // POWER

    // EXP
    Number<T> exp() {
        Number<T> result(std::exp(m_value));

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            std::vector<T> saved_values = {m_value};
            result.set_grad_fn(std::make_shared<ExpBack<T>>(saved_values, next_operators));
        }


        return result;
    }
    // EXP

    // LOG
    Number<T> log() {
        Number<T> result(std::log(m_value));

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            std::vector<T> saved_values = {m_value};
            result.set_grad_fn(std::make_shared<LogBack<T>>(saved_values, next_operators));
        }


        return result;
    }
    // LOG

    // SQRT
    Number<T> sqrt() {
        Number<T> result(std::sqrt(m_value));

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            std::vector<T> saved_values = {m_value};
            result.set_grad_fn(std::make_shared<SqrtBack<T>>(saved_values, next_operators));
        }


        return result;
    }
    // SQRT

    // SIN
    Number<T> sin() {
        Number<T> result(std::sin(m_value));

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            std::vector<T> saved_values = {m_value};
            result.set_grad_fn(std::make_shared<SinBack<T>>(saved_values, next_operators));
        }


        return result;
    }
    // SIN

    // COS
    Number<T> cos() {
        Number<T> result(std::cos(m_value));

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            std::vector<T> saved_values = {m_value};
            result.set_grad_fn(std::make_shared<CosBack<T>>(saved_values, next_operators));
        }


        return result;
    }
    // COS

    // TAN
    Number<T> tan() {
        Number<T> result(std::tan(m_value));

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            std::vector<T> saved_values = {m_value};
            result.set_grad_fn(std::make_shared<TanBack<T>>(saved_values, next_operators));
        }


        return result;
    }
    // TAN

    // ABS
    Number<T> abs() {
        Number<T> result(std::abs(m_value));

        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};

            std::vector<T> saved_values = {m_value};
            result.set_grad_fn(std::make_shared<AbsBack<T>>(saved_values, next_operators));
        }


        return result;
    }
    // ABS

    // OPERATOR ==
    bool operator==(const Number<T>& rhs) {
        return m_value == rhs.m_value;
    }
    // OPERATOR ==

    // OPERATOR !=
    bool operator!=(const Number<T>& rhs) {
        return m_value != rhs.m_value;
    }
    // OPERATOR !=

    // OPERATOR <
    bool operator<(const Number<T>& rhs) {
        return m_value < rhs.m_value;
    }
    // OPERATOR <

    // OPERATOR <=
    bool operator<=(const Number<T>& rhs) {
        return m_value <= rhs.m_value;
    }
    // OPERATOR <=

    // OPERATOR >
    bool operator>(const Number<T>& rhs) {
        return m_value > rhs.m_value;
    }
    // OPERATOR >

    // OPERATOR >=
    bool operator>=(const Number<T>& rhs) {
        return m_value >= rhs.m_value;
    }
    // OPERATOR >=

    // OPERATOR +=
    Number<T>& operator+=(Number<T>& rhs) {
        m_value += rhs.m_value;

        if (count_gradient || rhs.count_gradient){
            set_leaf(false);
            set_count_gradient(true);
        } else {
            set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};

            set_grad_fn(std::make_shared<AddBack<T>>(next_operators));
        }

        return *this;
    }
    
    Number<T>& operator+=(Number<T>&& rhs) {
        return *this += rhs;
    }
    // OPERATOR +=

    // OPERATOR -=
    Number<T>& operator-=(Number<T>& rhs) {
        m_value -= rhs.m_value;

        if (count_gradient || rhs.count_gradient){
            set_leaf(false);
            set_count_gradient(true);
        } else {
            set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};

            set_grad_fn(std::make_shared<SubBack<T>>(next_operators));
        }

        return *this;
    }

    Number<T>& operator-=(Number<T>&& rhs) {
        return *this -= rhs;
    }

    // OPERATOR -=

    // OPERATOR *=
    Number<T>& operator*=(Number<T>& rhs) {
        m_value *= rhs.m_value;

        if (count_gradient || rhs.count_gradient){
            set_leaf(false);
            set_count_gradient(true);
        } else {
            set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};

            set_grad_fn(std::make_shared<MulBack<T>>(saved_values, next_operators));
        }

        return *this;
    }

    Number<T>& operator*=(Number<T>&& rhs) {
        return *this *= rhs;
    }
    // OPERATOR *=

    // OPERATOR /=
    Number<T>& operator/=(Number<T>& rhs) {
        m_value /= rhs.m_value;

        if (count_gradient || rhs.count_gradient){
            set_leaf(false);
            set_count_gradient(true);
        } else {
            set_count_gradient(false);
        }

        std::vector<T> saved_values = {m_value, rhs.m_value};

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        if (rhs.is_leaf && rhs.count_gradient && rhs.get_grad_fn().get() == nullptr) {
            rhs.set_grad_fn(std::make_shared<AccumulateGrad<T>>(&rhs));
        }
        if(count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn(), rhs.get_grad_fn()};

            set_grad_fn(std::make_shared<DivBack<T>>(saved_values, next_operators));
        }

        return *this;
    }

    Number<T>& operator/=(Number<T>&& rhs) {
        return *this /= rhs;
    }

    // OPERATOR /=

    // activation functions

    // relu
    Number<T> relu() {
        Number<T> result = *this;
        if (m_value < 0) {
            result = 0;
        }
        
        if (count_gradient){
            result.set_leaf(false);
            result.set_count_gradient(true);
        } else {
            result.set_count_gradient(false);
        }

        if (is_leaf && count_gradient && get_grad_fn().get() == nullptr) {
            result.set_grad_fn(std::make_shared<AccumulateGrad<T>>(this));
        }
        
        if(result.count_gradient){
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_fn()};
            std::vector<T> saved_values = {m_value};

            result.set_grad_fn(std::make_shared<ReluBack<T>>(saved_values, next_operators));
        }

        return result;
    }

    
    void debug_print() {
        std::cout << "Value: " << m_value << std::endl;
        std::cout << "Gradient: " << m_gradient << std::endl;
        std::cout << "Is leaf: " << is_leaf << std::endl;
        std::cout << "Count gradient: " << count_gradient << std::endl;
        std::cout << "Grad fn: ";
        if (m_grad_fn.get() != nullptr) {
            m_grad_fn->print();
        } else {
            std::cout << "None" << std::endl;
        }
    }

    Number<T>& operator=(const Number<T>& rhs) {
        m_value = rhs.m_value;
        m_gradient = rhs.m_gradient;
        set_grad_fn(rhs.m_grad_fn);
        // std::shared_ptr<Operator<T>> grad_fn = rhs.get_grad_fn();
        // if (grad_fn.get() != nullptr) {
        //     set_grad_fn(grad_fn);
        // } else {
        //     set_grad_fn(nullptr);
        // }
        count_gradient = rhs.count_gradient;
        is_leaf = rhs.is_leaf;

        return *this;
    }

    Number<T> operator=(const T& rhs) {
        m_value = rhs;
        m_gradient = 0;
        set_grad_fn(nullptr);
        count_gradient = false;
        is_leaf = true;

        return *this;
    }

    void backward(T gradient = 1) {
        if (m_grad_fn == nullptr) {
            return;
        }
        m_gradient = gradient;


        m_grad_fn->evaluate(m_gradient);
    }

    void zero_grad() {
        m_gradient = 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const Number<T>& number) {
        os << number.m_value;
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const Number<T>* number) {
        os << number->m_value;
        return os;
    }
};

template <typename T>
Number<T> abs(Number<T>& x) {
    Number<T> result = x.abs();
    return result;
}
template <typename T>
Number<T> abs(Number<T>&& x) {
    Number<T> result = x.abs();
    return result;
}

template <typename T>
Number<T> exp(Number<T>& x) {
    Number<T> result = x.exp();
    return result;
}
template <typename T>
Number<T> exp(Number<T>&& x) {
    Number<T> result = x.exp();
    return result;
}

template <typename T>
Number<T> log(Number<T>& x) {
    Number<T> result = x.log();
    return result;
}
template <typename T>
Number<T> log(Number<T>&& x) {
    Number<T> result = x.log();
    return result;
}

template <typename T>
Number<T> sqrt(Number<T>& x) {
    Number<T> result = x.sqrt();
    return result;
}
template <typename T>
Number<T> sqrt(Number<T>&& x) {
    Number<T> result = x.sqrt();
    return result;
}

template <typename T>
Number<T> sin(Number<T>& x) {
    Number<T> result = x.sin();
    return result;
}
template <typename T>
Number<T> sin(Number<T>&& x) {
    Number<T> result = x.sin();
    return result;
}

template <typename T>
Number<T> cos(Number<T>& x) {
    Number<T> result = x.cos();
    return result;
}
template <typename T>
Number<T> cos(Number<T>&& x) {
    Number<T> result = x.cos();
    return result;
}

template <typename T>
Number<T> tan(Number<T>& x) {
    Number<T> result = x.tan();
    return result;
}
template <typename T>
Number<T> tan(Number<T>&& x) {
    Number<T> result = x.tan();
    return result;
}

template <typename T>
Number<T> pow(Number<T>& x, Number<T>& y) {
    Number<T> result = x.pow(y);
    return result;
}

template <typename T>
Number<T> pow(Number<T>& x, T& y) {
    Number<T> result = x.pow(y);
    return result;
}
template <typename T>
Number<T> pow(Number<T>&& x, Number<T>&& y) {
    Number<T> result = x.pow(y);
    return result;
}
template <typename T>
Number<T> pow(Number<T>& x, Number<T>&& y) {
    Number<T> result = x.pow(y);
    return result;
}
template <typename T>
Number<T> pow(Number<T>&& x, Number<T>& y) {
    Number<T> result = x.pow(y);
    return result;
}

template <typename T>
Number<T> pow(Number<T>&& x, T&& y) {
    Number<T> result = x.pow(y);
    return result;
}
template <typename T>
Number<T> pow(Number<T>& x, T&& y) {
    Number<T> result = x.pow(y);
    return result;
}
template <typename T>
Number<T> pow(Number<T>&& x, T& y) {
    Number<T> result = x.pow(y);
    return result;
}

} // namespace sdlm

namespace std {
    template <typename T>
    sdlm::Number<T> abs(sdlm::Number<T>& x) {
        return sdlm::abs(x);
    }
    
    template <typename T>
    sdlm::Number<T> abs(sdlm::Number<T>&& x) {
        return sdlm::abs(x);
    }
    
    template <typename T>
    sdlm::Number<T> exp(sdlm::Number<T>& x) {
        return sdlm::exp(x);
    }

    template <typename T>
    sdlm::Number<T> exp(sdlm::Number<T>&& x) {
        return sdlm::exp(x);
    }

    template <typename T>
    sdlm::Number<T> log(sdlm::Number<T>& x) {
        return sdlm::log(x);
    }

    template <typename T>
    sdlm::Number<T> log(sdlm::Number<T>&& x) {
        return sdlm::log(x);
    }

    template <typename T>
    sdlm::Number<T> sqrt(sdlm::Number<T>& x) {
        return sdlm::sqrt(x);
    }

    template <typename T>
    sdlm::Number<T> sqrt(sdlm::Number<T>&& x) {
        return sdlm::sqrt(x);
    }

    template <typename T>
    sdlm::Number<T> sin(sdlm::Number<T>& x) {
        return sdlm::sin(x);
    }

    template <typename T>
    sdlm::Number<T> sin(sdlm::Number<T>&& x) {
        return sdlm::sin(x);
    }

    template <typename T>
    sdlm::Number<T> cos(sdlm::Number<T>& x) {
        return sdlm::cos(x);
    }

    template <typename T>
    sdlm::Number<T> cos(sdlm::Number<T>&& x) {
        return sdlm::cos(x);
    }

    template <typename T>
    sdlm::Number<T> tan(sdlm::Number<T>& x) {
        return sdlm::tan(x);
    }

    template <typename T>
    sdlm::Number<T> tan(sdlm::Number<T>&& x) {
        return sdlm::tan(x);
    }

    template <typename T>
    sdlm::Number<T> pow(sdlm::Number<T>& x, sdlm::Number<T>& y) {
        return sdlm::pow(x, y);
    }

    template <typename T>
    sdlm::Number<T> pow(sdlm::Number<T>&& x, sdlm::Number<T>&& y) {
        return sdlm::pow(x, std::move(y));
    }

    template <typename T>
    sdlm::Number<T> pow(sdlm::Number<T>& x, T& y) {
        return sdlm::pow(x, y);
    }

    template <typename T>
    sdlm::Number<T> pow(sdlm::Number<T>&& x, T&& y) {
        return sdlm::pow(x, std::move(y));
    }

} // namespace std