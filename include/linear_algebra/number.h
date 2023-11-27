#pragma once

#include <cmath>
#include <iostream>
// to get random id
#include <random>

namespace ln {

template <typename T>
class Number {
private:
    bool count_gradient = true;
    T m_value;
    T m_gradient;
    // random id to identify the number
    u_int16_t id;

public:
    Number() : m_value(0), m_gradient(0) { 
        // generate random id
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 65535);
        id = dis(gen);
        }
    // Number(const T& value) : m_value(value) {}
    Number(const T& value, const T& gradient = 0) : m_value(value), m_gradient(gradient) {
        // generate random id
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 65535);
        id = dis(gen);
    }

    T value() const { return m_value; }

    // set count_gradient to true to count gradient
    void set_count_gradient(bool count_gradient) {
        this->count_gradient = count_gradient;
    }

    void set_gradient(const T& gradient) {
        m_gradient = gradient;
    }

    T gradient() const {
        return m_gradient;
    }

    u_int16_t get_id() const {
        return id;
    }

    Number& operator+=(const Number& rhs) {
        if (count_gradient){
            m_gradient += rhs.m_gradient;
        }

        m_value += rhs.m_value;

        return *this;
    }

    Number& operator-=(const Number& rhs) {
        if (count_gradient){
            m_gradient -= rhs.m_gradient;
        }

        m_value -= rhs.m_value;

        return *this;
    }

    Number& operator*=(const Number& rhs) {
        if (count_gradient){
            m_gradient = m_gradient * rhs.m_value + m_value * rhs.m_gradient;
        }

        m_value *= rhs.m_value;


        return *this;
    }

    Number& operator/=(const Number& rhs) {
        if (count_gradient) {
            m_gradient = (m_gradient * rhs.m_value - m_value * rhs.m_gradient) / (rhs.m_value * rhs.m_value);
        }

        m_value /= rhs.m_value;

        return *this;
    }

    Number& abs() {
        if (count_gradient) {
            if (m_value < 0) {
                m_gradient = -m_gradient;
            } 
            
        }

        m_value = std::abs(m_value);

        return *this;
    }

    Number& pow(const Number& rhs) {
        if (count_gradient) {
            m_gradient = m_gradient * rhs.m_value * std::pow(m_value, rhs.m_value - 1); 
        }

        m_value = std::pow(m_value, rhs.m_value);

        return *this;
    }

    // this function is used to calculate gradient of x ^ y, but gradient is relative to y
    Number& pow_inv_for_grad(const Number& rhs) {
        // here x is m_value, y is rhs.m_value
        if (count_gradient) {
            m_gradient = std::pow(m_value, rhs.m_value) * std::log(rhs.m_value) * rhs.m_gradient; // x ^ y * ln(y) * y' ??? should be x ^ y * ln(x) * y' ???
        }

        m_value = std::pow(m_value, rhs.m_value);

        return *this;
    }

    Number& sqrt() {
        if (count_gradient) {
            m_gradient = m_gradient / (2 * std::sqrt(m_value));
        }

        m_value = std::sqrt(m_value);

        return *this;
    }

    Number& exp() {
        if (count_gradient) {
            m_gradient = m_gradient * std::exp(m_value);
        }

        m_value = std::exp(m_value);

        return *this;
    }

    Number& log() {
        if (count_gradient) {
            m_gradient = m_gradient / m_value;
        }

        m_value = std::log(m_value);

        return *this;
    }

    Number& log10() {
        if (count_gradient) {
            m_gradient = m_gradient / (m_value * std::log(10));
        }

        m_value = std::log10(m_value);

        return *this;
    }
    

    Number& operator%=(const Number& rhs) {
        m_value %= rhs.m_value;

        return *this;
    }

    Number& operator&=(const Number& rhs) {
        m_value &= rhs.m_value;
        return *this;
    }

    Number& operator|=(const Number& rhs) {
        m_value |= rhs.m_value;
        return *this;
    }

    Number& operator^=(const Number& rhs) {
        m_value ^= rhs.m_value;
        return *this;
    }

    Number& operator<<=(const Number& rhs) {
        m_value <<= rhs.m_value;
        return *this;
    }

    Number& operator>>=(const Number& rhs) {
        m_value >>= rhs.m_value;
        return *this;
    }

    Number& operator++() {
        ++m_value;
        return *this;
    }

    Number& operator--() {
        --m_value;
        return *this;
    }

    Number operator++(int) {
        Number tmp(*this);
        ++m_value;
        return tmp;
    }

    Number operator--(int) {
        Number tmp(*this);
        --m_value;
        return tmp;
    }

    Number operator+() const { return *this; }
    Number operator-() const { return Number(-m_value); }
    Number operator~() const { return Number(~m_value); }

    Number operator+(const Number& rhs) const { 
        Number result = Number(*this) += rhs;
        return result; 
        }
    Number operator-(const Number& rhs) const {
        Number result = Number(*this) -= rhs;
        return result; 
        }
    Number operator*(const Number& rhs) const {
        Number result = Number(*this) *= rhs;
        return result; 
        }
    Number operator/(const Number& rhs) const {
        Number result = Number(*this) /= rhs;
        return result; 
        }
    Number operator%(const Number& rhs) const {
        Number result = Number(*this) %= rhs;
        return result; 
        }
    Number operator&(const Number& rhs) const {
        Number result = Number(*this) &= rhs;
        return result; 
        }
    Number operator|(const Number& rhs) const {
        Number result = Number(*this) |= rhs;
        return result; 
        }
    Number operator^(const Number& rhs) const {
        Number result = Number(*this) ^= rhs;
        return result; 
        }
    Number operator<<(const Number& rhs) const {
        Number result = Number(*this) <<= rhs;
        return result; 
        }
    Number operator>>(const Number& rhs) const {
        Number result = Number(*this) >>= rhs;
        return result; 
        }

    // copy assignment
    Number& operator=(const Number& rhs) {
        // std::cout << "copy assignment" << std::endl;
        m_value = rhs.m_value;
        m_gradient = rhs.m_gradient;
        return *this;
    }



    // bool operator==(const Number& rhs) const { return m_value == rhs.m_value; }
    // bool operator!=(const Number& rhs) const { return m_value != rhs.m_value; }
    // bool operator<(const Number& rhs) const { return m_value < rhs.m_value; }
    // bool operator>(const Number& rhs) const { return m_value > rhs.m_value; }
    // bool operator<=(const Number& rhs) const { return m_value <= rhs.m_value; }
    // bool operator>=(const Number& rhs) const { return m_value >= rhs.m_value; }





    friend std::ostream& operator<<(std::ostream& os, const Number& number) {
        os << number.m_value;
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Number& number) {
        is >> number.m_value;
        return is;
    }


};


template <typename T>
Number<T> abs(const Number<T>& x) {
    // copy x
    Number<T> result = x;
    return result.abs();
}

template <typename T>
Number<T> pow(const Number<T>& x, const Number<T>& y) {
    Number<T> result = x;

    return y.gradient() == 0 ? result.pow(y) : result.pow_inv_for_grad(y) ;
}

template <typename T>
Number<T> sqrt(const Number<T>& x) {
    Number<T> result = x;
    return result.sqrt();
}

template <typename T>
Number<T> exp(const Number<T>& x) {
    Number<T> result = x;
    return result.exp();
}

template <typename T>
Number<T> log(const Number<T>& x) {
    Number<T> result = x;
    return result.log();
}

template <typename T>
Number<T> log10(const Number<T>& x) {
    Number<T> result = x;
    return result.log10();
}

} // namespace ln

// hash function for Number
namespace std {
    template <typename T>
    struct hash<ln::Number<T>> {
        std::size_t operator()(const ln::Number<T>& number) const {
            // since id is unique, we can use it as hash
            return number.get_id();
        }
    };
    template <typename T>
    struct equal_to<ln::Number<T>> {
        bool operator()(const ln::Number<T>& lhs, const ln::Number<T>& rhs) const {
            return lhs.get_id() == rhs.get_id();
        }
    };

    // pow function for Number
    template <typename T>
    ln::Number<T> pow(const ln::Number<T>& x, const ln::Number<T>& y) {
        return ln::pow(x, y);
    }

    // sqrt function for Number
    template <typename T>
    ln::Number<T> sqrt(const ln::Number<T>& x) {
        return ln::sqrt(x);
    }

    // exp function for Number
    template <typename T>
    ln::Number<T> exp(const ln::Number<T>& x) {
        return ln::exp(x);
    }

    // log function for Number
    template <typename T>
    ln::Number<T> log(const ln::Number<T>& x) {
        return ln::log(x);
    }

    // log10 function for Number
    template <typename T>
    ln::Number<T> log10(const ln::Number<T>& x) {
        return ln::log10(x);
    }


} // namespace std