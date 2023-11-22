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

    Number operator+(const Number& rhs) const { return Number(*this) += rhs; }
    Number operator-(const Number& rhs) const { return Number(*this) -= rhs; }
    Number operator*(const Number& rhs) const { return Number(*this) *= rhs; }
    Number operator/(const Number& rhs) const { return Number(*this) /= rhs; }
    Number operator%(const Number& rhs) const { return Number(*this) %= rhs; }
    Number operator&(const Number& rhs) const { return Number(*this) &= rhs; }
    Number operator|(const Number& rhs) const { return Number(*this) |= rhs; }
    Number operator^(const Number& rhs) const { return Number(*this) ^= rhs; }
    Number operator<<(const Number& rhs) const { return Number(*this) <<= rhs; }
    Number operator>>(const Number& rhs) const { return Number(*this) >>= rhs; }


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
    return Number<T>(std::abs(x.value()));
}

template <typename T>
Number<T> pow(const Number<T>& x, const Number<T>& y) {
    return Number<T>(std::pow(x.value(), y.value()));
}

template <typename T>
Number<T> sqrt(const Number<T>& x) {
    return Number<T>(std::sqrt(x.value()));
}

template <typename T>
Number<T> exp(const Number<T>& x) {
    return Number<T>(std::exp(x.value()));
}

template <typename T>
Number<T> log(const Number<T>& x) {
    return Number<T>(std::log(x.value()));
}

template <typename T>
Number<T> log10(const Number<T>& x) {
    return Number<T>(std::log10(x.value()));
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
} // namespace std