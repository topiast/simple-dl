#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <functional>


namespace sdlm {

template <typename T>
class Tensor {
    private:
    std::vector<T> m_data;
    std::vector<int> m_shape;

    public:
    Tensor() {}

    Tensor(const std::vector<T>& data, const std::vector<int>& shape) : m_data(data), m_shape(shape) {}

    Tensor(const std::vector<int>& shape) : m_shape(shape) {
        int size = 1;
        for (int i = 0; i < shape.size(); i++) {
            size *= shape[i];
        }
        std::cout << "size in MB: " << size * sizeof(T) / 1000000 << std::endl;
        m_data = std::vector<T>(size);
    }

    Tensor& zeros(const std::vector<int>& shape) {
        m_shape = shape;
        int size = 1;
        for (int i = 0; i < shape.size(); i++) {
            size *= shape[i];
        }
        m_data = std::vector<T>(size, T(0));
        return *this;
    }

    Tensor& ones(const std::vector<int>& shape) {
        m_shape = shape;
        int size = 1;
        for (int i = 0; i < shape.size(); i++) {
            size *= shape[i];
        }
        m_data = std::vector<T>(size, T(1));
        return *this;
    }

    Tensor& random(const std::vector<int>& shape, float mean = 0.f, float stddev = 1.f) {
        m_shape = shape;
        int size = 1;
        for (int i = 0; i < shape.size(); i++) {
            size *= shape[i];
        }
        m_data = std::vector<T>(size, T(0));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> d(mean, stddev);

        for (int i = 0; i < size; i++) {
            float r = d(gen);
            
            m_data[i] = r;
        }

        return *this;
    }

    Tensor& fill(T value) {
        for (int i = 0; i < m_data.size(); i++) {
            m_data[i] = value;
        }
        return *this;
    }

    T& operator[](int index) {
        return m_data[index];
    }


    Tensor operator+(const Tensor& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Incompatible shapes for addition");
        }

        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] += rhs.m_data[i];
        }

        return result;
    }

    Tensor operator-(const Tensor& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Incompatible shapes for subtraction");
        }

        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] -= rhs.m_data[i];
        }

        return result;
    }

    Tensor row_add(const Tensor& rhs) const {
        if (m_shape[1] != rhs.m_shape[1]) {
            throw std::invalid_argument("Incompatible shapes for row addition");
        }

        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (int i = 0; i < m_shape[0]; ++i) {
            for (int j = 0; j < m_shape[1]; ++j) {
                result.m_data[i * m_shape[1] + j] += rhs.m_data[j];
            }
        }

        return result;
    }

    Tensor& operator+=(const Tensor& rhs) {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Incompatible shapes for addition");
        }

        for (size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] += rhs.m_data[i];
        }

        return *this;
    }

    Tensor& operator-=(const Tensor& rhs) {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Incompatible shapes for subtraction");
        }

        for (size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] -= rhs.m_data[i];
        }

        return *this;
    }

    Tensor operator*(const Tensor& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Incompatible shapes for element-wise multiplication");
        }

        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] *= rhs.m_data[i];
        }

        return result;
    }

    Tensor& operator*=(const Tensor& rhs) {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Incompatible shapes for element-wise multiplication");
        }

        for (size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] *= rhs.m_data[i];
        }

        return *this;
    }

    Tensor operator*(T rhs) const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] *= rhs;
        }

        return result;
    }

    Tensor& operator*=(T rhs) {
        for (size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] *= rhs;
        }

        return *this;
    }


    Tensor operator+(T rhs) const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] += rhs;
        }

        return result;
    }

    Tensor& operator+=(T rhs) const {
        for (size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] += rhs;
        }

        return *this;
    }

    Tensor operator-(T rhs) const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] -= rhs;
        }

        return result;
    }

    Tensor& operator-=(T rhs) {
        for (size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] -= rhs;
        }

        return *this;
    }

    Tensor operator/(T rhs) const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] /= rhs;
        }

        return result;
    }

    Tensor& operator/=(T rhs) {
        for (size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] /= rhs;
        }

        return *this;
    }

    Tensor operator-() const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] = -result.m_data[i];
        }

        return result;
    }


    Tensor matmul(const Tensor& rhs) const {
        if (m_shape.size() != 2 || rhs.m_shape.size() != 2) {
            throw std::invalid_argument("Incompatible shapes for matrix multiplication. Tensors must be 2D");
        }

        if (m_shape[1] != rhs.m_shape[0]) {
            std::string error = "Incompatible shapes for matrix multiplication. ";
            error += "First tensor has shape " + std::to_string(m_shape[0]) + "x" + std::to_string(m_shape[1]);
            error += " and second tensor has shape " + std::to_string(rhs.m_shape[0]) + "x" + std::to_string(rhs.m_shape[1]);
            throw std::invalid_argument(error);
        }

        Tensor result;
        result.m_shape = {m_shape[0], rhs.m_shape[1]};
        result.m_data = std::vector<T>(result.m_shape[0] * result.m_shape[1], T(0));

        for (int i = 0; i < m_shape[0]; ++i) {
            for (int j = 0; j < rhs.m_shape[1]; ++j) {
                for (int k = 0; k < m_shape[1]; ++k) {
                    result.m_data[i * result.m_shape[1] + j] += m_data[i * m_shape[1] + k] * rhs.m_data[k * rhs.m_shape[1] + j];
                }
            }
        }

        return result;
    }

    Tensor transpose() const {
        if (m_shape.size() != 2) {
            throw std::invalid_argument("Incompatible shape for transpose");
        }

        Tensor result;
        result.zeros({m_shape[1], m_shape[0]});

        for (size_t i = 0; i < m_shape[0]; ++i) {
            for (size_t j = 0; j < m_shape[1]; ++j) {
                result.m_data[j * m_shape[0] + i] = m_data[i * m_shape[1] + j];
            }
        }

        return result;
    }

    Tensor sum(int axis) const {
        if (axis >= m_shape.size()) {
            throw std::invalid_argument("Invalid axis");
        }

        Tensor result;
        result.m_shape = m_shape;
        result.m_shape.erase(result.m_shape.begin() + axis);
        result.m_data = std::vector<T>(result.m_shape[0] * result.m_shape[1], T(0));

        int stride = 1;
        for (int i = m_shape.size() - 1; i > axis; --i) {
            stride *= m_shape[i];
        }

        for (int i = 0; i < m_data.size(); ++i) {
            result.m_data[(i / stride) % result.m_shape[0]] += m_data[i];
        }

        return result;
    }

    T sum() const {
        T result = T(0);
        for (size_t i = 0; i < m_data.size(); ++i) {
            result += m_data[i];
        }
        return result;
    }

    Tensor pow(T exponent) const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] = std::pow(result.m_data[i], exponent);
        }

        return result;
    }

    Tensor exp() const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] = std::exp(result.m_data[i]);
        }

        return result;
    }

    Tensor log() const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] = std::log(result.m_data[i]);
        }

        return result;
    }

    Tensor clip(T min, T max) const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            if (result.m_data[i] < min) {
                result.m_data[i] = min;
            } else if (result.m_data[i] > max) {
                result.m_data[i] = max;
            }
        }

        return result;
    }

    T norm(T p) const {
        T result = T(0);
        for (size_t i = 0; i < m_data.size(); ++i) {
            result += std::pow(std::abs(m_data[i]), p);
        }
        return std::pow(result, T(1) / p);
    }

    Tensor one_hot(int num_classes, std::function<int(T)> cast) const {
        Tensor result;
        result.m_shape = {m_shape[0], num_classes};
        result.m_data = std::vector<T>(result.m_shape[0] * result.m_shape[1], T(0));

        for (int i = 0; i < m_shape[0]; ++i) {
            auto index = cast(m_data[i]);
            result.m_data[i * result.m_shape[1] + index] = T(1);
        }

        return result;
    }

    Tensor reshape(const std::vector<int>& shape) const {
        Tensor result;
        result.m_shape = shape;
        result.m_data = m_data;

        int size = 1;
        for (int i = 0; i < shape.size(); ++i) {
            size *= shape[i];
        }

        if (size != m_data.size()) {
            throw std::invalid_argument("Incompatible shapes for reshape");
        }

        return result;
    }

    // flattens the last two dimensions
    Tensor flatten() const {
        Tensor result;
        for (int i = 0; i < m_shape.size() - 2; ++i) {
            result.m_shape.push_back(m_shape[i]);
        }
        result.m_shape.push_back(m_shape[m_shape.size() - 2] * m_shape[m_shape.size() - 1]);
        
        // copy data each element at a time
        result.m_data = std::vector<T>(get_size());
        for (int i = 0; i < get_size(); ++i) {
            result.m_data[i] = m_data[i];
        }

        return result;
    }

    Tensor normalize(T min = 0, T max = 1) const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        T min_val = m_data[0];
        T max_val = m_data[0];
        for (size_t i = 0; i < m_data.size(); ++i) {
            if (m_data[i] < min_val) {
                min_val = m_data[i];
            }
            if (m_data[i] > max_val) {
                max_val = m_data[i];
            }
        }

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] = (m_data[i] - min_val) / (max_val - min_val) * (max - min) + min;
        }

        return result;
    }

    // Activation functions
    Tensor sigmoid() const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            // This is hacky, but it works for now
            result.m_data[i]++;
            result.m_data[i] = ((std::exp(-result.m_data[i]))).pow(-1);
        }

        return result;
    }

    Tensor softmax() const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); i += m_shape.back()) {
            T sum = 0;
            for (size_t j = 0; j < m_shape.back(); ++j) {
                sum += std::exp(result.m_data[i + j]);
            }
            for (size_t j = 0; j < m_shape.back(); ++j) {
                result.m_data[i + j] = std::exp(result.m_data[i + j]) / sum;
            }
        }

        return result;
    }

    Tensor relu() const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            // so that the gradient is correct
            result.m_data[i] = result.m_data[i].relu();
        }

        return result;
    }

    Tensor tanh() const {
        Tensor result;
        result.m_shape = m_shape;
        result.m_data = m_data;

        for (size_t i = 0; i < m_data.size(); ++i) {
            result.m_data[i] = (std::exp(result.m_data[i]) - std::exp(-result.m_data[i])) / (std::exp(result.m_data[i]) + std::exp(-result.m_data[i]));
        }

        return result;
    }

    // Activation functions

    const std::vector<int>& get_shape() const {
        return m_shape;

    };

    const size_t get_size() const {
        return m_data.size();
    }

    std::vector<T>& get_values() {
        return m_data;
    }

    void print_data() const {
        for (int i = 0; i < m_data.size(); i++) {
            std::cout << m_data[i] << " ";
        }
        std::cout << std::endl;
    }

    // pretty print tensor
    void print() const {
        if (m_shape.size() == 0) {
            std::cout << "[]";
            return;
        }

        int cum = 1;
        for (auto it = m_shape.end()-1; it > m_shape.begin(); it--){
            cum *= *it;
        }

        std::cout << "[";
        printTensorHelper(0, m_shape, m_data, cum);
        std::cout << "]" << std::endl;
    }

    void printTensorHelper(int index, const std::vector<int>& shape, const std::vector<T>& data, int index_cum) const {
        if (shape.size() == 1) {
            std::cout << "[";
            for (int i = 0; i < shape.back(); ++i) {
                std::cout << data[index + i];
                if (i != shape.back() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]";
        } else {
            for (int i = 0; i < shape.front(); ++i) {
                std::cout << std::endl;
                std::cout << "  ";
                printTensorHelper(index + i * index_cum, std::vector<int>(shape.begin() + 1, shape.end()), data, index_cum/shape[1]);
                if (i != shape.front() - 1) {
                    std::cout << ",";
                }
            }
            std::cout << std::endl;
        }
    }

    // print the shape of the tensor
    void print_shape() const {
        std::cout << "[";
        for (int i = 0; i < m_shape.size(); i++) {
            std::cout << m_shape[i];
            if (i != m_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }


    // Set values for tensors at specified indices
    void set_values(const std::vector<int>& indices, const std::vector<T>& values) {
        int index = 0;
        int stride = 1;
        for (int i = m_shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= m_shape[i] || indices[i] < 0) {
                throw std::out_of_range("Index out of bounds");
            }
            index += indices[i] * stride;
            stride *= m_shape[i];
        }

        if (index + values.size() > m_data.size()) {
            throw std::invalid_argument("Values must fit in tensor");
        }

        for (size_t i = 0; i < values.size(); ++i) {
            m_data[index + i] = values[i];
        }
    }

    // Overloaded set_values function for setting values at a single index
    void set_values(int index, const std::vector<T>& values) {
        set_values(std::vector<int>{index}, values);
    }

    // Overloaded set_values function for variable arguments
    template<typename... Args>
    void set_values(int index, Args... args) {
        set_values(std::vector<int>{index}, {static_cast<T>(args)...});
    }

    // Overloaded set_values function for multi-dimensional indices
    template<typename... Args>
    void set_values(const std::vector<int>& indices, Args... args) {
        set_values(indices, {static_cast<T>(args)...});
    }

    Tensor<T> get_values(const std::vector<int>& indices) const {
        std::vector<int> strides(m_shape.size());
        strides.back() = 1;
        for (int i = m_shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * m_shape[i + 1];
        }

        int index = 0;
        for (int i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides[i];
        }

        // get the new shape
        if (indices.size() == m_shape.size()) {
            return Tensor<T>(std::vector<T>{m_data[index]}, std::vector<int>{1});
        }
        std::vector<int> new_shape(m_shape.begin() + indices.size(), m_shape.end());

 
        Tensor<T> result;
        result.m_shape = new_shape;

        result.m_data = std::vector<T>(strides[0]);

        for (int i = 0; i < strides[0]; ++i) {
            result.m_data[i] = m_data[index + i];
        }

        return result;
    }






    
};

    
} // namespace sdlm