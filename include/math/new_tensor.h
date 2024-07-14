#pragma once

#include "math/new_backoperator.h"

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
class Tensor {
    private:
        std::vector<int> m_shape;
        std::vector<int> m_strides;
        std::vector<T> m_values;
        std::vector<T> m_gradients;

        std::shared_ptr<Operator<T>> m_grad_op = nullptr;
        // bool m_is_transposed = false;
        bool m_requires_grad = false;
        bool m_is_leaf = true;

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

    public:
        Tensor() = default;

        // Single value tensor
        Tensor(const T& value) {
            m_shape = {1};
            m_strides = {1};
            m_values = {value};
            m_gradients = {0};
        }

        Tensor(const std::vector<int>& shape) : m_shape(shape) {
            m_strides.resize(m_shape.size());
            m_strides[m_shape.size() - 1] = 1;
            for (int i = m_shape.size() - 2; i >= 0; i--) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }
            m_values.resize(m_strides[0] * m_shape[0]);
        }

        Tensor(const std::vector<int>& shape, const std::vector<T>& values) : m_shape(shape), m_values(values) {
            m_strides.resize(m_shape.size());
            m_strides[m_shape.size() - 1] = 1;
            for (int i = m_shape.size() - 2; i >= 0; i--) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }
        }

        Tensor(const std::vector<int>& shape, const T& value) : m_shape(shape) {
            m_strides.resize(m_shape.size());
            m_strides[m_shape.size() - 1] = 1;
            for (int i = m_shape.size() - 2; i >= 0; i--) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }
            m_values.resize(m_strides[0] * m_shape[0], value);
        }

        Tensor(const std::vector<int>& shape, const T& value, const T& gradient) : m_shape(shape) {
            m_strides.resize(m_shape.size());
            m_strides[m_shape.size() - 1] = 1;
            for (int i = m_shape.size() - 2; i >= 0; i--) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }
            m_values.resize(m_strides[0] * m_shape[0], value);
            m_gradients.resize(m_strides[0] * m_shape[0], gradient);
        }

        Tensor(const std::vector<int>& shape, const std::vector<T>& values, const std::vector<T>& gradients) 
            : m_shape(shape), m_values(values), m_gradients(gradients) {
            m_strides.resize(m_shape.size());
            m_strides[m_shape.size() - 1] = 1;
            for (int i = m_shape.size() - 2; i >= 0; i--) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }
        }

        // Define copy constructor

        // Copy constructor
        Tensor(const Tensor& other) 
            : m_shape(other.m_shape), 
              m_strides(other.m_strides), 
              m_values(other.m_values), 
              m_gradients(other.m_gradients), 
              m_requires_grad(other.m_requires_grad), 
              m_is_leaf(other.m_is_leaf) {
            if (other.m_grad_op) {
                m_grad_op = other.m_grad_op;
            }
        }

        Tensor no_grad() {
            return Tensor(m_shape, m_values);
        }


        void zeros(const std::vector<int>& shape) {
            m_shape = shape;
            m_strides.resize(m_shape.size());
            m_strides[m_shape.size() - 1] = 1;
            for (int i = m_shape.size() - 2; i >= 0; i--) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }
            m_values.resize(m_strides[0] * m_shape[0], 0);
            // m_gradients.resize(m_strides[0] * m_shape[0], 0);
        }

        void ones(const std::vector<int>& shape) {
            m_shape = shape;
            m_strides.resize(m_shape.size());
            m_strides[m_shape.size() - 1] = 1;
            for (int i = m_shape.size() - 2; i >= 0; i--) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }
            m_values.resize(m_strides[0] * m_shape[0], 1);
            // m_gradients.resize(m_strides[0] * m_shape[0], 0);
        }
        

        Tensor& random(const std::vector<int>& shape, float mean = 0.f, float stddev = 1.f) {
            m_shape = shape;
            int size = 1;
            for (int i = 0; i < shape.size(); i++) {
                size *= shape[i];
            }
            // strides
            m_strides.resize(m_shape.size());
            m_strides[m_shape.size() - 1] = 1;
            for (int i = m_shape.size() - 2; i >= 0; i--) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }

            m_values = std::vector<T>(size, T(0));

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> d(mean, stddev);

            for (int i = 0; i < size; i++) {
                float r = d(gen);
                
                m_values[i] = r;
            }

            return *this;
        }

        Tensor& fill(T value) {
            for (int i = 0; i < m_values.size(); i++) {
                m_values[i] = value;
            }
            return *this;
        }

        T& operator[](int index) {
            return m_values[index];
        }

        T& operator[](const std::vector<int>& indices) {
            if(indices.size() < m_shape.size()) {
                int index = 0;
                int stride = 1;
                for (int i = m_shape.size() - 1; i >= 0; --i) {
                    if (indices[i] >= m_shape[i] || indices[i] < 0) {
                        throw std::out_of_range("Index out of bounds");
                    }
                    index += indices[i] * stride;
                    stride *= m_shape[i];
                }
                return m_values[index];
            } else {
                throw std::invalid_argument("Index size mismatch");
            }
        }
        

        Tensor<T> head(const int n) const {
            std::vector<int> new_shape = m_shape;
            new_shape[0] = n;
            return Tensor<T>(new_shape, std::vector<T>(m_values.begin(), m_values.begin() + n * m_strides[0]));
        }


        Tensor<T> normalize(const T& min, const T& max) const {
            std::vector<T> new_values = m_values;
            T old_min = m_values[0];
            T old_max = m_values[0];
            for (int i = 0; i < m_values.size(); i++) {
                if (m_values[i] < old_min) {
                    old_min = m_values[i];
                }
                if (m_values[i] > old_max) {
                    old_max = m_values[i];
                }
            }

            for (int i = 0; i < m_values.size(); i++) {
                new_values[i] = min + (m_values[i] - old_min) * (max - min) / (old_max - old_min);
            }

            return Tensor<T>(m_shape, new_values);
        }

        Tensor<T> flatten() const {
            std::vector<int> new_shape = {m_shape[0], 1};
            return Tensor<T>(new_shape, m_values);
        }

        Tensor transpose() {
            if (m_shape.size() != 2) {
                throw std::invalid_argument("Incompatible shape for transpose");
            }

            Tensor result;
            result.zeros({m_shape[1], m_shape[0]});

            for (size_t i = 0; i < m_shape[0]; ++i) {
                for (size_t j = 0; j < m_shape[1]; ++j) {
                    result.m_values[j * m_shape[0] + i] = m_values[i * m_shape[1] + j];
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
                    result.m_values[j * m_shape[0] + i] = m_values[i * m_shape[1] + j];
                }
            }

            return result;
        }

        Tensor<T> reshape(const std::vector<int>& new_shape) {
            int new_size = 1;
            for (int i = 0; i < new_shape.size(); i++) {
                new_size *= new_shape[i];
            }

            if (new_size != m_values.size()) {
                std::cout << "Error: reshape size mismatch" << std::endl;
                return Tensor<T>();
            }

            return Tensor<T>(new_shape, m_values);
        }

        // Tensor<T> transpose() const {
        //     std::vector<int> new_shape = m_shape;
        //     std::reverse(new_shape.begin(), new_shape.end());
        //     return Tensor<T>(new_shape, m_values, m_gradients, !m_is_transposed);
        // }

        bool requires_gradient() const {
            return m_requires_grad;
        }

        bool has_gradient() const {
            return m_gradients.size() > 0;
        }

        bool is_leaf() const {
            return m_is_leaf;
        }

        std::vector<int> shape() const {
            return m_shape;
        }

        void zero_grad() {
            if (m_gradients.size() > 0) { // NOTE: may not always be initialized
                for (int i = 0; i < m_gradients.size(); i++) {
                    m_gradients[i] = 0;
                }
            } else {
                m_gradients = std::vector<T>(m_values.size(), 0);
            }
        }

        void set_requires_gradient(const bool count_gradient) {
            m_requires_grad = count_gradient;
            // initialize gradients to 0
            if (count_gradient) {
                m_gradients = std::vector<T>(m_values.size(), 0);
            }
        }

        std::shared_ptr<Operator<T>> get_grad_op() {
            return m_grad_op;
        }

        void set_grad_op(std::shared_ptr<Operator<T>> grad_op) {
            m_grad_op = grad_op;
        }


        void set_leaf(const bool is_leaf) {
            m_is_leaf = is_leaf;
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
            printTensorHelper(0, m_shape, m_values, cum);
            std::cout << "]" << std::endl;
        }

    // TODO: do not calculate the strides
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

        if (index + values.size() > m_values.size()) {
            throw std::invalid_argument("Values must fit in tensor");
        }

        for (size_t i = 0; i < values.size(); ++i) {
            m_values[index + i] = values[i];
        }
    }


    // Overloaded set_values function for setting values at a single index
    void set_values(int index, const std::vector<T>& values) {
        auto indices = std::vector<int>(m_shape.size(), 0);
        indices.front() = index;
        set_values(indices, values);
        // set_values(std::vector<int>{index}, values);
    }

    // Overloaded set_values function for variable arguments
    template<typename... Args>
    void set_values(int index, Args... args) {
        auto indices = std::vector<int>(m_shape.size(), 0);
        indices.front() = index;
        set_values(indices, {static_cast<T>(args)...});
    }

    // Overloaded set_values function for multi-dimensional indices
    template<typename... Args>
    void set_values(const std::vector<int>& indices, Args... args) {
        set_values(indices, {static_cast<T>(args)...});
    }

    // Get index from multi-dimensional indices
    int get_index(const std::vector<int>& indices) const {
        int index = 0;
        for (int i = 0; i < indices.size(); ++i) {
            index += indices[i] * m_strides[i];
        }
        return index;
    }

    // Overloaded operator() function for getting values at a single index
    T& operator()(int index) {
        return m_values[index];
    }

    // Overloaded operator() function for getting values at a single index
    T operator()(int index) const {
        return m_values[index];
    }

    // Overloaded operator() function for getting values at multi-dimensional indices
    T& operator()(const std::vector<int>& indices) {
        return m_values[get_index(indices)];
    }

    // Overloaded operator() function for getting values at multi-dimensional indices
    T operator()(const std::vector<int>& indices) const {
        return m_values[get_index(indices)];
    }

    // Overloaded operator() function for getting values at multi-dimensional indices using variadic templates
    template<typename... Args>
    T& operator()(Args... args) {
        return m_values[get_index({static_cast<int>(args)...})];
    }

    // Overloaded operator() function for getting values at multi-dimensional indices using variadic templates
    template<typename... Args>
    T operator()(Args... args) const {
        return m_values[get_index({static_cast<int>(args)...})];
    }

    // TODO: do not calculate the strides
    // Get values for tensors at specified indices
    Tensor<T> get_values(const std::vector<int>& indices) const {
        // std::vector<int> strides(m_shape.size());
        // strides.back() = 1;
        // for (int i = m_shape.size() - 2; i >= 0; --i) {
        //     strides[i] = strides[i + 1] * m_shape[i + 1];
        // }

        int index = 0;
        for (int i = 0; i < indices.size(); ++i) {
            index += indices[i] * m_strides[i];
        }

        // get the new shape
        if (indices.size() == m_shape.size()) {
            return Tensor<T>(std::vector<T>{m_values[index]}, std::vector<int>{1});
        }
        std::vector<int> new_shape(m_shape.begin() + indices.size(), m_shape.end());

 
        Tensor<T> result;
        result.m_shape = new_shape;

        result.m_values = std::vector<T>(m_strides[0]);

        for (int i = 0; i < m_strides[0]; ++i) {
            result.m_values[i] = m_values[index + i];
        }

        return result;
    }

    T value() const {
        if (m_values.size() != 1) {
            throw std::invalid_argument("Tensor must have a single value");
        }
        return m_values[0];
    }

    T gradient() const {
        if (m_gradients.size() != 1) {
            throw std::invalid_argument("Tensor must have a single gradient");
        }
        return m_gradients[0];
    }

    std::vector<T> values() const {
        return m_values;
    }

    std::vector<T> gradients() const {
        return m_gradients;
    }

    Tensor<T> gradient_tensor() const {
        if (m_gradients.size() != m_values.size()) {
            throw std::invalid_argument("Gradient size mismatch");
        }
        return Tensor<T>(m_shape, m_gradients);
    }

    void set_gradient(const std::vector<T>& gradients) {
        m_gradients = gradients;
    }

    void set_gradient(const Tensor<T>& gradient) {
        m_gradients = gradient.values();
    }


    // creates accumulate grad operator if needed
    void setup_accumulate_grad() {
        if (requires_gradient() && is_leaf() && get_grad_op() == nullptr) {
            set_grad_op(std::make_shared<AccumulateGrad<T>>(this));
        }
    }

    // COMPARISSON OPERATIONS
    // EQUALITY
    bool operator==(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            return false;
        }

        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] != rhs.m_values[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator==(const T& rhs) const {
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] != rhs) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const Tensor<T>& rhs) const {
        return !(*this == rhs);
    }

    bool operator!=(const T& rhs) const {
        return !(*this == rhs);
    }

    // GREATER THAN
    bool operator>(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] <= rhs.m_values[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator>(const T& rhs) const {
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] <= rhs) {
                return false;
            }
        }

        return true;
    }

    // LESS THAN
    bool operator<(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] >= rhs.m_values[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator<(const T& rhs) const {
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] >= rhs) {
                return false;
            }
        }

        return true;
    }

    // GREATER THAN OR EQUAL TO
    bool operator>=(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] < rhs.m_values[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator>=(const T& rhs) const {
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] < rhs) {
                return false;
            }
        }

        return true;
    }

    // LESS THAN OR EQUAL TO
    bool operator<=(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] > rhs.m_values[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator<=(const T& rhs) const {
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] > rhs) {
                return false;
            }
        }

        return true;
    }

    // reduce sum
    Tensor<T> reduce_sum(int axis) const {
        if (axis < 0 || axis >= m_shape.size()) {
            throw std::invalid_argument("Invalid axis");
        }

        std::vector<int> shape = m_shape;

        int outer_dim = 1;
        for (int i = 0; i < axis; ++i) {
            outer_dim *= shape[i];
        }

        int reduce_dim = shape[axis];
        int inner_dim = 1;
        for (int i = axis + 1; i < shape.size(); ++i) {
            inner_dim *= shape[i];
        }

        std::vector<T> result(outer_dim * inner_dim, 0);

        int total_size = m_values.size();
        for (int i = 0; i < total_size; ++i) {
            int outer_idx = i / (reduce_dim * inner_dim);
            int inner_idx = i % inner_dim;
            result[outer_idx * inner_dim + inner_idx] += m_values[i];
        }

        shape.erase(shape.begin() + axis);
        return Tensor<T>(shape, result);
    }

    // MATH OPERATIONS
    // TODO: add, subtract, multiply, divide, power, sqrt, exp, log, abs, sum, mean, dot, matmul, conv2d, maxpool2d, avgpool2d, relu, softmax, sigmoid, tanh

    // ADDITION
    Tensor<T> operator+(Tensor<T>& rhs) {
        if (m_shape == rhs.m_shape) {
            return add(rhs);
        } 

        // broadcast
        if (m_shape.size() > rhs.m_shape.size()) {
            return broadcast_add(rhs);
        } else {
            return rhs.broadcast_add(*this);
        }
    }

    Tensor<T> operator+(Tensor<T>&& rhs) {
        return *this + rhs;
    }

    // overload const without gradient graph creation
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (m_shape == other.m_shape) {
            return add(other);
        } 

        // broadcast
        if (m_shape.size() > other.m_shape.size()) {
            return broadcast_add(other);
        } else {
            return other.broadcast_add(*this);
        }
    }

    // broadcast addition
    Tensor<T> broadcast_add(Tensor<T>& rhs) {
        // Get the shapes of lhs and rhs
        const auto& lhs_shape = shape();
        const auto& rhs_shape = rhs.shape();

        // Ensure rhs has one less dimension than lhs
        if (rhs_shape.size() + 1 != lhs_shape.size()) {
            throw std::invalid_argument("RHS must have one dimension less than LHS.");
        }

        // Ensure all dimensions of rhs match corresponding dimensions of lhs except for the one dimension less
        for (size_t i = 0; i < rhs_shape.size(); ++i) {
            if (rhs_shape[i] != lhs_shape[i + 1]) {
                throw std::invalid_argument("Dimension mismatch between LHS and RHS.");
            }
        }

        // Create a new tensor with the shape of lhs
        Tensor<T> result(lhs_shape);

        // get the first dimension of rhs
        int rhs_dim = rhs_shape[0];

        // iterate over the values of the new tensor
        for (int i = 0; i < result.m_values.size(); i++) {
            // get the index of the current value in the new tensor
            std::vector<int> index(lhs_shape.size(), 0);
            int temp = i;
            for (int j = 1; j < lhs_shape.size(); j++) {
                index[j] = temp / result.m_strides[j];
                temp = temp % result.m_strides[j];
            }

            // add the value of the current index in lhs to the value of the current index in rhs
            result.m_values[i] = m_values[i] + rhs.m_values[i % rhs_dim];
        }


        
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();
        rhs.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || rhs.requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), rhs.get_grad_op()};

            result.set_grad_op(std::make_shared<AddBack<T>>(next_operators));
        }

        return result;
    }

    Tensor<T> broadcast_add(Tensor<T>&& rhs) {
        return *this + rhs;
    }

    // overload const without gradient graph creation
    Tensor<T> broadcast_add(const Tensor<T>& rhs) const {
        // Get the shapes of lhs and rhs
        const auto& lhs_shape = shape();
        const auto& rhs_shape = rhs.shape();

        // Ensure rhs has one less dimension than lhs
        if (rhs_shape.size() + 1 != lhs_shape.size()) {
            throw std::invalid_argument("RHS must have one dimension less than LHS.");
        }

        // Ensure all dimensions of rhs match corresponding dimensions of lhs except for the one dimension less
        for (size_t i = 0; i < rhs_shape.size(); ++i) {
            if (rhs_shape[i] != lhs_shape[i + 1]) {
                throw std::invalid_argument("Dimension mismatch between LHS and RHS.");
            }
        }

        // Create a new tensor with the shape of lhs
        Tensor<T> result(lhs_shape);

        // get the first dimension of rhs
        int rhs_dim = rhs_shape[0];

        // iterate over the values of the new tensor
        for (int i = 0; i < result.m_values.size(); i++) {
            // get the index of the current value in the new tensor
            std::vector<int> index(lhs_shape.size(), 0);
            int temp = i;
            for (int j = 1; j < lhs_shape.size(); j++) {
                index[j] = temp / result.m_strides[j];
                temp = temp % result.m_strides[j];
            }

            // add the value of the current index in lhs to the value of the current index in rhs
            result.m_values[i] = m_values[i] + rhs.m_values[i % rhs_dim];
        }

        
        result.set_leaf(false);

        return result;
    }


    Tensor<T> add(Tensor<T>& rhs) {
        if (m_shape != rhs.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] + rhs.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();
        rhs.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || rhs.requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), rhs.get_grad_op()};

            result.set_grad_op(std::make_shared<AddBack<T>>(next_operators));
        }


        return result;            
    }
    Tensor<T> add(Tensor<T>&& rhs) {
        return *this + rhs;
    }

    // overload const without gradient graph creation
    Tensor<T> add(const Tensor<T>& other) const {
        if (m_shape != other.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] + other.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // SUBTRACTION
    Tensor<T> operator-(Tensor<T>& rhs) {
        if (m_shape != rhs.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] - rhs.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();
        rhs.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || rhs.requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), rhs.get_grad_op()};

            result.set_grad_op(std::make_shared<SubBack<T>>(next_operators));
        }

        return result;            
    }

    Tensor<T> operator-(Tensor<T>&& rhs) {
        return *this - rhs;
    }

    // overload const without gradient graph creation
    Tensor<T> operator-(const Tensor<T>& other) const {
        if (m_shape != other.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] - other.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // UNARY MINUS
    Tensor<T> operator-() {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = -m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<NegBack<T>>(next_operators));
        }

        return result;            
    }

    Tensor<T> operator-() const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = -m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // ELEMENTWISE MULTIPLICATION
    Tensor<T> operator*(Tensor<T>& other) {
        if (m_shape != other.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            
            new_values[i] = m_values[i] * other.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();
        other.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || other.requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), other.get_grad_op()};
            
            std::vector<Tensor<T>> saved_values = { *this, other };

            result.set_grad_op(std::make_shared<MulBack<T>>(saved_values, next_operators));
            
        }

        return result;            
    }

    Tensor<T> operator*(Tensor<T>&& other) {
        return *this * other;
    }

    // TODO: update backward, update operators, all math operations are done

    // overload const without gradient graph creation
    Tensor<T> operator*(const Tensor<T>& other) const {
        if (m_shape != other.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            
            new_values[i] = m_values[i] * other.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // ELEMENTWISE DIVISION
    Tensor<T> operator/(Tensor<T>& other) {
        if (m_shape != other.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            if (other.m_values[i] == 0) {
                std::cout << "Error: division by zero" << std::endl;
                return Tensor<T>();
            }
            new_values[i] = m_values[i] / other.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();
        other.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || other.requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), other.get_grad_op()};
            
            std::vector<Tensor<T>> saved_values = { *this, other };

            result.set_grad_op(std::make_shared<DivBack<T>>(saved_values, next_operators));
            
        }

        return result;            
    }

    Tensor<T> operator/(Tensor<T>&& other) {
        return *this / other;
    }

    // overload const without gradient graph creation
    Tensor<T> operator/(const Tensor<T>& other) const {
        if (m_shape != other.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            if (other.m_values[i] == 0) {
                std::cout << "Error: division by zero" << std::endl;
                return Tensor<T>();
            }
            new_values[i] = m_values[i] / other.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // POWER
    Tensor pow(Tensor<T>& other) {
        if (m_shape != other.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::pow(m_values[i], other.m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();
        other.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || other.requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), other.get_grad_op()};
            
            std::vector<Tensor<T>> saved_values = { *this, other };

            result.set_grad_op(std::make_shared<PowBack<T>>(saved_values, next_operators));
            
        }

        return result;            
    }

    Tensor pow(Tensor<T>&& other) {
        return pow(other);
    }

    Tensor pow(const Tensor<T>& other) const {
        if (m_shape != other.m_shape) {
            std::cout << "Error: shape mismatch" << std::endl;
            return Tensor<T>();
        }

        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::pow(m_values[i], other.m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // TODO: figure this out
    // Tensor pow(const T& other) const {
    //     std::vector<T> new_values(m_values.size());
    //     for (int i = 0; i < m_values.size(); i++) {
    //         new_values[i] = std::pow(m_values[i], other);
    //     }

    //     Tensor<T> result(m_shape, new_values);
    //     result.set_leaf(false);

    //     return result;            
    // }



    Tensor<T> operator^(Tensor<T>& other) {
        if (m_shape == other.m_shape) {
            // elementwise power
            return pow(other);
        } else if (other.m_shape.size() == 1 && other.m_shape[0] == 1) {
            // scalar power
            throw std::invalid_argument("Scalar power not implemented");
        } else {
            throw std::invalid_argument("Incompatible shapes for power operation");
        }
    }

    Tensor<T> operator^(Tensor<T>&& other) {
        return *this ^ other;
    }

    // overload const without gradient graph creation
    Tensor<T> operator^(const Tensor<T>& other) const {
        if (m_shape == other.m_shape) {
            // elementwise power
            return pow(other);
        } else if (other.m_shape.size() == 1 && other.m_shape[0] == 1) {
            // scalar power
            throw std::invalid_argument("Scalar power not implemented");
        } else {
            throw std::invalid_argument("Incompatible shapes for power operation");
        }
    }

    // SQRT
    Tensor<T> sqrt() {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::sqrt(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<SqrtBack<T>>(std::vector<Tensor<T>>{ *this }, next_operators));
        }

        return result;            
    }

    Tensor<T> sqrt() const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::sqrt(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // LOG
    Tensor<T> log() {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] <= 0) {
                std::cout << "Error: log of non-positive number" << std::endl;
                return Tensor<T>();
            }
            new_values[i] = std::log(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<LogBack<T>>(std::vector<Tensor<T>>{ *this }, next_operators));
        }

        return result;            
    }

    Tensor<T> log() const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] <= 0) {
                std::cout << "Error: log of non-positive number" << std::endl;
                return Tensor<T>();
            }
            new_values[i] = std::log(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // EXP
    Tensor<T> exp() {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::exp(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<ExpBack<T>>(std::vector<Tensor<T>>{ *this }, next_operators));
        }

        return result;            
    }

    Tensor<T> exp() const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::exp(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // ABS
    Tensor<T> abs() {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::abs(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<AbsBack<T>>(std::vector<Tensor<T>>{ *this }, next_operators));
        }

        return result;            
    }

    Tensor<T> abs() const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::abs(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // TRIGONOMETRIC FUNCTIONS
    // SIN
    Tensor<T> sin() {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::sin(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<SinBack<T>>(std::vector<Tensor<T>>{ *this }, next_operators));
        }

        return result;            
    }

    Tensor<T> sin() const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::sin(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // COS
    Tensor<T> cos() {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::cos(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<CosBack<T>>(std::vector<Tensor<T>>{ *this }, next_operators));
        }

        return result;            
    }

    Tensor<T> cos() const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::cos(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // TAN
    Tensor<T> tan() {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::tan(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<TanBack<T>>(std::vector<Tensor<T>>{ *this }, next_operators));
        }

        return result;            
    }

    Tensor<T> tan() const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::tan(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // TENSOR SPESIFIC OPERATIONS
    // SUM
    Tensor<T> sum() {
        T sum = 0;
        for (int i = 0; i < m_values.size(); i++) {
            sum += m_values[i];
        }

        return Tensor<T>(std::vector<T>{sum}, std::vector<int>{1});
    }

    Tensor<T> sum() const {
        T sum = 0;
        for (int i = 0; i < m_values.size(); i++) {
            sum += m_values[i];
        }

        return Tensor<T>(std::vector<T>{sum}, std::vector<int>{1});
    }

    // MATMUL
    Tensor<T> matmul(Tensor<T>& other) {
        if (m_shape.size() != 2 || other.m_shape.size() != 2) {
            throw std::invalid_argument("Matrix multiplication requires 2D tensors");
        }

        if (m_shape[1] != other.m_shape[0]) {
            throw std::invalid_argument("Matrix multiplication requires compatible shapes");
        }

        std::vector<int> new_shape = {m_shape[0], other.m_shape[1]};
        std::vector<T> new_values(new_shape[0] * new_shape[1], 0);

        for (int i = 0; i < m_shape[0]; i++) {
            for (int j = 0; j < other.m_shape[1]; j++) {
                for (int k = 0; k < m_shape[1]; k++) {
                    new_values[i * new_shape[1] + j] += m_values[i * m_shape[1] + k] * other.m_values[k * other.m_shape[1] + j];
                }
            }
        }

        Tensor<T> result(new_shape, new_values);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();
        other.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || other.requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), other.get_grad_op()};
            
            std::vector<Tensor<T>> saved_values = { *this, other };

            result.set_grad_op(std::make_shared<MatmulBack<T>>(saved_values, next_operators));
            
        }

        return result;            
    }

    Tensor<T> matmul(Tensor<T>&& other) {
        return matmul(other);
    }

    // overload const without gradient graph creation
    Tensor<T> matmul(const Tensor<T>& other) const {
        if (m_shape.size() != 2 || other.m_shape.size() != 2) {
            throw std::invalid_argument("Matrix multiplication requires 2D tensors");
        }

        if (m_shape[1] != other.m_shape[0]) {
            throw std::invalid_argument("Matrix multiplication requires compatible shapes");
        }

        std::vector<int> new_shape = {m_shape[0], other.m_shape[1]};
        std::vector<T> new_values(new_shape[0] * new_shape[1], 0);

        for (int i = 0; i < m_shape[0]; i++) {
            for (int j = 0; j < other.m_shape[1]; j++) {
                for (int k = 0; k < m_shape[1]; k++) {
                    new_values[i * new_shape[1] + j] += m_values[i * m_shape[1] + k] * other.m_values[k * other.m_shape[1] + j];
                }
            }
        }

        Tensor<T> result(new_shape, new_values);
        result.set_leaf(false);

        return result;            
    }



    // END OF MATH OPERATIONS

    void backward(const Tensor& gradient = Tensor(1)) {
        if (m_grad_op == nullptr) {
            throw std::invalid_argument("No gradient operator set");
            return;
        }
        
        // check that the shape of the tensor is one

        m_grad_op->evaluate(gradient);
    }

};

// MATH OPERATIONS

template <typename T>
Tensor<T> pow(Tensor<T>& base, Tensor<T>& exponent) {
    return base.pow(exponent);
}

template <typename T>
Tensor<T> pow(Tensor<T>& base, Tensor<T>&& exponent) {
    return base.pow(exponent);
}

template <typename T>
Tensor<T> pow(Tensor<T>&& base, Tensor<T>& exponent) {
    return base.pow(exponent);
}

template <typename T>
Tensor<T> pow(Tensor<T>&& base, Tensor<T>&& exponent) {
    return base.pow(exponent);
}

template <typename T>
Tensor<T> log(Tensor<T>& tensor) {
    return tensor.log();
}

template <typename T>
Tensor<T> log(Tensor<T>&& tensor) {
    return tensor.log();
}

template <typename T>
Tensor<T> exp(Tensor<T>& tensor) {
    return tensor.exp();
}

template <typename T>
Tensor<T> exp(Tensor<T>&& tensor) {
    return tensor.exp();
}

template <typename T>
Tensor<T> sqrt(Tensor<T>& tensor) {
    return tensor.sqrt();
}

template <typename T>
Tensor<T> sqrt(Tensor<T>&& tensor) {
    return tensor.sqrt();
}

template <typename T>
Tensor<T> abs(Tensor<T>& tensor) {
    return tensor.abs();
}

template <typename T>
Tensor<T> abs(Tensor<T>&& tensor) {
    return tensor.abs();
}

template <typename T>
Tensor<T> sin(Tensor<T>& tensor) {
    return tensor.sin();
}

template <typename T>
Tensor<T> sin(Tensor<T>&& tensor) {
    return tensor.sin();
}

template <typename T>
Tensor<T> cos(Tensor<T>& tensor) {
    return tensor.cos();
}

template <typename T>
Tensor<T> cos(Tensor<T>&& tensor) {
    return tensor.cos();
}

template <typename T>
Tensor<T> tan(Tensor<T>& tensor) {
    return tensor.tan();
}



} // namespace sdlm

namespace std {
    
    template <typename T>
    sdlm::Tensor<T> pow(sdlm::Tensor<T>& base, sdlm::Tensor<T>& exponent) {
        return sdlm::pow(base, exponent);
    }

    template <typename T>
    sdlm::Tensor<T> pow(sdlm::Tensor<T>& base, sdlm::Tensor<T>&& exponent) {
        return sdlm::pow(base, exponent);
    }

    template <typename T>
    sdlm::Tensor<T> pow(sdlm::Tensor<T>&& base, sdlm::Tensor<T>& exponent) {
        return sdlm::pow(base, exponent);
    }

    template <typename T>
    sdlm::Tensor<T> pow(sdlm::Tensor<T>&& base, sdlm::Tensor<T>&& exponent) {
        return sdlm::pow(base, exponent);
    }

    template <typename T>
    sdlm::Tensor<T> log(sdlm::Tensor<T>& tensor) {
        return sdlm::log(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> log(sdlm::Tensor<T>&& tensor) {
        return sdlm::log(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> exp(sdlm::Tensor<T>& tensor) {
        return sdlm::exp(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> exp(sdlm::Tensor<T>&& tensor) {
        return sdlm::exp(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> sqrt(sdlm::Tensor<T>& tensor) {
        return sdlm::sqrt(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> sqrt(sdlm::Tensor<T>&& tensor) {
        return sdlm::sqrt(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> abs(sdlm::Tensor<T>& tensor) {
        return sdlm::abs(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> abs(sdlm::Tensor<T>&& tensor) {
        return sdlm::abs(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> sin(sdlm::Tensor<T>& tensor) {
        return sdlm::sin(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> sin(sdlm::Tensor<T>&& tensor) {
        return sdlm::sin(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> cos(sdlm::Tensor<T>& tensor) {
        return sdlm::cos(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> cos(sdlm::Tensor<T>&& tensor) {
        return sdlm::cos(tensor);
    }
    
    template <typename T>
    sdlm::Tensor<T> tan(sdlm::Tensor<T>& tensor) {
        return sdlm::tan(tensor);
    }

    template <typename T>
    sdlm::Tensor<T> tan(sdlm::Tensor<T>&& tensor) {
        return sdlm::tan(tensor);
    }

} // namespace std
      