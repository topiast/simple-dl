#pragma once

#include "math/backoperator.h"

#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
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

        // Convert flat index to multi-dimensional index
        std::vector<int> unravel_index(int flat_index, const std::vector<int>& shape) const {
            std::vector<int> index(shape.size());
            for (int i = shape.size() - 1; i >= 0; --i) {
                index[i] = flat_index % shape[i];
                flat_index /= shape[i];
            }
            return index;
        }

        // Convert multi-dimensional index to flat index
        int ravel_index(const std::vector<int>& index, const std::vector<int>& shape) const {
            int flat_index = 0;
            int stride = 1;
            for (int i = shape.size() - 1; i >= 0; --i) {
                flat_index += index[i] * stride;
                stride *= shape[i];
            }
            return flat_index;
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
        
        Tensor& uniform(const std::vector<int>& shape, float low = 0.f, float high = 1.f) {
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
            std::uniform_real_distribution<float> d(low, high);

            for (int i = 0; i < size; i++) {
                float r = d(gen);
                
                m_values[i] = r;
            }

            return *this;
        }

        Tensor& normal(const std::vector<int>& shape, float mean = 0.f, float stddev = 1.f) {
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

        Tensor<T> tail(const int n) const {
            std::vector<int> new_shape = m_shape;
            new_shape[0] = n;
            return Tensor<T>(new_shape, std::vector<T>(m_values.end() - n * m_strides[0], m_values.end()));
        }

        Tensor<T> slice(const int start, const int end) const {
            std::vector<int> new_shape = m_shape;
            new_shape[0] = end - start;
            return Tensor<T>(new_shape, std::vector<T>(m_values.begin() + start * m_strides[0], m_values.begin() + end * m_strides[0]));
        }

        Tensor<T> slice(const int start, const int end, const int axis) const {
            if (axis < 0 || axis >= m_shape.size()) {
                throw std::invalid_argument("Invalid axis");
            }

            std::vector<int> new_shape = m_shape;
            new_shape[axis] = end - start;
            return Tensor<T>(new_shape, std::vector<T>(m_values.begin() + start * m_strides[axis], m_values.begin() + end * m_strides[axis]));
        }

        std::vector<int> shuffle_indices() {
            std::vector<int> indices(m_shape[0]);  
            std::iota(indices.begin(), indices.end(), 0);
            
            std::random_device rd;
            std::mt19937 g(rd()); // Mersenne Twister RNG
            std::shuffle(indices.begin(), indices.end(), g); // Use std::shuffle
            
            return indices;
        }

        Tensor<T> shuffle() const {
            std::vector<int> indices = shuffle_indices();
            std::vector<T> new_values(m_values.size());
            for (int i = 0; i < m_shape[0]; i++) {
                for (int j = 0; j < m_strides[0]; j++) {
                    new_values[i * m_strides[0] + j] = m_values[indices[i] * m_strides[0] + j];
                }
            }
            return Tensor<T>(m_shape, new_values);
        }   

        Tensor<T> select_indices(const std::vector<int>& indices, int axis) const {
            if (axis < 0 || axis >= m_shape.size()) {
                throw std::invalid_argument("Invalid axis");
            }
            
            // Validate indices are within bounds
            const int axis_size = m_shape[axis];
            for (const int index : indices) {
                if (index < 0 || index >= axis_size) {
                    throw std::out_of_range("Index out of bounds for axis " + std::to_string(axis));
                }
            }

            // Calculate new shape
            std::vector<int> new_shape = m_shape;
            new_shape[axis] = indices.size();

            // Calculate total elements in new tensor
            size_t total_elements = 1;
            for (const int dim : new_shape) {
                total_elements *= dim;
            }

            // Create buffer for new values
            std::vector<T> new_values;
            new_values.reserve(total_elements);

            // Calculate coordinates and copy values
            const std::vector<int>& original_strides = m_strides;
            for (size_t flat_idx = 0; flat_idx < total_elements; ++flat_idx) {
                // Convert flat index to coordinates in new tensor
                std::vector<int> coords(new_shape.size());
                size_t remainder = flat_idx;
                
                for (int dim = new_shape.size() - 1; dim >= 0; --dim) {
                    coords[dim] = remainder % new_shape[dim];
                    remainder /= new_shape[dim];
                }

                // Replace coordinate at target axis with actual index
                coords[axis] = indices[coords[axis]];

                // Calculate original flat index
                size_t original_flat = 0;
                for (size_t dim = 0; dim < coords.size(); ++dim) {
                    original_flat += coords[dim] * original_strides[dim];
                }

                new_values.push_back(m_values[original_flat]);
            }

            return Tensor<T>(new_shape, new_values);
        }  

        int number_of_batches(const int batch_size) const {
            return (m_shape[0] + batch_size - 1) / batch_size;
        }       

        // take batch size and index i and return the i-th batch. If last batch is smaller than batch size, return the last batch
        Tensor<T> batch(const int batch_size, const int i) const {
            int start = i * batch_size;
            int end = std::min((i + 1) * batch_size, m_shape[0]);
            return slice(start, end);
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

        // flattens the last two dimensions
        Tensor flatten() const {
            Tensor result;
            for (int i = 0; i < m_shape.size() - 2; ++i) {
                result.m_shape.push_back(m_shape[i]);
            }
            result.m_shape.push_back(m_shape[m_shape.size() - 2] * m_shape[m_shape.size() - 1]);
            
            result.m_values = m_values;

            return result;
        }

        Tensor transpose() {
            if (m_shape.size() != 2) {
                throw std::invalid_argument("Incompatible shape for transpose");
            }

            Tensor result;
            result.zeros({m_shape[1], m_shape[0]});

            #pragma omp parallel for if(m_values.size() > 2000)
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

            #pragma omp parallel for if(m_values.size() > 2000) 
            for (size_t i = 0; i < m_shape[0]; ++i) {
                for (size_t j = 0; j < m_shape[1]; ++j) {
                    result.m_values[j * m_shape[0] + i] = m_values[i * m_shape[1] + j];
                }
            }

            return result;
        }

        Tensor<T> diag() const {
            if (m_shape.size() != 1) {
                throw std::invalid_argument("Incompatible shape for diag");
            }

            Tensor result;
            result.zeros({m_shape[0], m_shape[0]});

            for (size_t i = 0; i < m_shape[0]; ++i) {
                result.m_values[i * m_shape[0] + i] = m_values[i];
            }

            return result;
        }

        Tensor<T> one_hot(const int num_classes) const {
            Tensor result;
            result.zeros({m_shape[0], num_classes});

            for (size_t i = 0; i < m_shape[0]; ++i) {
                result.m_values[i * num_classes + m_values[i]] = 1;
            }

            return result;
        }

        Tensor<T> argmax(int axis) const {
            if (axis < 0 || axis >= m_shape.size()) {
                throw std::invalid_argument("Invalid axis");
            }

            std::vector<int> new_shape = m_shape;
            new_shape.erase(new_shape.begin() + axis);

            Tensor<T> result;
            result.zeros(new_shape);

            for (size_t i = 0; i < m_shape[0]; ++i) {
                std::vector<int> indices = unravel_index(i, m_shape);
                indices.erase(indices.begin() + axis);

                int max_index = 0;
                T max_value = m_values[i * m_strides[0]];
                for (size_t j = 1; j < m_shape[axis]; ++j) {
                    indices.insert(indices.begin() + axis, j);
                    T value = m_values[ravel_index(indices, m_shape)];
                    if (value > max_value) {
                        max_index = j;
                        max_value = value;
                    }
                    indices.erase(indices.begin() + axis);
                }

                indices.insert(indices.begin() + axis, max_index);
                result.m_values[ravel_index(indices, new_shape)] = max_index;
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

    // Get values for tensors at specified indices
    Tensor<T> get_values(const std::vector<int>& indices) const {
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

    int size () const {
        return m_values.size();
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

    std::vector<T> gradients() {
        return m_gradients;
    }

    Tensor<T> gradient_tensor() const {
        if (m_gradients.size() != m_values.size()) {
            throw std::invalid_argument("Gradient size mismatch");
        }
        return Tensor<T>(m_shape, m_gradients);
    }

    void set_gradient(const std::vector<T>& gradients) {
        if (gradients.size() != m_values.size()) {
            throw std::invalid_argument("Gradient size mismatch");
        }
        m_gradients = gradients;
    }

    void set_gradient(const Tensor<T>& gradient) {
        if (gradient.shape() != m_shape) {
            throw std::invalid_argument("Gradient shape mismatch");
        }
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
    // bool operator==(const Tensor<T>& rhs) const {
    //     if (m_shape != rhs.m_shape) {
    //         return false;
    //     }

    //     for (int i = 0; i < m_values.size(); i++) {
    //         if (m_values[i] != rhs.m_values[i]) {
    //             return false;
    //         }
    //     }

    //     return true;
    // }

    // bool operator==(const T& rhs) const {
    //     for (int i = 0; i < m_values.size(); i++) {
    //         if (m_values[i] != rhs) {
    //             return false;
    //         }
    //     }

    //     return true;
    // }

    Tensor<T> operator==(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] == rhs.m_values[i];
        }

        return Tensor<T>(m_shape, new_values);
    }

    Tensor<T> operator!=(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] != rhs.m_values[i];
        }

        return Tensor<T>(m_shape, new_values);
    }

    Tensor<T> operator!=(const T& rhs) const {
        std::vector<T> new_values(m_values.size());
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] != rhs;
        }

        return Tensor<T>(m_shape, new_values);
    }

    // GREATER THAN
    Tensor<T> operator>(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] > rhs.m_values[i];
        }

        return Tensor<T>(m_shape, new_values);
    }

    Tensor<T> operator>(const T& rhs) const {
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] > rhs;
        }

        return Tensor<T>(m_shape, new_values);
    }

    // LESS THAN
    Tensor<T> operator<(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] < rhs.m_values[i];
        }

        return Tensor<T>(m_shape, new_values);
    }

    Tensor<T> operator<(const T& rhs) const {
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] < rhs;
        }

        return Tensor<T>(m_shape, new_values);
    }

    // GREATER THAN OR EQUAL TO
    Tensor<T> operator>=(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] >= rhs.m_values[i];
        }

        return Tensor<T>(m_shape, new_values);
    }

    Tensor<T> operator>=(const T& rhs) const {
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] >= rhs;
        }

        return Tensor<T>(m_shape, new_values);
    }

    // LESS THAN OR EQUAL TO
    Tensor<T> operator<=(const Tensor<T>& rhs) const {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Shape mismatch");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] <= rhs.m_values[i];
        }

        return Tensor<T>(m_shape, new_values);
    }

    Tensor<T> operator<=(const T& rhs) const {
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] <= rhs;
        }

        return Tensor<T>(m_shape, new_values);
    }

    // map function
    Tensor<T> map(std::function<T(T)> func) const {
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = func(m_values[i]);
        }

        return Tensor<T>(m_shape, new_values);
    }


    Tensor<T> reduce_sum(int axis, bool keepdims = false) const {
        // Handle negative axis (Python-style indexing)
        if (axis < 0) {
            axis += m_shape.size();
        }
        if (axis < 0 || axis >= m_shape.size()) {
            throw std::invalid_argument("Invalid axis");
        }

        std::vector<int> shape = m_shape;
        const int original_axis = axis;

        // Calculate dimensions
        int outer_dim = 1;
        for (int i = 0; i < axis; ++i) {
            outer_dim *= shape[i];
        }

        int reduce_dim = shape[axis];
        int inner_dim = 1;
        for (int i = axis + 1; i < shape.size(); ++i) {
            inner_dim *= shape[i];
        }

        // Compute reduced values
        std::vector<T> result(outer_dim * inner_dim, 0);
        for (int i = 0; i < m_values.size(); ++i) {
            int outer_idx = i / (reduce_dim * inner_dim);
            int inner_idx = i % inner_dim;
            result[outer_idx * inner_dim + inner_idx] += m_values[i];
        }

        // Handle keepdims
        if (keepdims) {
            shape[original_axis] = 1;  // Keep reduced dimension as size 1
        } else {
            shape.erase(shape.begin() + original_axis);
        }

        return Tensor<T>(shape, result);
    }

    // ACTIVATION FUNCTIONS
    // RELU
    Tensor<T> relu() {
        Tensor<T> result(m_shape);
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            result.m_values[i] = std::max(m_values[i], T(0));
        }

        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            std::vector<Tensor<T>> saved_values = { *this };

            result.set_grad_op(std::make_shared<ReluBack<T>>(saved_values, next_operators));
        }

        return result;
    }

    Tensor<T> relu() const {
        Tensor<T> result(m_shape);
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            result.m_values[i] = std::max(m_values[i], T(0));
        }

        result.set_leaf(false);

        return result;
    }

    // SOFTMAX
    Tensor<T> softmax() {
        Tensor<T> result(m_shape);
        
        for (int i = 0; i < m_values.size(); i += m_shape.back()) {
            T sum = 0;
            for (int j = 0; j < m_shape.back(); j++) {
                sum += std::exp(m_values[i + j]);
            }
            # pragma omp parallel for if(m_shape.back() > 2000)
            for (int j = 0; j < m_shape.back(); j++) {
                result.m_values[i + j] = std::exp(m_values[i + j]) / sum;
            }
        }

        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            std::vector<Tensor<T>> saved_values = { *this };

            result.set_grad_op(std::make_shared<SoftmaxBack<T>>(saved_values, next_operators));
        }

        return result;
    }

    Tensor<T> softmax() const {
        Tensor<T> result(m_shape);
        
        for (int i = 0; i < m_values.size(); i += m_shape.back()) {
            T sum = 0;
            for (int j = 0; j < m_shape.back(); j++) {
                sum += std::exp(m_values[i + j]);
            }
            # pragma omp parallel for if(m_shape.back() > 2000)
            for (int j = 0; j < m_shape.back(); j++) {
                result.m_values[i + j] = std::exp(m_values[i + j]) / sum;
            }
        }

        result.set_leaf(false);

        return result;
    }



    // SIGMOID
    Tensor<T> sigmoid() {
        Tensor<T> result(m_shape);
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            result.m_values[i] = 1 / (1 + std::exp(-m_values[i]));
        }

        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            std::vector<Tensor<T>> saved_values = { *this };

            result.set_grad_op(std::make_shared<SigmoidBack<T>>(saved_values, next_operators));
        }

        return result;
    }

    Tensor<T> sigmoid() const {
        Tensor<T> result(m_shape);
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            result.m_values[i] = 1 / (1 + std::exp(-m_values[i]));
        }

        result.set_leaf(false);

        return result;
    }

    // TANH
    Tensor<T> tanh() {
        Tensor<T> result(m_shape);
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            result.m_values[i] = std::tanh(m_values[i]);
        }

        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);
        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            std::vector<Tensor<T>> saved_values = { *this };

            result.set_grad_op(std::make_shared<TanhBack<T>>(saved_values, next_operators));
        }

        return result;
    }

    Tensor<T> tanh() const {
        Tensor<T> result(m_shape);
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            result.m_values[i] = std::tanh(m_values[i]);
        }

        result.set_leaf(false);

        return result;
    }

    // END OF ACTIVATION FUNCTIONS

    // MATH OPERATIONS
    // TODO: add, subtract, multiply, divide, power, sqrt, exp, log, abs, sum, mean, dot, matmul, conv2d, maxpool2d, avgpool2d, relu, softmax, sigmoid, tanh

    // ADDITION
    Tensor<T> operator+(Tensor<T>& rhs) {
        if (m_shape == rhs.m_shape) {
            return add(rhs);
        } 

        return broadcast_add(rhs);
    }

    Tensor<T> operator+(Tensor<T>&& rhs) {
        return *this + rhs;
    }

    // overload const without gradient graph creation
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (m_shape == other.m_shape) {
            return add(other);
        } 

        return broadcast_add(other);
    }

    Tensor<T> broadcast_add(Tensor<T>& rhs) {
        const auto& lhs_shape = shape();
        const auto& rhs_shape = rhs.shape();

        // Align shapes by padding with 1s on the left for RHS
        size_t max_dims = std::max(lhs_shape.size(), rhs_shape.size());
        std::vector<int> lhs_padded = lhs_shape;
        std::vector<int> rhs_padded = rhs_shape;

        while (lhs_padded.size() < max_dims) {
            lhs_padded.insert(lhs_padded.begin(), 1);
        }
        while (rhs_padded.size() < max_dims) {
            rhs_padded.insert(rhs_padded.begin(), 1);
        }

        // Check broadcast compatibility
        std::vector<int> result_shape(max_dims);
        for (size_t i = 0; i < max_dims; ++i) {
            if (lhs_padded[i] != rhs_padded[i] && lhs_padded[i] != 1 && rhs_padded[i] != 1) {
                throw std::invalid_argument("Shapes are not broadcastable");
            }
            result_shape[i] = std::max(lhs_padded[i], rhs_padded[i]);
        }

        // Compute broadcasted values
        Tensor<T> result(result_shape);
        for (int i = 0; i < result.size(); ++i) {
            // Calculate multi-dimensional indices for LHS and RHS
            auto lhs_index = unravel_index(i, result_shape);
            auto rhs_index = lhs_index;

            // Adjust indices for broadcasting
            for (size_t dim = 0; dim < max_dims; ++dim) {
                if (lhs_padded[dim] == 1) {
                    lhs_index[dim] = 0;
                }
                if (rhs_padded[dim] == 1) {
                    rhs_index[dim] = 0;
                }
            }

            // Convert indices to flat offsets
            std::vector<int> lhs_original_index(
                lhs_index.end() - lhs_shape.size(),
                lhs_index.end()
            );
            std::vector<int> rhs_original_index(
                rhs_index.end() - rhs_shape.size(),
                rhs_index.end()
            );

            int lhs_flat = ravel_index(lhs_original_index, lhs_shape);
            int rhs_flat = ravel_index(rhs_original_index, rhs_shape);

            result.m_values[i] = m_values[lhs_flat] + rhs.m_values[rhs_flat];
        }

        // Gradient setup
        result.set_leaf(false);
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
        const auto& lhs_shape = shape();
        const auto& rhs_shape = rhs.shape();

        // Align shapes by padding with 1s on the left for RHS
        size_t max_dims = std::max(lhs_shape.size(), rhs_shape.size());
        std::vector<int> lhs_padded = lhs_shape;
        std::vector<int> rhs_padded = rhs_shape;

        while (lhs_padded.size() < max_dims) {
            lhs_padded.insert(lhs_padded.begin(), 1);
        }
        while (rhs_padded.size() < max_dims) {
            rhs_padded.insert(rhs_padded.begin(), 1);
        }

        // Check broadcast compatibility
        std::vector<int> result_shape(max_dims);
        for (size_t i = 0; i < max_dims; ++i) {
            if (lhs_padded[i] != rhs_padded[i] && lhs_padded[i] != 1 && rhs_padded[i] != 1) {
                throw std::invalid_argument("Shapes are not broadcastable");
            }
            result_shape[i] = std::max(lhs_padded[i], rhs_padded[i]);
        }

        // Compute broadcasted values
        Tensor<T> result(result_shape);
        for (int i = 0; i < result.size(); ++i) {
            // Calculate multi-dimensional indices for LHS and RHS
            auto lhs_index = unravel_index(i, result_shape);
            auto rhs_index = lhs_index;

            // Adjust indices for broadcasting
            for (size_t dim = 0; dim < max_dims; ++dim) {
                if (lhs_padded[dim] == 1) {
                    lhs_index[dim] = 0;
                }
                if (rhs_padded[dim] == 1) {
                    rhs_index[dim] = 0;
                }
            }

            // Convert indices to flat offsets
            std::vector<int> lhs_original_index(
                lhs_index.end() - lhs_shape.size(),
                lhs_index.end()
            );
            std::vector<int> rhs_original_index(
                rhs_index.end() - rhs_shape.size(),
                rhs_index.end()
            );

            int lhs_flat = ravel_index(lhs_original_index, lhs_shape);
            int rhs_flat = ravel_index(rhs_original_index, rhs_shape);

            result.m_values[i] = m_values[lhs_flat] + rhs.m_values[rhs_flat];
        }

        // Gradient setup
        result.set_leaf(false);
        return result;
    }

    Tensor<T> add(Tensor<T>& rhs) {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Error: shape mismatch in add");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
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
            throw std::invalid_argument("Error: shape mismatch in const add");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = m_values[i] + other.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // SUBTRACTION

    Tensor<T> operator-(Tensor<T>& rhs) {
        if (m_shape == rhs.m_shape) {
            return sub(rhs);
        } 

        // broadcast
        if (m_shape.size() > rhs.m_shape.size()) {
            return broadcast_add(-rhs);
        } else {
            return -rhs.broadcast_add(*this);
        }
    }

    Tensor<T> operator-(Tensor<T>&& rhs) {
        return *this - rhs;
    }

    // overload const without gradient graph creation
    Tensor<T> operator-(const Tensor<T>& other) const {
        if (m_shape == other.m_shape) {
            return sub(other);
        } 

        // broadcast
        if (m_shape.size() > other.m_shape.size()) {
            return broadcast_add(-other);
        } else {
            return -other.broadcast_add(*this);
        }
    }

    // SUBTRACTION
    Tensor<T> sub(Tensor<T>& rhs) {
        if (m_shape != rhs.m_shape) {
            throw std::invalid_argument("Error: shape mismatch in subtraction");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
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

    Tensor<T> sub(Tensor<T>&& rhs) {
        return *this - rhs;
    }

    // overload const without gradient graph creation
    Tensor<T> sub(const Tensor<T>& other) const {
        if (m_shape != other.m_shape) {
            throw std::invalid_argument("Error: shape mismatch in const subtraction");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
    Tensor<T> mul(Tensor<T>& other) {
        if (m_shape != other.m_shape) {
            throw std::invalid_argument("Error: shape mismatch in multiplication");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
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

    Tensor<T> mul(Tensor<T>&& other) {
        return this->mul(other);
    }

    // TODO: update backward, update operators, all math operations are done

    // overload const without gradient graph creation
    Tensor<T> mul(const Tensor<T>& other) const {
        if (m_shape != other.m_shape) {
            throw std::invalid_argument("Error: shape mismatch in const multiplication");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            
            new_values[i] = m_values[i] * other.m_values[i];
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // SCALAR MULTIPLICATION (single value tensor)
    Tensor<T> mul_scalar(Tensor<T>& other) {
        if (other.m_shape.size() != 1 || other.m_shape[0] != 1) {
            throw std::invalid_argument("Error: scalar multiplication requires a single value tensor");
        }
        T scalar = other.m_values[0];
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (size_t i = 0; i < m_values.size(); ++i) {
            new_values[i] = m_values[i] * scalar;
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        setup_accumulate_grad();
        other.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || other.requires_gradient();
        result.set_requires_gradient(requires_grad);

        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), other.get_grad_op()};
            std::vector<Tensor<T>> saved_values = {*this, other};
            result.set_grad_op(std::make_shared<MulBackScalar<T>>(saved_values, next_operators));
        }

        return result;
    }

    Tensor<T> mul_scalar(Tensor<T>&& other) {
        return mul_scalar(other);
    }

    Tensor<T> mul_scalar(const Tensor<T>& other) const {
        if (other.m_shape.size() != 1 || other.m_shape[0] != 1) {
            throw std::invalid_argument("Error: scalar multiplication requires a single value tensor");
        }
        T scalar = other.m_values[0];
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (size_t i = 0; i < m_values.size(); ++i) {
            new_values[i] = m_values[i] * scalar;
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;
    }

    Tensor<T> operator*(Tensor<T>& other) {
        if (m_shape == other.m_shape) {
            return mul(other);
        } else if (other.m_shape.size() == 1 && other.m_shape[0] == 1) {
            return mul_scalar(other);
        } else if (m_shape.size() == 1 && m_shape[0] == 1) {
            return other.mul_scalar(*this);
        } else {
            throw std::invalid_argument("Error: shape mismatch in multiplication");
        }
    }

    Tensor<T> operator*(Tensor<T>&& other) {
        return *this * other;
    }

    // overload const without gradient graph creation
    Tensor<T> operator*(const Tensor<T>& other) const {
        if (m_shape == other.m_shape) {
            return mul(other);
        } else if (other.m_shape.size() == 1 && other.m_shape[0] == 1) {
            return mul_scalar(other);
        } else if (m_shape.size() == 1 && m_shape[0] == 1) {
            return other.mul_scalar(*this);
        } else {
            throw std::invalid_argument("Error: shape mismatch in const multiplication");
        }
    }

    // ELEMENTWISE DIVISION
    Tensor<T> operator/(Tensor<T>& other) {
        if (m_shape != other.m_shape) {
            throw std::invalid_argument("Error: shape mismatch in division");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            if (other.m_values[i] == 0) {
                throw std::invalid_argument("Error: division by zero");
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
            throw std::invalid_argument("Error: shape mismatch in const division");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            if (other.m_values[i] == 0) {
                throw std::invalid_argument("Error: division by zero");
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
            throw std::invalid_argument("Error: shape mismatch in power");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
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
            throw std::invalid_argument("Error: shape mismatch in const power");
        }

        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            new_values[i] = std::pow(m_values[i], other.m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // SCALAR POWER

    Tensor<T> pow_scalar(Tensor<T>& other) {
        T exponent = other.m_values[0];
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (size_t i = 0; i < m_values.size(); ++i) {
            new_values[i] = std::pow(m_values[i], exponent);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        setup_accumulate_grad();
        other.setup_accumulate_grad();

        bool requires_grad = requires_gradient() || other.requires_gradient();
        result.set_requires_gradient(requires_grad);

        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op(), other.get_grad_op()};
            std::vector<Tensor<T>> saved_values = {*this, other};
            result.set_grad_op(std::make_shared<PowBackScalar<T>>(saved_values, next_operators));
        }

        return result;
    }

    Tensor<T> pow_scalar(Tensor<T>&& other) {
        return pow_scalar(other);
    }

    Tensor<T> pow_scalar(const Tensor<T>& other) const {
        T exponent = other.m_values[0];
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (size_t i = 0; i < m_values.size(); ++i) {
            new_values[i] = std::pow(m_values[i], exponent);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;
    }

    // OVERLOADING OPERATOR ^

    Tensor<T> operator^(Tensor<T>& other) {
        if (m_shape == other.m_shape) {
            // elementwise power
            return pow(other);
        } else if (other.m_shape.size() == 1 && other.m_shape[0] == 1) {
            // power with single value tensor
            return pow_scalar(other);
            // throw std::invalid_argument("Scalar power not implemented");
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
            return pow_scalar(other);
            throw std::invalid_argument("Scalar power not implemented");
        } else {
            throw std::invalid_argument("Incompatible shapes for power operation");
        }
    }

    // SQRT
    Tensor<T> sqrt() {
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] < 0) {
                throw std::invalid_argument("Error: square root of negative number");
            }
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
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] < 0) {
                throw std::invalid_argument("Error: square root of negative number");
            }
            new_values[i] = std::sqrt(m_values[i]);
        }

        Tensor<T> result(m_shape, new_values);
        result.set_leaf(false);

        return result;            
    }

    // LOG
    Tensor<T> log() {
        std::vector<T> new_values(m_values.size());
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] <= 0) {
                throw std::invalid_argument("Error: log of non-positive number");
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
        # pragma omp parallel for if(m_values.size() > 2000)
        for (int i = 0; i < m_values.size(); i++) {
            if (m_values[i] <= 0) {
                throw std::invalid_argument("Error: log of non-positive number");
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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
        # pragma omp parallel for if(m_values.size() > 2000)
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

        Tensor<T> result(sum);
        result.set_leaf(false);

        // figure out wether the parent needs AccumulateGrad
        setup_accumulate_grad();

        bool requires_grad = requires_gradient();
        result.set_requires_gradient(requires_grad);

        if (requires_grad) {
            std::vector<std::shared_ptr<Operator<T>>> next_operators = {get_grad_op()};

            result.set_grad_op(std::make_shared<SumBack<T>>(next_operators, m_shape));
        }

        return result;

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

        #pragma omp parallel for if(m_values.size() > 2000)
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

        # pragma omp parallel for if(m_values.size() > 2000)
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

    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
        if (tensor.values().size() == 1) {
            os << "Tensor(" << tensor.values()[0] << ")";
        } else {
            os << "Tensor(";
            for (int i = 0; i < tensor.shape().size(); i++) {
                os << tensor.shape()[i];
                if (i < tensor.shape().size() - 1) {
                    os << "x";
                }
            }
            os << ")";
        }
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>* tensor) {
        if (tensor->values().size() == 1) {
            os << "Tensor(" << tensor->values()[0] << ")";
        } else {
            os << "Tensor(";
            for (int i = 0; i < tensor->shape().size(); i++) {
                os << tensor->shape()[i];
                if (i < tensor->shape().size() - 1) {
                    os << "x";
                }
            }
            os << ")";
        }
        return os;
    }

    

};


// MATH OPERATIONS

template <typename T>
Tensor<T> pow(Tensor<T>& base, Tensor<T>& exponent) {
    return base^(exponent);
}

template <typename T>
Tensor<T> pow(Tensor<T>& base, Tensor<T>&& exponent) {
    return base^(exponent);
}

template <typename T>
Tensor<T> pow(Tensor<T>&& base, Tensor<T>& exponent) {
    return base^(exponent);
}

template <typename T>
Tensor<T> pow(Tensor<T>&& base, Tensor<T>&& exponent) {
    return base^(exponent);
}

template <typename T>
Tensor<T> pow(const Tensor<T>& base, const Tensor<T>& exponent) {
    return base^(exponent);
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
Tensor<T> log(const Tensor<T>& tensor) {
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
Tensor<T> exp(const Tensor<T>& tensor) {
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
Tensor<T> sqrt(const Tensor<T>& tensor) {
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
Tensor<T> abs(const Tensor<T>& tensor) {
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
Tensor<T> sin(const Tensor<T>& tensor) {
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
Tensor<T> cos(const Tensor<T>& tensor) {
    return tensor.cos();
}

template <typename T>
Tensor<T> tan(Tensor<T>& tensor) {
    return tensor.tan();
}

template <typename T>
Tensor<T> tan(Tensor<T>&& tensor) {
    return tensor.tan();
}

template <typename T>
Tensor<T> tan(const Tensor<T>& tensor) {
    return tensor.tan();
}

// activation functions
template <typename T>
Tensor<T> sigmoid(Tensor<T>& tensor) {
    return tensor.sigmoid();
}

template <typename T>
Tensor<T> sigmoid(Tensor<T>&& tensor) {
    return tensor.sigmoid();
}

template <typename T>
Tensor<T> sigmoid(const Tensor<T>& tensor) {
    return tensor.sigmoid();
}

template <typename T>
Tensor<T> relu(Tensor<T>& tensor) {
    return tensor.relu();
}

template <typename T>
Tensor<T> relu(Tensor<T>&& tensor) {
    return tensor.relu();
}

template <typename T>
Tensor<T> relu(const Tensor<T>& tensor) {
    return tensor.relu();
}

template <typename T>
Tensor<T> softmax(Tensor<T>& tensor) {
    return tensor.softmax();
}

template <typename T>
Tensor<T> softmax(Tensor<T>&& tensor) {
    return tensor.softmax();
}

template <typename T>
Tensor<T> softmax(const Tensor<T>& tensor) {
    return tensor.softmax();
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
      