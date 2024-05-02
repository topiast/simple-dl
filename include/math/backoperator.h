#pragma once 

#include "math/number.h"

#include <cmath>
#include <vector>
#include <iostream>

namespace sdlm {

template <typename T>
class Number;

template <typename T>
class Operator {
    public:
        virtual void evaluate(const T& x) const = 0;
        virtual void print() const = 0;
        virtual Operator<T>* clone() const = 0;
        virtual ~Operator() {}

        void debug_evalutate(const T& x) const {
            // check if member variables are set
            std::cout << "m_saved_values.size(): " << m_saved_values.size() << std::endl;
            std::cout << "m_next_operators.size(): " << m_next_operators.size() << std::endl;
            evaluate(x);
        }


        void set(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) {
            m_saved_values = saved_values;
            m_next_operators = next_operators;
        }

        Operator<T>& operator=(const Operator<T>& rhs) {
            m_saved_values = rhs.m_saved_values;
            m_next_operators = rhs.m_next_operators;

            return *this;
        }



    protected:
        std::vector<T> m_saved_values;
        std::vector<Operator<T>*> m_next_operators;

        // Operator() {}
        Operator(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : m_saved_values(saved_values), m_next_operators(next_operators) {}

        // copy constructor
        Operator(const Operator<T>& other) {
            m_saved_values = other.m_saved_values;
            m_next_operators = other.m_next_operators;
        }

};

template <typename T>
class AccumulateGrad: public Operator<T> { 
    private:
        Number<T>* m_variable;
    public:
        AccumulateGrad(Number<T>* variable) : Operator<T>(std::vector<T>(), std::vector<Operator<T>*>()), m_variable(variable) {}
        // AccumulateGrad() {}
        ~AccumulateGrad() {}
        // copy constructor
        AccumulateGrad(const AccumulateGrad<T>& other) : Operator<T>(other) {
            m_variable = other.m_variable;
        }

        void evaluate(const T& x) const override {
            m_variable->set_gradient(m_variable->gradient() + x);
            std::cout << "Setting grad for " << m_variable->value() << " to " << m_variable->gradient() << std::endl;
        }

        void print() const override {
            std::cout << "AccumulateGrad" << std::endl;
        }

        Operator<T>* clone() const override {
            return new AccumulateGrad<T>(*this);
        }
};

template <typename T>
class MulBack: public Operator<T> { 
    public:
        MulBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // MulBack() {}
        ~MulBack() {}

        void evaluate(const T& x) const override {
            auto saved_values = this->m_saved_values; 
            auto next_operators = this->m_next_operators; 

            auto n0 = saved_values[0];
            auto n1 = saved_values[1];

            next_operators[0]->evaluate(n1 * x);
            next_operators[1]->evaluate(n0 * x);
        }

        void print() const override {
            std::cout << "MulBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new MulBack<T>(*this);
        }
};

template <typename T>
class DivBack: public Operator<T> { 
    public:
        DivBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // DivBack() {}
        ~DivBack() {}

        void evaluate(const T& x) const override {
            auto saved_values = this->m_saved_values; 
            auto next_operators = this->m_next_operators; 

            auto n0 = saved_values[0]; // numerator
            auto n1 = saved_values[1]; // denominator

            next_operators[0]->evaluate(x / n1);
            next_operators[1]->evaluate(-x * n0 / (n1 * n1));
        }

        void print() const override {
            std::cout << "DivBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new DivBack<T>(*this);
        }
};

template <typename T>
class AddBack: public Operator<T> { 
    public:
        AddBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        AddBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~AddBack() {}

        void evaluate(const T& x) const override {


            this->m_next_operators[0]->evaluate(x);
            this->m_next_operators[1]->evaluate(x);
        }

        void print() const override {
            std::cout << "AddBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new AddBack<T>(*this);
        }
};

template <typename T>
class SubBack: public Operator<T> { 
    public:
        SubBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        SubBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~SubBack() {}

        void evaluate(const T& x) const override {
            auto next_operators = this->m_next_operators; 

            next_operators[0]->evaluate(x);
            next_operators[1]->evaluate(-x);
        }

        void print() const override {
            std::cout << "SubBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new SubBack<T>(*this);
        }
};

template <typename T>
class NegBack: public Operator<T> { 
    public:
        NegBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        NegBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~NegBack() {}

        void evaluate(const T& x) const override {
            this->m_next_operators[0]->evaluate(-x);
        }

        void print() const override {
            std::cout << "NegBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new NegBack<T>(*this);
        }
};

template <typename T>
class PowBack: public Operator<T> { 
    public:
        PowBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        PowBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~PowBack() {}

        void evaluate(const T& x) const override {
            auto saved_values = this->m_saved_values; 
            auto next_operators = this->m_next_operators; 

            auto n0 = saved_values[0]; // base
            auto n1 = saved_values[1]; // exponent

            next_operators[0]->evaluate(x * n1 * std::pow(n0, n1 - 1));
        }

        void print() const override {
            std::cout << "PowBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new PowBack<T>(*this);
        }
};

template <typename T>
class ExpBack: public Operator<T> { 
    public:
        ExpBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        ExpBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~ExpBack() {}

        void evaluate(const T& x) const override {
            auto next_operators = this->m_next_operators; 

            next_operators[0]->evaluate(x * std::exp(this->m_saved_values[0]));
        }

        void print() const override {
            std::cout << "ExpBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new ExpBack<T>(*this);
        }
};

template <typename T>
class LogBack: public Operator<T> { 
    public:
        LogBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        LogBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~LogBack() {}

        void evaluate(const T& x) const override {
            auto next_operators = this->m_next_operators; 

            next_operators[0]->evaluate(x / this->m_saved_values[0]);
        }

        void print() const override {
            std::cout << "LogBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new LogBack<T>(*this);
        }
};

template <typename T>
class SqrtBack: public Operator<T> { 
    public:
        SqrtBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        SqrtBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~SqrtBack() {}

        void evaluate(const T& x) const override {
            auto next_operators = this->m_next_operators; 

            next_operators[0]->evaluate(x / (2 * std::sqrt(this->m_saved_values[0])));
        }

        void print() const override {
            std::cout << "SqrtBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new SqrtBack<T>(*this);
        }
};

template <typename T>
class SinBack: public Operator<T> { 
    public:
        SinBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        SinBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~SinBack() {}

        void evaluate(const T& x) const override {
            auto next_operators = this->m_next_operators; 

            next_operators[0]->evaluate(x * std::cos(this->m_saved_values[0]));
        }

        void print() const override {
            std::cout << "SinBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new SinBack<T>(*this);
        }
};

template <typename T>
class CosBack: public Operator<T> { 
    public:
        CosBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        CosBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~CosBack() {}

        void evaluate(const T& x) const override {
            auto next_operators = this->m_next_operators; 

            next_operators[0]->evaluate(-x * std::sin(this->m_saved_values[0]));
        }

        void print() const override {
            std::cout << "CosBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new CosBack<T>(*this);
        }
};

template <typename T>
class TanBack: public Operator<T> { 
    public:
        TanBack(const std::vector<T>& saved_values, const std::vector<Operator<T>*>& next_operators) : Operator<T>(saved_values, next_operators) {}
        // without saved_values
        TanBack(const std::vector<Operator<T>*>& next_operators) : Operator<T>(std::vector<T>(), next_operators) {}
        ~TanBack() {}

        void evaluate(const T& x) const override {
            auto next_operators = this->m_next_operators; 

            next_operators[0]->evaluate(x / (std::cos(this->m_saved_values[0]) * std::cos(this->m_saved_values[0])));
        }

        void print() const override {
            std::cout << "TanBack" << std::endl;
        }

        Operator<T>* clone() const override {
            return new TanBack<T>(*this);
        }
};


} // namespace sdlm

