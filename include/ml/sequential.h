#pragma once

#include <vector>
#include <iostream>
#include <string>

#include "math/number.h"
#include "math/tensor.h"
#include "ml/module.h"

namespace sdl {

template <typename T>
class Sequential : public Module<T> {
    private:

        std::vector<Module<T>*> modules;

        public:

            Sequential() {}

            Sequential(std::vector<Module<T>*> modules) : modules(modules) {}

            Sequential(std::initializer_list<Module<T>*> modules) : modules(modules) {}

            sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>& input) override {
                sdlm::Tensor<sdlm::Number<T>> output = input;
                for (auto& module : modules) {
                    output = module->forward(output);
                }
                return output;
            }

            sdlm::Tensor<sdlm::Number<T>> forward(sdlm::Tensor<sdlm::Number<T>>&& input) override {
                return forward(input);
            }


            std::vector<sdlm::Number<T>*> get_parameters() override {
                std::vector<sdlm::Number<T>*> parameters;
                for (auto& module : modules) {
                    auto module_parameters = module->get_parameters();
                    parameters.insert(parameters.end(), module_parameters.begin(), module_parameters.end());
                }
                return parameters;
            }

            void add(Module<T>* module) {
                modules.push_back(module);
            }

            void add(std::vector<Module<T>*> modules) {
                for (auto& module : modules) {
                    add(module);
                }
            }

            void print() {
                std::cout << "Sequential(";
                for (int i = 0; i < modules.size(); i++) {
                    std::cout << modules[i]->get_name();
                    if (i < modules.size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << ")" << std::endl;
            }

            std::string get_name() override {
                return "Sequential";
            }

            void print_pointers() {
                for (auto pointer : this->get_parameters()) {
                    std::cout << pointer << std::endl;
                }
            }

    };

    } // namespace sdl
