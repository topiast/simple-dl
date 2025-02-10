#pragma once

#include <vector>
#include <iostream>
#include <string>

#include "math/tensor.h"
#include "ml/module.h"

namespace sdl {

template <typename T>
class Sequential : public Module<T> {
    private:

        std::vector<Module<T>*> modules;

        public:

            Sequential() {}
            // TODO: add validation to the constructors
            Sequential(std::vector<Module<T>*> modules) : modules(modules) {}

            Sequential(std::initializer_list<Module<T>*> modules) : modules(modules) {}

            sdlm::Tensor<T> forward(sdlm::Tensor<T>& input) override {
                sdlm::Tensor<T> output = input;
                for (auto& module : modules) {
                    output = module->forward(output);
                }
                return output;
            }

            sdlm::Tensor<T> forward(sdlm::Tensor<T>&& input) override {
                return forward(input);
            }


            std::vector<sdlm::Tensor<T>*> get_parameters() override {
                std::vector<sdlm::Tensor<T>*> parameters;
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
