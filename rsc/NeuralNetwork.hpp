#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include "LayerWrapper.hpp"
#include "OptimizerWrapper.hpp"

typedef std::vector<float> vec1d;
typedef std::vector<vec1d> vec2d;

class neuralNetwork
{
private:
    std::vector<layerWrapper> layers;
    optimizerWrapper optimizer;

public:
    neuralNetwork();

    void setOptimizer(optimizerWrapper optimizer);
    void addLayer(int n_inputs, int n_neurons, weightinitializationMethod weight_init_method, activationFunction activation_function);
    vec1d forward(vec1d x);
    void backward(vec1d x, vec1d target, errorFunctionDerivative d_error_function);
};

#endif