#ifndef LAYER_WRAPPER_HPP
#define LAYER_WRAPPER_HPP

#include "LayerDense.hpp"
#include "ActivationLayer.hpp"
#include "ActivationFunctions.hpp"

typedef std::vector<float> vec1d;
typedef std::vector<vec1d> vec2d;

class layerWrapper
{
public:
    layerDense layer_dense;
    activationLayer activation_layer;

    layerWrapper(int n_inputs, int n_neurons, weightinitializationMethod weight_init_method, activationFunction activation_function);

    vec1d forward(vec1d x);
    vec1d backward(vec1d gradients);
};

#endif