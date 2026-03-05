#include "LayerWrapper.hpp"

layerWrapper::layerWrapper(int n_inputs, int n_neurons, weightinitializationMethod weight_init_method, activationFunction activation_function)
    : layer_dense(n_inputs, n_neurons, weight_init_method),
      activation_layer(activation_function)
{
}

vec1d layerWrapper::forward(vec1d x)
{
    return activation_layer.forward(layer_dense.forward(x));
}

vec1d layerWrapper::backward(vec1d gradients)
{
    vec1d activation_layer_gradient = activation_layer.backward(gradients);
    vec1d layer_dense_gradient = layer_dense.backward(activation_layer_gradient);
    return layer_dense_gradient;
}
