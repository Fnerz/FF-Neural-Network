#include "NeuralNetwork.hpp"

neuralNetwork::neuralNetwork()
    : layers({}), optimizer()
{
}

void neuralNetwork::setOptimizer(optimizerWrapper optimizer)
{
    this->optimizer = optimizer;
}

void neuralNetwork::addLayer(int n_inputs, int n_neurons, weightinitializationMethod weight_init_method, activationFunction activation_function)
{
    layerWrapper lw = layerWrapper(n_inputs, n_neurons, weight_init_method, activation_function);
    this->layers.push_back(lw);
}

vec1d neuralNetwork::forward(vec1d x)
{
    vec1d last_out = x;

    for (auto& layer : layers)
    {
        last_out = layer.forward(last_out);
    }
    return last_out;
}

void neuralNetwork::backward(vec1d x, vec1d target, errorFunctionDerivative d_error_function)
{
    vec1d output = this->forward(x);

    vec1d output_errors = {};
    for (int i = 0; i < output.size(); i++)
    {
        float output_error = d_error_function(output[i], target[i]);
        output_errors.push_back(output_error);
    }

    vec1d last_gradient = output_errors;

    for (int i = this->layers.size() - 1; i > -1; i--)
    {
        last_gradient = this->layers[i].backward(last_gradient);
    }

    optimizer.optimize(this->layers);
}
