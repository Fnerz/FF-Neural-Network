#include "ActivationLayer.hpp"
#include <cmath>

activationLayer::activationLayer(activationFunction activation_function)
    : activation_function(linear), d_activation_function(d_linear), input({}), output({})
{
    if (activation_function == activationFunction::Tanh)
    {
        this->activation_function = tanh_wrapper;
        this->d_activation_function = d_tanh;
    }
    else if (activation_function == activationFunction::Sigmoid)
    {
        this->activation_function = sigmoid;
        this->d_activation_function = d_sigmoid;
    }
    else if (activation_function == activationFunction::ReLu)
    {
        this->activation_function = relu;
        this->d_activation_function = d_relu;
    }
}

vec1d activationLayer::forward(vec1d input)
{
    this->input = input;
    vec1d output = {};

    output = this->activation_function(input);
    this->output = output;
    return output;
}

vec1d activationLayer::backward(vec1d gradients)
{
    vec1d proped_gradient = {};
    vec1d activation_derivatives = this->d_activation_function(this->input);

    for (int i = 0; i < gradients.size(); i++)
    {
        proped_gradient.push_back(gradients[i] * activation_derivatives[i]);
    }
    return proped_gradient;
}
