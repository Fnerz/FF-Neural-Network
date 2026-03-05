#include "LayerDense.hpp"
#include <random>

float layerDense::random_float(float min, float max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());  // Mersenne Twister PRNG
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

layerDense::layerDense(int n_inputs, int n_neurons, weightinitializationMethod weight_init_method)
    : weights({}), biases({}), input({}), weight_gradients({}), bias_gradients({})
{
    float weight_min_max = 0;
    if (weight_init_method == weightinitializationMethod::he)
    {
        weight_min_max = HE_initialization(n_inputs, n_neurons);
    }
    else if (weight_init_method == weightinitializationMethod::xavier)
    {
        weight_min_max = xavier_initialization(n_inputs, n_neurons);
    }

    for (int n = 0; n < n_neurons; n++)
    {
        biases.push_back(0);
        vec1d neuron = {};
        for (int w = 0; w < n_inputs; w++)
        {
            neuron.push_back(random_float(-weight_min_max, weight_min_max));
        }
        weights.push_back(neuron);
    }
}

vec1d layerDense::forward(vec1d x)
{
    this->input = x;
    vec1d output = {};
    for (int i = 0; i < this->weights.size(); i++)
    {
        float neuron_output = this->biases[i];
        for (int j = 0; j < this->weights[i].size(); j++)
        {
            neuron_output += this->weights[i][j] * x[j];
        }
        output.push_back(neuron_output);
    }
    return output;
}

vec1d layerDense::backward(vec1d gradients)
{
    vec1d proped_gradient(this->weights[0].size(), 0);

    this->weight_gradients = {};
    this->bias_gradients = {};

    for (int i = 0; i < gradients.size(); i++)
    {
        float error = gradients[i];
        vec1d neuron_gradients = {};

        for (int j = 0; j < this->weights[i].size(); j++)
        {
            proped_gradient[j] += error * this->weights[i][j];

            float dE_dw = error * this->input[j]; // derivative of the error with respect to weight

            neuron_gradients.push_back(dE_dw);
        }
        weight_gradients.push_back(neuron_gradients);
        bias_gradients.push_back(error);
    }

    return proped_gradient;
}
