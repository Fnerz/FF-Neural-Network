#include "Optimizer.hpp"

// ========== gradientDescent ==========

gradientDescent::gradientDescent(float learning_rate)
    : learning_rate(learning_rate)
{
}

void gradientDescent::optimize(std::vector<layerWrapper>& layers)
{
    for (auto& layer_wrapper : layers)
    {
        for (int n = 0; n < layer_wrapper.layer_dense.weights.size(); n++)
        {
            for (int w = 0; w < layer_wrapper.layer_dense.weights[n].size(); w++)
            {
                layer_wrapper.layer_dense.weights[n][w] -= layer_wrapper.layer_dense.weight_gradients[n][w] * learning_rate;
            }
            layer_wrapper.layer_dense.biases[n] -= layer_wrapper.layer_dense.bias_gradients[n] * learning_rate;
        }
    }
}

// ========== decayingLearningRate ==========

decayingLearningRate::decayingLearningRate(float learning_rate, float decay)
    : learning_rate(learning_rate), decay(decay), current_learning_rate(0), epoch_counter(0)
{
}

void decayingLearningRate::optimize(std::vector<layerWrapper>& layers)
{
    current_learning_rate = learning_rate * (1 / (1 + decay * epoch_counter));
    for (auto& layer : layers)
    {
        for (int n = 0; n < layer.layer_dense.weights.size(); n++)
        {
            for (int w = 0; w < layer.layer_dense.weights[n].size(); w++)
            {
                layer.layer_dense.weights[n][w] -= layer.layer_dense.weight_gradients[n][w] * learning_rate;
            }
            layer.layer_dense.biases[n] -= layer.layer_dense.bias_gradients[n] * learning_rate;
        }
    }
    epoch_counter += 1;
}

// ========== sgdMomentum ==========

sgdMomentum::sgdMomentum(float learning_rate, float momentum_coefficient, std::vector<int> layer_structure)
    : learning_rate(learning_rate), momentum_coefficient(momentum_coefficient), epoch_counter(1),
      gradient_sums({}), gradient_bias_sums({})
{
    for (int i = 1; i < layer_structure.size(); i++)
    {
        vec2d layer = {};
        for (int n = 0; n < layer_structure[i]; n++)
        {
            vec1d neuron = {};
            for (int w = 0; w < layer_structure[i - 1]; w++)
            {
                neuron.push_back(0);
            }
            layer.push_back(neuron);
        }
        gradient_sums.push_back(layer);
    }
}

void sgdMomentum::optimize(std::vector<layerWrapper>& layers)
{
    for (int l = 0; l < layers.size(); l++)
    {
        for (int n = 0; n < layers[l].layer_dense.weights.size(); n++)
        {
            for (int w = 0; w < layers[l].layer_dense.weights[n].size(); w++)
            {
                float gradient = layers[l].layer_dense.weight_gradients[n][w];
                gradient_sums[l][n][w] = momentum_coefficient * gradient_sums[l][n][w] + gradient;

                float momentum = gradient_sums[l][n][w] / epoch_counter;

                layers[l].layer_dense.weights[n][w] -= momentum * learning_rate;
            }
            layers[l].layer_dense.biases[n] -= layers[l].layer_dense.bias_gradients[n] * learning_rate;
        }
    }
    epoch_counter += 1;
}
