#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "LayerWrapper.hpp"

class layerWrapper;

typedef std::vector<float> vec1d;
typedef std::vector<vec1d> vec2d;
typedef std::vector<vec2d> vec3d;

enum class optimizerType
{
    stochsticGradientDescent,
    decayingLearningRate,
    momentumOptimizer
};

class gradientDescent
{
private:
    float learning_rate;

public:
    gradientDescent(float learning_rate);
    void optimize(std::vector<layerWrapper>& layers);
};

class decayingLearningRate
{
private:
    float current_learning_rate;
    float learning_rate;
    float decay;
    float epoch_counter;

public:
    decayingLearningRate(float learning_rate, float decay);
    void optimize(std::vector<layerWrapper>& layers);
};

class sgdMomentum
{
private:
    float learning_rate;
    float momentum_coefficient;
    float epoch_counter;
    vec3d gradient_sums;
    vec2d gradient_bias_sums;

public:
    sgdMomentum(float learning_rate, float momentum_coefficient, std::vector<int> layer_structure);
    void optimize(std::vector<layerWrapper>& layers);
};

#endif