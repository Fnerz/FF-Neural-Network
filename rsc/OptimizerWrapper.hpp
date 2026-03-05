#ifndef OPTIMIZER_WRAPPER_HPP
#define OPTIMIZER_WRAPPER_HPP

#include "LayerWrapper.hpp"
#include "Optimizer.hpp"

class optimizerWrapper
{
private:
    optimizerType optimizer_type;
    gradientDescent gradient_descent_obj;
    decayingLearningRate decaying_lr_obj;
    sgdMomentum sgd_momentum_obj;

public:
    optimizerWrapper();

    void optimize(std::vector<layerWrapper>& layers);
    void setGradientDescent(float learning_rate = 0.1);
    void setDecayingLearningRate(float learning_rate = 1, float decay = 0.01);
    void setSgdMomentum(float learning_rate, float momentum_coefficient, std::vector<int> layer_structure);
};

#endif