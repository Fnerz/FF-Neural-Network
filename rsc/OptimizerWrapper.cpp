#include "OptimizerWrapper.hpp"

optimizerWrapper::optimizerWrapper()
    : optimizer_type(optimizerType::stochsticGradientDescent),
      gradient_descent_obj(1),
      decaying_lr_obj(1, 1),
      sgd_momentum_obj(1, 1, {})
{
}

void optimizerWrapper::optimize(std::vector<layerWrapper>& layers)
{
    if (optimizer_type == optimizerType::stochsticGradientDescent)
    {
        gradient_descent_obj.optimize(layers);
    }
    else if (optimizer_type == optimizerType::decayingLearningRate)
    {
        decaying_lr_obj.optimize(layers);
    }
}

void optimizerWrapper::setGradientDescent(float learning_rate)
{
    gradient_descent_obj = gradientDescent(learning_rate);
    optimizer_type = optimizerType::stochsticGradientDescent;
}

void optimizerWrapper::setDecayingLearningRate(float learning_rate, float decay)
{
    decaying_lr_obj = decayingLearningRate(learning_rate, decay);
    optimizer_type = optimizerType::decayingLearningRate;
}

void optimizerWrapper::setSgdMomentum(float learning_rate, float momentum_coefficient, std::vector<int> layer_structure)
{
    sgd_momentum_obj = sgdMomentum(learning_rate, momentum_coefficient, layer_structure);
}
