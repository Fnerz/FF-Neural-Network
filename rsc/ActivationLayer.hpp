#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <vector>
#include <functional>
#include "ActivationFunctions.hpp"

typedef std::vector<float> vec1d;
typedef std::vector<vec1d> vec2d;

using errorFunctionDerivative = float(*)(float, float);

class activationLayer
{
private:
    vec1d input;
    vec1d output;

    std::function<vec1d(vec1d)> activation_function;
    std::function<vec1d(vec1d)> d_activation_function;

public:
    activationLayer(activationFunction activation_function);
    
    vec1d forward(vec1d input);
    vec1d backward(vec1d gradients);
};

#endif