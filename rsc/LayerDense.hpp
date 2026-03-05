#ifndef LAYER_DENSE_HPP
#define LAYER_DENSE_HPP

#include <vector>
#include "WeightinitializationMethod.hpp"

typedef std::vector<float> vec1d;
typedef std::vector<vec1d> vec2d;

class layerDense
{
public:
    vec2d weights;
    vec1d biases;
    vec1d input;
    vec2d weight_gradients;
    vec1d bias_gradients;

    layerDense(int n_inputs, int n_neurons, weightinitializationMethod weight_init_method);

    vec1d forward(vec1d x);
    vec1d backward(vec1d gradients);

private:
    float random_float(float min, float max);
};

#endif