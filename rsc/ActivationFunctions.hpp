#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <cmath>
#include <vector>

typedef std::vector<float> vec1d;
typedef std::vector<vec1d> vec2d;


enum class activationFunction
{
    Linear,
    Sigmoid,
    ReLu,
    Tanh
};


vec1d linear(vec1d x);
vec1d d_linear(vec1d x);


vec1d sigmoid(vec1d x);
vec1d d_sigmoid(vec1d x);



vec1d relu(vec1d x);
vec1d d_relu(vec1d x);


vec1d tanh_wrapper(vec1d x); // names tanh_wrapper because naming it "tanh" it caused trouble with "std::tanh"
vec1d d_tanh(vec1d x);





#endif