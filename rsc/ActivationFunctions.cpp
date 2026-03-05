#include "ActivationFunctions.hpp"
#include <cmath>


vec1d linear(vec1d x)
{
    return x;
}

vec1d d_linear(vec1d x)
{
    vec1d ones(x.size(), 1);
    return ones;
}

vec1d sigmoid(vec1d x)
{
    vec1d output = {};
    for (auto x_item : x)
    {
        output.push_back(1 / (1 + exp(-x_item)));
    }
    return output;
}

vec1d d_sigmoid(vec1d x)
{
    vec1d output = {};
    for (auto x_item : x)
    {
        float s_x_item = 1 / (1 + exp(-x_item)); // prefix s for sigmoided
        output.push_back(s_x_item * (1 - s_x_item));
    }
    return output;
}

vec1d relu(vec1d x)
{
    vec1d output = {};
    for (auto x_item : x)
    {
        output.push_back((x_item > 0) ? x_item : 0);
    }
    return output;
}

vec1d d_relu(vec1d x)
{
    vec1d output = {};
    for (auto x_item : x)
    {
        output.push_back((x_item > 0) ? 1 : 0);
    }
    return output;
}

vec1d tanh_wrapper(vec1d x) // names tanh_wrapper because naming it "tanh" it caused trouble with "std::tanh"
{
    vec1d output = {};
    for (auto x_item : x)
    {
        output.push_back(std::tanh(x_item));
    }
    return output;
}

vec1d d_tanh(vec1d x)
{
    vec1d output = {};
    for (auto x_item : x)
    {
        output.push_back(1 / (std::pow(std::cosh(x_item), 2)));
    }
    return output;
}

