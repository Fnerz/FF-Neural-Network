#include "ThreadDevice.hpp"

threadDevice::threadDevice()
    : layers({})
{
}

vec1d threadDevice::forward(vec1d x)
{
    vec1d last_out = x;

    for (auto& layer : this->layers)
    {
        last_out = layer.forward(last_out);
    }
    return last_out;
}

void threadDevice::backward(vec1d x, vec1d target, errorFunctionDerivative d_error_function)
{
    vec1d output = this->forward(x);

    vec1d output_errors = {};
    for (int i = 0; i < output.size(); i++)
    {
        float output_error = d_error_function(output[i], target[i]);
        output_errors.push_back(output_error);
    }

    vec1d last_gradient = output_errors;

    for (int i = this->layers.size() - 1; i > -1; i--)
    {
        last_gradient = this->layers[i].backward(last_gradient);
    }
}
