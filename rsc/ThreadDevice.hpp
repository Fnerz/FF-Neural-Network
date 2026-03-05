#ifndef THREAD_DEVICE_HPP
#define THREAD_DEVICE_HPP

#include <vector>
#include "LayerWrapper.hpp"

typedef std::vector<float> vec1d;
typedef std::vector<vec1d> vec2d;
typedef std::vector<vec2d> vec3d;

class threadDevice
{
private:
    std::vector<layerWrapper> layers;

public:
    threadDevice();

    vec1d forward(vec1d x);
    void backward(vec1d x, vec1d target, errorFunctionDerivative d_error_function);
};

#endif