#include "WeightinitializationMethod.hpp"
#include <cmath>

float HE_initialization(int n_inputs, int unused)
{
    return std::sqrt(6 / n_inputs);
}

float xavier_initialization(int n_inputs, int n_neurons)
{
    return std::sqrt(6 / (n_inputs + n_neurons));
}
