#ifndef WEIGHT_INIT_METHODS_HPP
#define WEIGHT_INIT_METHODS_HPP

enum class weightinitializationMethod
{
    xavier,
    he
};

float HE_initialization(int n_inputs, int unused);
float xavier_initialization(int n_inputs, int n_neurons);

#endif