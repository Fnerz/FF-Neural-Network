#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include "./rsc/NeuralNetwork.hpp"

typedef std::vector<vec2d> vec3d;

void seed_rng() {
    std::srand(static_cast<unsigned int>(std::time(0)));
}

int random_int(int min, int max) {
    return std::rand() % (max - min + 1) + min;
}

float d_mse(float predicted, float target)
{
    return 2*(predicted-target);
}

float mse(vec1d predicted, vec1d target)
{
    float mean_error = 0;

    for (int i = 0; i < predicted.size(); i++)
    {
        mean_error += std::pow((predicted[i] - target[i]), 2);
    }

    return mean_error / predicted.size();
}

void printTrainingRes(vec1d input, vec1d target, vec1d output)
{
    std::cout << "input: ";
    for (auto i : input)
    {
        std::cout << i << " ";
    }
    std::cout << "target: ";
    for (auto t : target)
    {
        std::cout << t << " ";
    }
    std::cout << "output: ";
    for (auto o : output)
    {
        std::cout << o << " ";
    }
    std::cout << std::endl;
}

vec3d generate_nn_data(size_t num_samples) {
    vec3d data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-5.0, 5.0);
    
    for (size_t i = 0; i < num_samples; ++i) {
        float x = dist(gen);
        float y = dist(gen);
        float output = std::sin(x) * std::cos(y) + std::pow(x, 2) - std::pow(y, 2);
        data.push_back({{{x, y}}, {{output}}});
    }
    return data;
}

int main()
{
    neuralNetwork nn = neuralNetwork();
    nn.addLayer(2,4, weightinitializationMethod::xavier, activationFunction::Tanh);
    nn.addLayer(4,3, weightinitializationMethod::xavier, activationFunction::Tanh);
    nn.addLayer(3,1, weightinitializationMethod::xavier, activationFunction::Sigmoid);
    
    optimizerWrapper optimizer;
    optimizer.setGradientDescent(.1);

    nn.setOptimizer(optimizer);

    int training_epochs = 1000000;

    vec3d xor_data = {{{1,0}, {1}},
                    {{0,1}, {1}},
                    {{0,0}, {0}},
                    {{1,1}, {0}}};

    vec3d rand_data = {{{1, 0}, {0, 1}},
                    {{0, 1}, {1, 0}},
                    {{0, 0}, {1, 1}},
                    {{1, 1}, {0, 0}}};

    vec3d complex_data = generate_nn_data(100);
            
    vec3d data_set = xor_data;

    seed_rng();
    for (int k = 0; k < training_epochs; k++)
    {
        int i = random_int(0, data_set.size() -1);

        nn.backward(data_set[i][0], data_set[i][1], d_mse);
    }

    float total_error = 0;
    for (int i = 0; i < data_set.size(); i++)
    {
        vec1d output = nn.forward(data_set[i][0]);
        float error = mse(output, data_set[i][1]);
        total_error += error;
        std::cout << "error: " << error << "\n";

        // printTrainingRes(data_set[i][0], data_set[i][1], output);
    }
    std::cout << "mean error: " << total_error / data_set.size() << "\n";



    return 0;
}


/*

// -- todo -- //

- multithreading
    nn class -> vector of nn "kernels", with same functionalty as current nn class
    
- optimizer
    # goals
    - implement more optimizer
    - refactoring for speed

*/