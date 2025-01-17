/********************************************************************************
 * @brief Example program demonstrating the Raspberry Pi GPIO drivers.
 ********************************************************************************/
#include "button.h"
#include "led.h"

#include <iostream>
#include <vector>

#include "neural_network.h"

/********************************************************************************
 * @brief Toggles a LED connected to pin 17 at press of a button connected
 *        to pin 27.
 ********************************************************************************/
int main()
{
    rpi::Led led1{17};
    rpi::Button button1{27};
    // while (1)
    // {
    //     if (button1.isEventDetected())
    //     {
    //         led1.toggle();
    //     }
    // }

    constexpr std::size_t epochCount{100000U};
    constexpr double learningRate{0.01};

    const std::vector<std::vector<double>> inputSets{
    {0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.0},
    {0.0, 0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0, 1.0},
    {0.0, 0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 1.0},
    {0.0, 0.0, 1.0, 1.0, 0.0},
    {0.0, 0.0, 1.0, 1.0, 1.0},
    {0.0, 1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0, 1.0},
    {0.0, 1.0, 0.0, 1.0, 0.0},
    {0.0, 1.0, 0.0, 1.0, 1.0},
    {0.0, 1.0, 1.0, 0.0, 0.0},
    {0.0, 1.0, 1.0, 0.0, 1.0},
    {0.0, 1.0, 1.0, 1.0, 0.0},
    {0.0, 1.0, 1.0, 1.0, 1.0},
    {1.0, 0.0, 0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0, 0.0, 1.0},
    {1.0, 0.0, 0.0, 1.0, 0.0},
    {1.0, 0.0, 0.0, 1.0, 1.0},
    {1.0, 0.0, 1.0, 0.0, 0.0},
    {1.0, 0.0, 1.0, 0.0, 1.0},
    {1.0, 0.0, 1.0, 1.0, 0.0},
    {1.0, 0.0, 1.0, 1.0, 1.0},
    {1.0, 1.0, 0.0, 0.0, 0.0},
    {1.0, 1.0, 0.0, 0.0, 1.0},
    {1.0, 1.0, 0.0, 1.0, 0.0},
    {1.0, 1.0, 0.0, 1.0, 1.0},
    {1.0, 1.0, 1.0, 0.0, 0.0},
    {1.0, 1.0, 1.0, 0.0, 1.0},
    {1.0, 1.0, 1.0, 1.0, 0.0},
    {1.0, 1.0, 1.0, 1.0, 1.0}
};

const std::vector<std::vector<double>> referenceSets{
    {0.0}, {1.0}, {1.0}, {0.0},
    {1.0}, {0.0}, {0.0}, {1.0},
    {1.0}, {0.0}, {0.0}, {1.0},
    {0.0}, {1.0}, {1.0}, {0.0},
    {1.0}, {0.0}, {0.0}, {1.0},
    {0.0}, {1.0}, {1.0}, {0.0},
    {0.0}, {1.0}, {1.0}, {0.0},
    {1.0}, {0.0}, {0.0}, {1.0}
};


    // Create a 5-3-1 neural network, use tanh as activation for the hidden layer.
    ml::NeuralNetwork network{5U, 3U, 1U, ml::ActFunc::Relu};

    // Add the training data.
    network.addTrainingData(inputSets, referenceSets);

    // If the training went well, print the result in the terminal.
    if (network.train(epochCount, learningRate))
    {
        network.printResults();
    }
    // Else print an error message.
    else
    {
        std::cout << "Failed to train the network!\n";
    }

    return 0;
}