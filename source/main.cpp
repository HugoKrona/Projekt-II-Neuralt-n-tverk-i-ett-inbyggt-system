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
int main() {
    rpi::Led led1{17};
    rpi::Button button1{27}, button2{22}, button3{23}, button4{24}, button5{25};

    constexpr std::size_t epochCount{110000U};
    constexpr double learningRate{0.01};

    const std::vector<std::vector<double>> inputSets{
        {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0, 1.0}, {0.0, 0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0, 1.0},
        {0.0, 0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0, 1.0}, {0.0, 1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0, 0.0}, {0.0, 1.0, 0.0, 1.0, 1.0},
        {0.0, 1.0, 1.0, 0.0, 0.0}, {0.0, 1.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 1.0, 1.0}, {1.0, 0.0, 1.0, 0.0, 0.0},
        {1.0, 0.0, 1.0, 0.0, 1.0}, {1.0, 0.0, 1.0, 1.0, 0.0}, {1.0, 0.0, 1.0, 1.0, 1.0},
        {1.0, 1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0, 1.0}, {1.0, 1.0, 0.0, 1.0, 0.0},
        {1.0, 1.0, 0.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0, 1.0, 0.0}, {1.0, 1.0, 1.0, 1.0, 1.0}};

    const std::vector<std::vector<double>> referenceSets{
        {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0},
        {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0}, {0.0}, {1.0},
        {1.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0}};

    // Create a 5-3-1 neural network, use tanh as activation for the hidden layer.
    ml::NeuralNetwork network{5U, 5U, 5U, 1U, ml::ActFunc::Tanh};

    // Add the training data.
    network.addTrainingData(inputSets, referenceSets);

    // If the training went well, print the result in the terminal.
    if (network.train(epochCount, learningRate)) {
        network.printResults();
    }
    // Else print an error message.
    else {
        std::cout << "Failed to train the network!\n";
    }

    std::cout << "Done\n";

    std::vector<double> input(5, 0.0);

    while (1) {
        // Update input vector based on button states.
        input[0] = button1.isPressed() ? 1.0 : 0.0;
        input[1] = button2.isPressed() ? 1.0 : 0.0;
        input[2] = button3.isPressed() ? 1.0 : 0.0;
        input[3] = button4.isPressed() ? 1.0 : 0.0;
        input[4] = button5.isPressed() ? 1.0 : 0.0;

        // Predict output using the neural network.
        std::vector<double> output = network.predict(input);

        const auto enable{output[0] >= 0.5 ? true : false};
        led1.write(enable);
    }

    return 0;
}