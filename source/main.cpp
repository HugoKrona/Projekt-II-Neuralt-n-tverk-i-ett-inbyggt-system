/**
 * @note The code looks really good overall. Really good work with implementing multiple
 *       hidden layers in the network. Great work! 
 * 
 *       The code is really easy to follow. Good work on naming variables, classes, 
 *       methods etc., as well as maintaining a consistent syntax. The inline comments
 *       are also really good; they're descriptive, while being concise.
 * 
 *       However, I have a few remarks, mostly with things for you to take into consideration.
 *       Please go through them and resolve them if you agree, otherwise remove them.
 * 
 *       As an additional note, I modified your .gitignore file so that files without an
 *       extension, except makefiles, are ignored.
 */

#include "button.h"
#include "led.h"

#include <iostream>
#include <vector>

#include "neural_network.h"

/********************************************************************************
 * @brief Trains a neural network to learn the XOR function.
 * 
 *        The network is then used to set the value of an LED by performing
 *        prediction based on input values from five buttons.
 ********************************************************************************/
int main() {
    // Create LED and button objects.
    rpi::Led led1{17};
    rpi::Button button1{27}, button2{22}, button3{23}, button4{24}, button5{25};

    // Create a neural network to learn the XOR function.
    constexpr std::size_t epochCount{110000U};
    constexpr double learningRate{0.01};

    // Define the input and reference sets for the XOR function.
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


    // Create a 5-5x5-1 neural network with hyperbolic tangent activation function.
    ml::NeuralNetwork network{5U, 5U, 5U, 1U, ml::ActFunc::Tanh};

    // Add the training data.
    network.addTrainingData(inputSets, referenceSets);

    // If training succeeded, print the results.
    if (network.train(epochCount, learningRate)) {
        network.printResults();
    }
    // Else print an error message and return.
    else {
        std::cout << "Failed to train the network!\n";
        return 1;
    }

    std::cout << "Training is done\n";

    std::vector<double> input(5, 0.0);

    while (1) {
        
        // Update input vector based on button states.
        std::vector<Button*> buttons{&button1, &button2, &button3, &button4, &button5};
        
        for (std::size_t i{}; i < buttons.size(); ++i) { 
            input[i] = buttons[i].isPressed() ? 1.0 : 0.0; 
        }

        
        // Predict output using the neural network.
        const auto output{network.predict(input)};

        // Enable LED based on the output.
        const auto enable{output[0] >= 0.5 ? true : false};
        led1.write(enable);
    }
    return 0;
}