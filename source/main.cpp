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

/** @remark Please update the comment below! */
/********************************************************************************
 * @brief Example program demonstrating the Raspberry Pi GPIO drivers.
 ********************************************************************************/
#include "button.h"
#include "led.h"

#include <iostream>
#include <vector>

#include "neural_network.h"

/** @note I added the latter part of this comment block. Please remove this comment. */
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

    /** @remark Isn't this a 5-5x5-1 network, i.e. aren't you using five hidden layers? */
    // Create a 5-5-5-1 neural network with hyperbolic tangent activation function.
    ml::NeuralNetwork network{5U, 5U, 5U, 1U, ml::ActFunc::Tanh};

    // Add the training data.
    network.addTrainingData(inputSets, referenceSets);

    // If training succeeded, print the results.
    if (network.train(epochCount, learningRate)) {
        network.printResults();
    }
    // Else print an error message.
    else {
        std::cout << "Failed to train the network!\n";

        /** @remark Consider returning error code 1 here to terminate the program
         *          if the training failed. In the current implementation, the system
         *          will still work if the training of the network failed. */
    }

    std::cout << "Training is done\n";

    std::vector<double> input(5, 0.0);

    while (1) {
        /** @remark This implementation is good. However, I can't help to mention that if you store
         *          store the addresses of the buttons in a vector, you could replace the five lines
         *          below with a for loop:
         * 
         *          std::vector<Button*> buttons{&button1, &button2, &button3, &button4, &button5};
         *
         *          for (std::size_t i{}; i < buttons.size(); ++i) { 
         *              input[i] = buttons[i].isPressed() ? 1.0 : 0.0; 
         *          }
         * 
         *          On the other hand, implementing my suggestion would require slightly more memory,
         *          since we would have to add a vector holding five pointers. Also, even though
         *          the for loop solution is more elegant, the difference will not be that big, 
         *          since only five buttons are used. However, if something like 20 buttons were 
         *          used, I would definitely suggest using the pointers. 
         */
        // Update input vector based on button states.
        input[0] = button1.isPressed() ? 1.0 : 0.0;
        input[1] = button2.isPressed() ? 1.0 : 0.0;
        input[2] = button3.isPressed() ? 1.0 : 0.0;
        input[3] = button4.isPressed() ? 1.0 : 0.0;
        input[4] = button5.isPressed() ? 1.0 : 0.0;
        
        // Predict output using the neural network.
        /** @remark Consider using auto here, i.e. auto output{network.predict(input)}; */
        std::vector<double> output = network.predict(input);

        // Enable LED based on the output.
        const auto enable{output[0] >= 0.5 ? true : false};
        led1.write(enable);
    }
    return 0;
}