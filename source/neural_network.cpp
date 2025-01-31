/*******************************************************************************
 * @brief Implementation details of the ml::NeuralNetwork class.
 ******************************************************************************/
#include "neural_network.h"
#include "utils.h"

namespace ml {

// -----------------------------------------------------------------------------
NeuralNetwork::NeuralNetwork(const std::size_t inputCount, const std::size_t hiddenLayerCount,
                             const std::size_t hiddenNodeCount, const std::size_t outputCount,
                             const ActFunc actFuncHidden, const ActFunc actFuncOutput)
    : myHiddenLayers{DenseLayer(hiddenNodeCount, inputCount, actFuncHidden)},
      myOutputLayer{outputCount, hiddenNodeCount, actFuncOutput}, myTrainingInput{},
      myTrainingOutput{} {
    for (std::size_t i{1U}; i < hiddenLayerCount; ++i) {
        myHiddenLayers.push_back(DenseLayer(hiddenNodeCount, hiddenNodeCount, actFuncHidden));
    }
}

// -----------------------------------------------------------------------------
std::size_t NeuralNetwork::inputCount() const noexcept {
    // Input count = the weight count of the hidden layers
    int total = 0;
    for (const auto &layer : myHiddenLayers) {
        total += layer.weightCount();
    }

    return total;
}

// -----------------------------------------------------------------------------
std::size_t NeuralNetwork::outputCount() const noexcept {
    // Output count = the node count in the output layer.
    return myOutputLayer.nodeCount();
}

// -----------------------------------------------------------------------------
std::size_t NeuralNetwork::trainingSetCount() const noexcept {
    // Training set count = the size of the input and output vectors.
    return myTrainingInput.size();
}

// -----------------------------------------------------------------------------
bool NeuralNetwork::train(const std::size_t epochCount, const double learningRate) {
    // If the given parameters are invalid or training sets are missing, return false.
    if ((epochCount == 0U) || (learningRate <= 0.0) || myTrainingInput.empty()) {
        return false;
    }

    // Train the network given number of epochs.
    for (std::size_t i{}; i < epochCount; ++i) {
        // Train the network with each set one by one.
        for (std::size_t j{}; j < trainingSetCount(); ++j) {
            feedforward(myTrainingInput[j]);

            backpropagate(myTrainingOutput[j]);
            optimize(myTrainingInput[j], learningRate);
        }
    }
    // Indicate that training was performed successfully.
    return true;
}

// -----------------------------------------------------------------------------
const std::vector<double> &NeuralNetwork::predict(const std::vector<double> &input) {
    // Update the outputs of the nodes in all layers.
    feedforward(input);

    // Return the output of the output layer.
    return myOutputLayer.output();
}

// -----------------------------------------------------------------------------
bool NeuralNetwork::addTrainingData(const std::vector<std::vector<double>> &input,
                                    const std::vector<std::vector<double>> &output) {
    // Assign given data to our member variables.
    myTrainingInput = input;
    myTrainingOutput = output;

    // If there is a mismatch between the input and output, remove the superfluous value.
    if (input.size() != output.size()) {
        // Reduce the size of the larger vector to match the smaller one.
        const auto setCount{input.size() < output.size() ? input.size() : output.size()};
        myTrainingInput.resize(setCount);
        myTrainingOutput.resize(setCount);
    }
    // Return true if one or more training sets are stored.
    return trainingSetCount() > 0U;
}

// -----------------------------------------------------------------------------
void NeuralNetwork::printResults(std::ostream &printSource) {
    // Iterate through or training sets one by one and print the predicted value.
    for (const auto &input : myTrainingInput) {
        printSource << "Input: ";
        utils::vector::print(input, printSource, ", ");
        printSource << "prediction: ";
        utils::vector::print(predict(input), printSource);
    }
}

// -----------------------------------------------------------------------------
void NeuralNetwork::feedforward(const std::vector<double> &input) {
    // Perform feedforward for the hidden layers with given input.

    std::vector<double> currentInput = input;
    for (auto &layer : myHiddenLayers) {
        layer.feedforward(currentInput);
        currentInput = layer.output();
    }

    // Perform feedforward for the output layer, use the output of the hidden layer as input.
    myOutputLayer.feedforward(myHiddenLayers[myHiddenLayers.size() - 1].output());
}

// -----------------------------------------------------------------------------
void NeuralNetwork::backpropagate(const std::vector<double> &output) {
    // Index for the last layer.
    const auto last{static_cast<int>(myHiddenLayers.size() - 1U)};

    // Perform backpropagation for the output layer with given output.
    myOutputLayer.backpropagate(output);

    // Perform backpropagation for the last hidden layer, use values from the output layer.
    myHiddenLayers[last].backpropagate(myOutputLayer);

    // Perform backpropagation for the hidden layer, use values from the output layer.

    // Perform backpropagation for the hidden layers, use values from next layer.
    for (auto i = last - 1; i >= 0; --i) {
        myHiddenLayers[i].backpropagate(myHiddenLayers[i + 1]);
    }
}

// -----------------------------------------------------------------------------
void NeuralNetwork::optimize(const std::vector<double> &input, const double learningRate) {
    // Optimize the hidden layer with given input.
    // myHiddenLayers.optimize(input, learningRate);

    std::vector<double> currentInput = input;
    for (auto &layer : myHiddenLayers) {
        layer.optimize(currentInput, learningRate);
        currentInput = layer.output();
    }

    // Optimize the output layer with the output of the hidden layer as input.
    myOutputLayer.optimize(myHiddenLayers[myHiddenLayers.size() - 1].output(), learningRate);
}

} // namespace ml