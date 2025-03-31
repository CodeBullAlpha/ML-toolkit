#include <iostream>
#include "Matrix.cpp"
#include <vector>
#include "Gradient decent.cpp"
#include "NeuralNetwork.cpp"

using namespace std;

int main() {

    ///////////////////////////////////////////////////////////////////////////
    //NEURAL NETWORK TRAINING
        // Sample data for training (let's assume it's a simple binary classification task)
    // Input features (e.g., two features per sample)
    vector<vector<double>> training_data =
    {
        {0.0, 0.0},  // Sample 1: Input (0, 0)
        {0.0, 1.0},  // Sample 2: Input (0, 1)
        {1.0, 0.0},  // Sample 3: Input (1, 0)
        {1.0, 1.0}   // Sample 4: Input (1, 1)
    };

    // Target labels for the samples
    vector<vector<int>> target_data =
    {
        {0},  // Sample 1: Label 0
        {1},  // Sample 2: Label 1
        {1},  // Sample 3: Label 1
        {0}   // Sample 4: Label 0
    };

    // Create a Neural Network with 2 input neurons, 3 hidden neurons, and 1 output neuron
    NeuralNetwork nn(2,5,3,0.0000001);  // 2 inputs, 5 hidden neurons, 3 output, learning rate = 0.1

    // Train the Neural Network with sample data
    int epochs = 100;  // Number of epochs (iterations)

    // Normalize before training
    //GradientDescent gd;
    //gd.normalizeData(training_data);

    nn.train(training_data, target_data, epochs);

    // Test the network on the training data after training
    cout << "\nTesting after training:" << endl;
    for (size_t i = 0; i < training_data.size(); ++i)
    {
        pair<Matrix,vector<double>> result = nn.forwardProp(training_data[i]);
        vector<double> output = result.second; // Access the second element (the vector)
        cout << "Input: (" << training_data[i][0] << ", " << training_data[i][1] << ") -> Prediction: ";
        for (double val : output) {
            cout << val << " ";  // Print the prediction (after applying softmax)
        }
        cout << endl;
    }


    return 0;
}
