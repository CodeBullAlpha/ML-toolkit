#include <iostream>
#include <vector>
#include <stdexcept>
#include "Matrix.cpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "Gradient decent.cpp"
#include <algorithm>

using namespace std;
class NeuralNetwork
{
private:
     Matrix weight1;
     Matrix weight2;
     Matrix bias1;
     Matrix bias2;
     double LearningRate;
     GradientDescent optimizer;

    //activation functions
    double Sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    double SigmoidDeri(double x)  // Derivative of Sigmoid
    {
        return x * (1 - x);
    }

    double LeakyReLU(double x)
    {
        return (x > 0) ? x : 0.01 * x;
    }

    double LeakyDeri(double x)  // Derivative of Leaky ReLU
    {
        return (x > 0) ? 1.0 : 0.01;
    }

    vector<double> softmax(vector<double> inputs)
    {
        std::vector<double> exp_values(inputs.size());
        double sum_exp = 0.0;

        // Find max input for numerical stability
        double max_input = *max_element(inputs.begin(), inputs.end());

        // Compute exponentials
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            exp_values[i] = exp(inputs[i] - max_input);
            sum_exp += exp_values[i];
        }

        // Normalize the exponentials (avoid division by zero)
        sum_exp = (sum_exp > 1e-8) ? sum_exp : 1e-8;

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            exp_values[i] /= sum_exp;
        }

        return exp_values;
    }

    double CrossEntropyLoss(const vector<double>& predicted, const vector<double>& actual)
    {
        double loss = 0.0;
        size_t n = predicted.size();  // Total samples

        for (size_t i = 0; i < n; ++i)
        {
            loss -= actual[i] * log(max(predicted[i], 1e-8))  // Handles y = 1
                  + (1 - actual[i]) * log(max(1 - predicted[i], 1e-8));  // Handles y = 0
        }

        return loss / n;  // Normalize over all samples
    }



public:
        NeuralNetwork(int input_size, int hidden_size, int output_size, double lr)
            : LearningRate(lr), optimizer() {

            srand(time(0));  // Seed random generator



            weight1 = Matrix(hidden_size, input_size);
            weight2 = Matrix(output_size, hidden_size);

            for (int r= 0; r< weight1.getRows(); ++r)
            {
                for (int c= 0; c< weight1.getCols(); ++c)
                {
                     weight1.setElements(r, c, ((double)rand() / RAND_MAX - 0.5)*0.1);  // Small random value between 0 and 1
                }
            }



            for (int r = 0;r< weight2.getRows(); ++r)
            {
                for (int c=0;c< weight2.getCols(); ++c)
                {
                    weight2.setElements(r, c, ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / hidden_size));
                }
            }

            // Small random biases
            bias1 = Matrix(hidden_size, 1);
            bias2 = Matrix(output_size, 1);

            for (int i = 0; i < bias1.getRows(); ++i) {
                bias1.setElements(i, 0, 0.01 * ((double)rand() / RAND_MAX - 0.5));
            }

            for (int i = 0; i < bias2.getRows(); ++i) {
                bias2.setElements(i, 0, 0.01 * ((double)rand() / RAND_MAX - 0.5));
            }
        }


        pair<Matrix, vector<double>> forwardProp(const vector<double>& inputs)
        {
            // Convert inputs to matrix format (input_size × 1)
            Matrix input_matrix(inputs.size(), 1);
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                input_matrix.setElements(i, 0, inputs[i]);
            }

            // Forward pass through first layer (input → hidden)
            Matrix hidden = weight1 * input_matrix;  // (hidden_size × 1)
            hidden = hidden + bias1;                 // (hidden_size × 1)

            // Apply Sigmoid activation to hidden layer
            for (int i = 0; i < hidden.getRows(); ++i)
            {
                hidden.setElements(i, 0, Sigmoid(hidden(i, 0)));
            }

            // Forward pass through second layer (hidden → output)

            Matrix output = weight2 * hidden;  // (output_size × 1)
            output = output + bias2;           // (output_size × 1)

            // Apply Softmax activation to output layer
            vector<double> output_vector(output.getRows());
            for (int i = 0; i < output.getRows(); ++i)
            {
                output_vector[i] = Sigmoid(output(i, 0));
            }

            return {hidden,output_vector};
        }

        void train(const vector<vector<double>>& training_data, const vector<vector<double>>& target_data, int epochs)
        {
            // Loop through epochs
            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                double trainingLoss = 0.0;

                // Loop through each sample in the training data
                for (size_t i = 0; i < training_data.size(); ++i)
                {

                    // Forward pass to get hidden and output
                    pair<Matrix,vector<double>> forward_result = forwardProp(training_data[i]);

                    Matrix hidden = forward_result.first;  // Hidden layer matrix

                    vector<double> outputs = forward_result.second;  // Output layer vector

                    // Compute loss (cross-entropy loss)
                    trainingLoss += CrossEntropyLoss(outputs, target_data[i]);

                    // Backpropagate through output layer (cross-entropy derivative)
                    vector<double> output_errors(outputs.size());
                    for (size_t r = 0; r < outputs.size(); ++r)
                    {
                        output_errors[r] = outputs[r] - target_data[i][r];  // Softmax + Cross-Entropy derivative
                    }

                    // Create output_error_matrix from output errors

                    Matrix output_error_matrix(output_errors.size(), 1);
                    for (int r = 0; r < output_error_matrix.getRows(); ++r)
                    {
                        output_error_matrix.setElements(r, 0, output_errors[r]);
                    }

                    // Backpropagate to the hidden layer (hidden -> output)
                    Matrix transposed_Hidden=hidden.getTransposed();
                    Matrix weight2_gradients = output_error_matrix * transposed_Hidden;
                    Matrix bias2_gradients = output_error_matrix;  // Bias gradients are simply the error vector

                    // Compute the errors for the hidden layer (input -> hidden)
                    Matrix transposed_weight2=weight2.getTransposed();
                    Matrix hidden_errors = transposed_weight2 * output_error_matrix;

                    // Apply Sigmoid derivative to hidden errors
                    for (int r = 0; r < hidden_errors.getRows(); ++r)
                    {
                        double hidden_val = hidden(r, 0);
                        double hidden_error_val = hidden_errors(r, 0);
                        double SigmoidDeri_val = SigmoidDeri(hidden_val);
                        double grad = hidden_error_val * SigmoidDeri_val;
                        hidden_errors.setElements(r, 0, grad);
                    }

                    // Compute gradients for the hidden layer (weight1, bias1)
                    Matrix input_matrix(training_data[i].size(), 1);  // Create a new input matrix
                    for (size_t j = 0; j < training_data[i].size(); ++j)
                    {
                        input_matrix.setElements(j, 0, training_data[i][j]);
                    }
                    Matrix input_transposed = input_matrix.getTransposed();
                    Matrix weight1_gradients = hidden_errors * input_transposed;  // input_matrix needs to be created
                    Matrix bias1_gradients = hidden_errors;  // Bias gradients are simply the hidden layer error vector

                    // Update weights and biases (hidden -> output) using the optimizer
                    optimizer.update(weight2, weight2_gradients, LearningRate);  // Update weight2
                    optimizer.update(bias2, bias2_gradients, LearningRate);      // Update bias2

                    // Update weights and biases (input -> hidden) using the optimizer
                    optimizer.update(weight1, weight1_gradients, LearningRate);  // Update weight1
                    optimizer.update(bias1, bias1_gradients, LearningRate);      // Update bias1


                }
                //cout<<"traning loss value="<<trainingLoss<<endl;
                // Print the training loss every epoch
                cout << "Epoch [" << epoch + 1 << "], Loss: " << trainingLoss / training_data.size() << endl;
            }
        }



};
