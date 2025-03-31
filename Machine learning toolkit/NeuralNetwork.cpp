#include <iostream>
#include <vector>
#include <stdexcept>
#include "Matrix.cpp"
#include <cmath>
#include "Gradient decent.cpp"

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
         return (1.0/1.0-exp(-1));
     }

     double leaky_relu(double x)
     {
         return (x>=0)?x:0.01*x;
     }

     //derivative of leaky ReLu
     double LeakyDeri(double x)
     {
         return (x>=0)?x:0.01;

     }

    vector<double> softmax(vector<double> inputs)
    {
        std::vector<double> exp_values(inputs.size());
        double sum_exp = 0.0;
        //std::size_t can store the maximum size of a theoretically possible object of any type (including array)
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            exp_values[i] = exp(inputs[i]);
            sum_exp += exp_values[i];
        }

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            exp_values[i] /= sum_exp;
        }

        return exp_values;
    }

     double cross_entropy_loss(const vector<double>& predicted, const vector<int>& actual)
     {
         double loss=0.0;

         for(size_t i=0;i<predicted.size();i++)
         {
             if(actual[i]==1)
             {
                 loss-=log(predicted[i]-1e-9);
             }
         }
         return loss;
     }

public:

    NeuralNetwork(int input_size, int hidden_size, int output_size, double lr): LearningRate(lr),optimizer()
        {
            // Initialize weight1 (input -> hidden) and weight2 (hidden -> output) as Matrix
            weight1= Matrix(hidden_size,input_size);
            weight2=Matrix(output_size,hidden_size);
            // Initialize bias1 and bias2
            bias1= Matrix(hidden_size,1);
            bias2= Matrix(output_size,1);
        }

        pair<Matrix, std::vector<double>> forwardProp(const vector<double>& inputs)
        {
            // Convert inputs to matrix format (1xN)
            Matrix input_matrix(inputs.size(), 1);
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                input_matrix.setElements(i, 0, inputs[i]);
            }

            // Forward pass through first layer (input -> hidden)

            Matrix hidden = weight1 * input_matrix;
            hidden = hidden + bias1;

            // Apply Leaky ReLU activation to hidden layer
            for (int i = 0; i < hidden.getRows(); ++i)
            {
                hidden.setElements(i, 0, leaky_relu(hidden(i, 0)));
            }

            // Forward pass through second layer (hidden -> output)
            cout<<"we are here2 "<<hidden.getRows()<<" by "<<hidden.getCols()<<endl;
            cout<<"we are here2 "<<weight2.getRows()<<" by "<<weight2.getCols()<<endl;
            Matrix output = weight2 * hidden;//problem
            output = output + bias2;

            // Apply Softmax activation to output layer
            std::vector<double> output_vector(output.getRows());
            for (int i = 0; i < output.getRows(); ++i)
            {
                output_vector[i] = output(i, 0);
            }

            return {hidden, softmax(output_vector)};
        }

void train(const std::vector<std::vector<double>>& training_data, const std::vector<std::vector<int>>& target_data, int epochs)
{
    // Loop through epochs
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double trainingLoss = 0.0;

        // Loop through each sample in the training data
        for (size_t i = 0; i < training_data.size(); ++i)
        {
            cout<<"we are here1"<<endl;
            // Forward pass to get hidden and output
            pair<Matrix,vector<double>> forward_result = forwardProp(training_data[i]);

            Matrix hidden = forward_result.first;  // Hidden layer matrix
            vector<double> outputs = forward_result.second;  // Output layer vector

            // Compute loss (cross-entropy loss)
            trainingLoss += cross_entropy_loss(outputs, target_data[i]);

            // Backpropagate through output layer (cross-entropy derivative)
            std::vector<double> output_errors(outputs.size());
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
            hidden.transpose();
            Matrix weight2_gradients = output_error_matrix * hidden;
            Matrix bias2_gradients = output_error_matrix;  // Bias gradients are simply the error vector

            // Compute the errors for the hidden layer (input -> hidden)
            weight2.transpose();
            Matrix hidden_errors = weight2 * output_error_matrix;

            // Apply Leaky ReLU derivative to hidden errors
            for (int r = 0; r < hidden_errors.getRows(); ++r)
            {
                double result = hidden_errors(r, 0) * LeakyDeri(hidden(r, 0));  // Apply Leaky ReLU derivative
                hidden_errors.setElements(r, 0, result);
            }

            // Compute gradients for the hidden layer (weight1, bias1)
            Matrix input_matrix(training_data[i].size(), 1);  // Create a new input matrix
            for (size_t j = 0; j < training_data[i].size(); ++j)
            {
                input_matrix.setElements(j, 0, training_data[i][j]);
            }
            input_matrix.transpose();
            Matrix weight1_gradients = hidden_errors * input_matrix;  // input_matrix needs to be created
            Matrix bias1_gradients = hidden_errors;  // Bias gradients are simply the hidden layer error vector

            // Update weights and biases (hidden -> output) using the optimizer
            optimizer.update(weight2, weight2_gradients, LearningRate);  // Update weight2
            optimizer.update(bias2, bias2_gradients, LearningRate);      // Update bias2

            // Update weights and biases (input -> hidden) using the optimizer
            optimizer.update(weight1, weight1_gradients, LearningRate);  // Update weight1
            optimizer.update(bias1, bias1_gradients, LearningRate);      // Update bias1


        }

        // Print the training loss every epoch
        cout << "Epoch [" << epoch + 1 << "], Loss: " << trainingLoss / training_data.size() << endl;
    }
}



};
