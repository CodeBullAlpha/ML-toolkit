#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "Matrix.cpp"
class GradientDescent
{


public:

    // Given a new sample (without bias) and theta, evaluate predicted value
    double prediction(vector<double>& sample, vector<double>& theta)
    {
        double result = theta[0]; // Assuming the first element of theta is the bias term
        for (unsigned int c = 0; c < sample.size(); c++)
        {
            result += theta[c + 1] * sample[c]; // Assuming sample has the features
        }
        return result;
    }


    // Hypothesis function: computes predictions = X * theta (theta is a vector)
    vector<double> hypothesis(Matrix& X, const vector<double> &theta)
    {
        int numRows = X.getRows();
        int numCols = X.getCols();  // Should equal the number of parameters in theta
        vector<double> predictions(numRows, 0.0);

        for (int r = 0; r < numRows; r++)
        {
            for (int c = 0; c < numCols; c++)
            {
                predictions[r] += X(r, c) * theta[c]; // Ensure X(r, c) access is correct
            }
        }
        return predictions;
    }

    // Gradient Descent function to optimize theta
    vector<double> gradientDescent( Matrix& X,vector<double>& y, double alpha, int iterations)
    {
        int numRows = X.getRows();
        int numCols = X.getCols();  // Number of parameters (features + bias)
        vector<double> theta(numCols, 0.0);  // Initialize theta to 0

        for (int iter = 0; iter < iterations; iter++)
        {
            vector<double> h = hypothesis(X, theta);  // Get predicted values
            vector<double> gradient(numCols, 0.0);

            // Calculate gradient for each parameter
            for (int c = 0; c < numCols; c++)
            {
                double sumError = 0.0;
                for (int r = 0; r < numRows; r++)
                {
                    sumError += (h[r] - y[r]) * X(r, c);  // Sum of errors * feature value
                }
                gradient[c] = (alpha / numRows) * sumError;  // Gradient calculation
            }

            // Update theta based on gradient
            for (int c = 0; c < numCols; c++)
            {
                theta[c] -= gradient[c];  // Update parameters
            }
        }

        return theta;
    }
    // Update function to apply gradient descent to weights and biases
    void update(Matrix& param,Matrix& gradient, double learning_rate)
    {
        // Update the parameter (weight or bias) by subtracting the learning rate * gradient
        for (int r = 0; r < param.getRows(); ++r)
        {
            for (int c = 0; c < param.getCols(); ++c)
            {
                double result=param(r,c) - learning_rate * gradient(r,c);
                param.setElements(r, c,result);
            }
        }
    }
};

#endif // GRADIENTDESCENT_H
