#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <cmath>
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
                predictions[r] += X(r, c) * theta[c];
            }
        }
        return predictions;
    }

    // Normalize training data (Feature Scaling)
    void normalizeData(vector<vector<double>>& data)
    {
        int numSamples = data.size();
        int numFeatures = data[0].size();

        for (int c = 0; c < numFeatures; ++c)
        {  // Normalize each feature column
            double sum = 0.0, sumSq = 0.0;

            // Compute mean and standard deviation
            for (int r = 0; r < numSamples; ++r)
            {
                sum += data[r][c];
                sumSq += data[r][c] * data[r][c];
            }

            double mean = sum / numSamples;
            double variance = (sumSq / numSamples) - (mean * mean);
            double stdDev = sqrt(variance + 1e-8);  // Prevent division by zero

            // Normalize column values
            for (int r = 0; r < numSamples; ++r) {
                data[r][c] = (data[r][c] - mean) / stdDev;
            }
        }
    }



    void gradientDescent(Matrix& X, vector<double>& y, vector<double>& theta, double alpha, int iterations)
    {
        int numRows = X.getRows();
        int numCols = X.getCols();  // Number of parameters (features + bias)

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
    }

    void update(Matrix& param, Matrix& gradient, double learning_rate)
    {

        for (int r = 0; r < param.getRows(); ++r)
        {
            for (int c = 0; c < param.getCols(); ++c)
            {
                double grad_value = gradient(r, c);
                double clip_threshold = 1.0; // or another value determined by experimentation
                if (fabs(grad_value) > clip_threshold)
                {
                    grad_value = clip_threshold * ((grad_value > 0) ? 1 : -1);
                }
                double result = param(r, c) - learning_rate * grad_value;
                param.setElements(r, c, result);
            }
        }
    }

};

#endif // GRADIENTDESCENT_H
