#include <iostream>
#include "Matrix.cpp"
#include <vector>
#include "Gradient decent.cpp"

using namespace std;

int main() {
    // Create a matrix for training data (X) and target values (y)
    Matrix X(5, 3); // Example: 5 samples with 2 features (+1 bias term)
    vector<double> y = {5, 7, 9, 11, 13}; // Example target values

    // Fill the matrix X (this can be your actual feature matrix)
    X.setElements(0, 0, 1); X.setElements(0, 1, 2); X.setElements(0, 2, 1); // Sample 1
    X.setElements(1, 0, 2); X.setElements(1, 1, 3); X.setElements(1, 2, 1); // Sample 2
    X.setElements(2, 0, 3); X.setElements(2, 1, 4); X.setElements(2, 2, 1); // Sample 3
    X.setElements(3, 0, 4); X.setElements(3, 1, 5); X.setElements(3, 2, 1); // Sample 4
    X.setElements(4, 0, 5); X.setElements(4, 1, 6); X.setElements(4, 2, 1); // Sample 5

    // Print the feature matrix X
    cout << "Feature Matrix (X):\n";
    X.print();

    // Initialize GradientDescent object with X
    GradientDescent gd;

    // Initial theta (randomly chosen)
    vector<double> theta = {0.1, 0.2, 0.3};  // Assuming 2 features + 1 bias term

    // Set learning rate and number of iterations
    double alpha = 0.01;
    int iterations = 1000;

    // Perform gradient descent to optimize theta
    vector<double> optimizedTheta = gd.gradientDescent(X, y, alpha, iterations);

    // Output the optimized theta values
    cout << "Optimized Theta:\n";
    for (double t : optimizedTheta) {
        cout << t << " ";
    }
    cout << endl;

    // Test the prediction with the optimized theta
    vector<double> sample = {1, 2}; // Example input (without bias term)
    double prediction = gd.prediction(sample, optimizedTheta);
    cout << "Prediction for sample {1, 2}: " << prediction << endl;

    return 0;
}
