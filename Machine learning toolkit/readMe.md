


```markdown
# Machine Learning Toolkit

A lightweight **C++ Machine Learning Toolkit** with built-in support for **matrix operations, linear regression, and neural networks**. Designed for speed, flexibility, and easy customization.

## Features

- **Matrix Class** – Efficient matrix operations for ML computations
- **Linear Regression** – Gradient Descent implementation
- **Neural Networks** – Supports forward & backpropagation
- **Activation Functions** – Sigmoid, ReLU, Tanh, and Softmax
- **Optimized Training** – Momentum-based gradient descent
- **Model Saving & Loading** – Store and reuse trained models

---



## Usage

### **1️ Import and Initialize**

```cpp
#include "Matrix.h"
#include "NeuralNetwork.h"
```

### **2️ Create and Train a Model**

```cpp
Matrix X(4, 2);  // Training data (features)
vector<double> y = {0, 1, 1, 0};  // Labels
NeuralNetwork nn(2, 4, 1);  // 2-input, 4-hidden, 1-output
nn.train(X, y, 0.01, 1000);
```

### **3️ Make Predictions**

```cpp
vector<double> sample = {1.0, 0.5};
double prediction = nn.predict(sample);
cout << "Predicted value: " << prediction << endl;
```

### **4️ Save & Load a Model**

```cpp
nn.saveModel("model.txt");
nn.loadModel("model.txt");
```

---

## Neural Network Class Prototype

```cpp
class NeuralNetwork {
private:
    Matrix weights1, weights2; // Weight matrices for layers
    vector<double> biases1, biases2; // Bias terms
    double learningRate;

public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);
    vector<double> forward(Matrix &X);
    void backward(Matrix &X, vector<double> &y);
    void train(Matrix &X, vector<double> &y, double alpha, int epochs);
    double predict(vector<double> &sample);
    void saveModel(const string &filename);
    void loadModel(const string &filename);
};
```

In LaTeX notation, a forward pass in a neural network can be represented as:

\[
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
\]

where:
- \( Z^{[l]} \) is the weighted input to layer \( l \)
- \( W^{[l]} \) is the weight matrix for layer \( l \)
- \( A^{[l-1]} \) is the activation from the previous layer
- \( b^{[l]} \) is the bias vector for layer \( l \)

The activation function, such as ReLU or Sigmoid, is then applied:

\[
A^{[l]} = g(Z^{[l]})
\]

---

## Contributing

Contributions are welcome! Feel free to fork this repository, submit issues, or open a pull request.

---

## License

This project is owned by Wesley
```

