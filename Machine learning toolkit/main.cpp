#include <iostream>
#include "Matrix.cpp"
#include <vector>
#include "Gradient decent.cpp"
#include "NeuralNetwork.cpp"

using namespace std;
int main()
{

   vector<vector<double>> inputs =
   {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
    };

    GradientDescent gd;
    gd.normalizeData(inputs);

    vector<vector<double>> targets =
    {
        {0.0}, {1.0}, {1.0}, {0.0}
    };

    NeuralNetwork nn(2, 8, 1, 0.5);
    nn.train(inputs, targets,1000);

    // Predict (forward propagate)
    for (const auto& input : inputs)
    {
        auto result = nn.forwardProp(input);
        cout<<"Sample->{"<<input[0]<<","<<input[1]<<"}" << "Prediction: " <<  (result.second[0]>0.5?1:0)<<endl;
    }



    return 0;
}

