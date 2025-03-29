#include "Matrix.cpp"

class GradientDecent
{
private:
    Matrix X;

public:
    GradientDecent(Matrix arr)
    {
        X=arr;
    }
    //given new sample(without bias) and theta evalute  predicted value
    double prediction(vector<double>& sample,vector<double>& theta)
    {
        //bias term
        double result=theta[0];
        for(int c=0;c<sample.size();c++)
        {
            result+=theta[c+1]*sample[c];
        }
        return result;
    }
};
