#include <iostream>
#include <vector>
#include <stdexcept>

#ifndef MATRIX_H
#define MATRIX_H
using namespace std;
 class Matrix
{
private:
    //declaration of variables for matrix
    vector<vector<double>> data;
    int rows;
    int cols;

public:

    Matrix()
    {

        rows=10;
        cols=10;
        data.resize(rows);
        for(int r=0;r<rows;r++)
        {
                data[r].resize(cols,0.0);
        }
    }

    Matrix(int rowss,int colss)
    {
        rows=rowss;
        cols=colss;
        data.resize(rows);
        for(int r=0;r<rows;r++)
        {
                data[r].resize(cols,0.0);
        }
    }

    //copy contructor
    Matrix(const Matrix& otherMatrix): rows(otherMatrix.rows),cols(otherMatrix.cols)
    {
        data.resize(rows);
        for(int r=0;r<rows;r++)
        {
            data.resize(cols);
            for(int c=0;c<cols;c++)
            {
                data[r][c]=otherMatrix.data[r][c];
            }
        }
    }

    //operator overloading
    const double& operator()(int row,int col)
    {
        //check if coordinates are within bounds
        if(row<0 || row>=rows || col<0 || col>=cols)
        {
            throw std::out_of_range("indexes out of bounds");
        }
        return data[row][col];
    }

    const Matrix operator*( Matrix& otherMatrix)
    const{
        if(otherMatrix.rows!=cols)
        {
            throw out_of_range("col b !=  row a");
        }
        // Create a new matrix to store the result
        Matrix result(rows, otherMatrix.cols);

        // Perform matrix multiplication
        for (int r = 0; r < rows; r++) {
            for (int c2 = 0; c2 < otherMatrix.cols; c2++) {
                double sum = 0.0;
                for (int  c1= 0; c1 < cols; c1++) {
                    sum += data[r][c1] * otherMatrix(c1, c2); // Dot product of row r of A and column c of B
                }
                result.setElements(r,c2,sum);
            }
        }

        return result;
    }

    const Matrix operator+(Matrix& other)
    {

        if(other.rows!=rows || other.cols!=cols)
        {
            throw invalid_argument("Matrices have different dimensions");
        }

            Matrix result(5,5);
            for(int r=0;r<rows;r++)
            {
                for(int c=0;c<cols;c++)
                {
                    result.setElements(r,c,(data[r][c]+other(r,c)));
                }
            }
            return result;
    }




    const int getRows()
    {
        return rows;
    }

    const int getCols()
    {
        return cols;
    }

    void print()
    {
        for(int r=0;r<rows;r++)
        {
            for(int c=0;c<cols;c++)
            {
                cout<<data[r][c]<<" ";
            }
            cout<<"\n";
        }
    }

    //setter   for setting elements in matrix
    void setElements(int r,int c,double element)
    {
        data[r][c]=element;
    }

    void transpose()
    {
        // Create a temporary matrix to store the transposed data
        vector<vector<double>> TransposedData(cols, vector<double>(rows, 0.0));

        // Fill the transposed matrix
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                TransposedData[r][c] = data[c][r];
            }
        }

        // Update the original matrix
        data = TransposedData;
        swap(rows, cols); // Swap rows and columns
    }
};
#endif // MATRIX_H
