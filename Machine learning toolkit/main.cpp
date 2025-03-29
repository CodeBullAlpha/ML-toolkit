#include <iostream>
#include "Matrix.cpp"

using namespace std;

int main()
{
    Matrix* objMatrix=  new Matrix(5,5);
    for(int r=0;r<objMatrix->getRows();r++)
    {
        for(int c=0;c<objMatrix->getCols();c++)
        {
            objMatrix->setElements(r,c,(r*c));
        }
    }
    objMatrix->transpose();
    objMatrix->print();
    delete objMatrix;

        // Create two matrices
    Matrix mat1(2, 3); // 2x3 matrix
    Matrix mat2(3, 2); // 3x2 matrix

    // Fill the matrices with some values
    mat1.setElements(0, 0,1); mat1.setElements(0, 1,2); mat1.setElements(0, 2,3);
    mat1.setElements(1, 0,4); mat1.setElements(1, 1,5); mat1.setElements(1, 2,6);

    mat2.setElements(0, 0,7); mat2.setElements(0, 1,8);
    mat2.setElements(1, 0,9); mat2.setElements(1, 1,10);
    mat2.setElements(2, 0,11); mat2.setElements(2, 1,12);

    // Print the matrices
    cout << "Matrix 1:\n";
    mat1.print();

    cout << "Matrix 2:\n";
    mat2.print();

    // Multiply the matrices
    Matrix matProduct = mat1 * mat2;

    // Print the result
    cout << "Product of Matrix 1 and Matrix 2:\n";
    matProduct.print();
    return 0;
}
