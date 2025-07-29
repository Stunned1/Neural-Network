using System;

namespace NeuralNetwork
{
    public static class ActivationFunctions
    {
        //FIX: 7.29 Fixed sigmoid to prevent numerical overflow
        public static double Sigmoid(double x)
        {
            // Clamp x to prevent overflow
            if (x > 500) return 1.0;
            if (x < -500) return 0.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double SigmoidDerivative(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid * (1.0 - sigmoid);
        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1.0 : 0.0;
        }

        // Softmax function for classification output
        public static Matrix Softmax(Matrix matrix)
        {
            // Find max value for numerical stability
            double maxVal = matrix[0, 0];
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    if (matrix[i, j] > maxVal)
                        maxVal = matrix[i, j];
                }
            }

            // Compute exp(x - max) for numerical stability
            var expMatrix = new Matrix(matrix.Rows, matrix.Columns);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    expMatrix[i, j] = Math.Exp(matrix[i, j] - maxVal);
                }
            }

            // Compute sum of exponentials
            double sum = 0;
            for (int i = 0; i < expMatrix.Rows; i++)
            {
                for (int j = 0; j < expMatrix.Columns; j++)
                {
                    sum += expMatrix[i, j];
                }
            }

            // Normalize to get probabilities
            var result = new Matrix(matrix.Rows, matrix.Columns);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[i, j] = expMatrix[i, j] / sum;
                }
            }

            return result;
        }

        // Softmax derivative (simplified for cross-entropy loss)
        public static Matrix SoftmaxDerivative(Matrix matrix)
        {
            // For softmax + cross-entropy, the derivative is handled in the loss function
            // This is a placeholder - the actual derivative is computed in the loss function
            return matrix; // This won't be used directly
        }
    }
}