using System;
namespace NeuralNetwork
{
    public class Matrix
    {
        private double[][] data;

        public Matrix(int rows, int cols)
        {
            data = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                data[i] = new double[cols];
            }
        }

        public int Rows { get { return data.Length; } }
        public int Columns { get { return data[0].Length; } }

        public double this[int row, int col] {
            get { return data[row][col]; }
            set { data[row][col] = value; }
        }

        // Adds two matrices together
        public static Matrix Add(Matrix a, Matrix b) {
            if (a.Rows != b.Rows || a.Columns != b.Columns) {
                throw new Exception("Matrices must have the same dimensions");
            }
            Matrix result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                    result[i, j] = a[i, j] + b[i, j];
            }
            return result;
        }

        public static Matrix Subtract(Matrix a, Matrix b) {
            if (a.Rows != b.Rows || a.Columns != b.Columns) {
                throw new Exception("Matrices must have the same dimensions");
            }
            Matrix result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
        }

        // Element-wise multiplication
        // used for backpropagation
        public static Matrix HadamardProduct(Matrix a, Matrix b) {
            if (a.Rows != b.Rows || a.Columns != b.Columns) {
                throw new Exception("Matrices must have the same dimensions");
            }
            Matrix result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                    result[i, j] = a[i, j] * b[i, j];
            }
            return result;
        }

        // Multiplies a matrix by a scalar
        // used for backpropagation
        public static Matrix ScalarMultiply(Matrix a, double scalar) {
            Matrix result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                    result[i, j] = a[i, j] * scalar;
            }
            return result;
        }

        // Multiplies two matrices together (Dot Product)
        // a = the weights (rows = output neurons, cols = input neurons)
        // b = the inputs (column vector (cols = 1))
        public static Matrix DotProduct(Matrix a, Matrix b) {
            if (a.Columns != b.Rows) {
                throw new Exception("Number of columns in A must match number of rows in B");
            }
            Matrix result = new Matrix(a.Rows, b.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    result[i, j] = 0;
                    for (int k = 0; k < a.Columns; k++)
                        result[i, j] += a[i, k] * b[k, j];
                }
            }
            return result;
        }

        public static Matrix Transpose(Matrix a) {
            Matrix result = new Matrix(a.Columns, a.Rows);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                    result[j, i] = a[i, j];
            }
            return result;
        }

        public static Matrix Map(Matrix a, Func<double, double> func) {
            Matrix result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                    result[i, j] = func(a[i, j]);
            }
            return result;
        }

        public static Matrix Sigmoid(Matrix a) {
            return Map(a, ActivationFunctions.Sigmoid);
        }

        public static Matrix ReLU(Matrix a) {
            return Map(a, ActivationFunctions.ReLU);
        }

        public static Matrix DerivativeSigmoid(Matrix a) {
            return Map(a, ActivationFunctions.SigmoidDerivative);
        }

        public static Matrix DerivativeReLU(Matrix a) {
            return Map(a, ActivationFunctions.ReLUDerivative);
        }

        public void Randomize(double min, double max) {
            Random random = new Random();
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this[i, j] = random.NextDouble() * (max - min) + min;
                }
            }
        }

        public static Matrix Clone(Matrix a) {
            Matrix result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                    result[i, j] = a[i, j];
            }
            return result;
        }


        // added ToString for debugging
        public override String ToString() {
            String result = "";
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                    result += this[i, j] + " ";
                result += "\n";
            }
            return result;
        }
    }
} 