namespace NeuralNetwork
{
    public class Layer
    {
        private Matrix weights;
        private Matrix biases;


        private Func<Matrix, Matrix> activationFunction;
        private Func<Matrix, Matrix> activationFunctionDerivative;

        //variables for backpropagation
        private Matrix input;
        private Matrix weightedSum;
        private Matrix activation;

        public Layer(int inputSize, int outputSize, 
                    Func<Matrix, Matrix> activationFunction, 
                    Func<Matrix, Matrix> activationFunctionDerivative)
        {
            this.weights = new Matrix(outputSize, inputSize);
            this.biases = new Matrix(outputSize, 1);
            this.weights.Randomize(-1, 1);
            this.biases.Randomize(-1, 1);
            this.activationFunction = activationFunction;
            this.activationFunctionDerivative = activationFunctionDerivative;
            this.input = new Matrix(inputSize, 1); //temporary variables for backpropagation
            this.weightedSum = new Matrix(outputSize, 1); //temporary variables for backpropagation
            this.activation = new Matrix(outputSize, 1); //temporary variables for backpropagation
        }

        public Matrix ForwardPass(Matrix input) {
            this.input = input;
            this.weightedSum = Matrix.DotProduct(weights, input);
            this.weightedSum = Matrix.Add(weightedSum, biases);
            this.activation = activationFunction(weightedSum);
            return activation;
        }

        public Matrix Backprop(Matrix error, double learningRate) {
            Matrix activationDerivative = activationFunctionDerivative(activation);
            Matrix delta = Matrix.HadamardProduct(error, activationDerivative);

            Matrix weightGradient = Matrix.DotProduct(delta, Matrix.Transpose(input));

            weights = Matrix.Subtract(weights, Matrix.ScalarMultiply(weightGradient, learningRate));
            biases = Matrix.Subtract(biases, Matrix.ScalarMultiply(delta, learningRate));

            Matrix errorToPreviousLayer = Matrix.DotProduct(delta, Matrix.Transpose(weights));
            return errorToPreviousLayer;
        }

        public void SetWeights(Matrix weights) {
            this.weights = weights;
        }

        public void SetBiases(Matrix biases) {
            this.biases = biases;
        }
        public Matrix GetWeights() {
            return weights;
        }
        public Matrix GetBiases() {
            return biases;
        }
    }
}