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
            //FIX: 7.29 Smaller weight initialization to prevent sigmoid saturation
            this.weights.Randomize(-0.1, 0.1);
            this.biases.Randomize(-0.1, 0.1);
            this.activationFunction = activationFunction;
            this.activationFunctionDerivative = activationFunctionDerivative;
        }

        public Matrix ForwardPass(Matrix input) {
            this.input = input;
            this.weightedSum = Matrix.DotProduct(weights, input);
            this.weightedSum = Matrix.Add(weightedSum, biases);
            this.activation = activationFunction(weightedSum);
            return activation;
        }

        public Matrix Backprop(Matrix error, double learningRate) {
            //FIX: 7.29 Fixed backpropagation - compute derivative on weightedSum, not activation
            Matrix activationDerivative = activationFunctionDerivative(weightedSum);
            Matrix delta = Matrix.HadamardProduct(error, activationDerivative);

            Matrix weightGradient = Matrix.DotProduct(delta, Matrix.Transpose(input));

            weights = Matrix.Subtract(weights, Matrix.ScalarMultiply(weightGradient, learningRate));
            biases = Matrix.Subtract(biases, Matrix.ScalarMultiply(delta, learningRate));

            Matrix errorToPreviousLayer = Matrix.DotProduct(Matrix.Transpose(weights), delta);
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