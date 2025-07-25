namespace NeuralNetwork
{
    public class Layer
    {
        private Matrix weights;
        private Matrix biases;


        private Func<Matrix, Matrix> activationFunction;
        private Func<Matrix, Matrix> activationFunctionDerivative;

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
        }

        public Matrix ForwardPass(Matrix input) {
            Matrix output = Matrix.DotProduct(input, weights);
            output = Matrix.Add(output, biases);
            output = activationFunction(output);
            return output;
        }


    }
}