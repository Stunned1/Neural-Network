namespace NeuralNetwork
{
    public class LossFunctions
    {
        public static double MeanSquaredError(Matrix predicted, Matrix actual) {
            Matrix diff = Matrix.Subtract(predicted, actual);
            Matrix squared = Matrix.HadamardProduct(diff, diff);
            double sum = 0;
            for (int i = 0; i < squared.Rows; i++) {
                for (int j = 0; j < squared.Columns; j++) {
                    sum += squared[i, j];
                }
            }
            int numElements = predicted.Rows * predicted.Columns; //Columns is usually 1, but could be more than 1 for batch training etc; so we'll multiply by columns here.
            return sum / numElements;
        }

        public static Matrix MeanSquaredErrorDerivative(Matrix predicted, Matrix actual) {
            //shows how far off each prediction is from the target
            Matrix diff = Matrix.Subtract(predicted, actual);
            double scalar = 2.0 / (predicted.Rows * predicted.Columns); //normalizes gradient and comes from the derivative of the squared term
            Matrix gradient = Matrix.ScalarMultiply(diff, scalar);
            return gradient;
        }

        // Cross-Entropy Loss for classification (much better than MSE)
        public static double CrossEntropyLoss(Matrix predicted, Matrix actual) {
            double sum = 0;
            for (int i = 0; i < predicted.Rows; i++) {
                for (int j = 0; j < predicted.Columns; j++) {
                    double pred = Math.Max(1e-15, Math.Min(1 - 1e-15, predicted[i, j])); // Clamp to avoid log(0)
                    sum -= actual[i, j] * Math.Log(pred);
                }
            }
            return sum;
        }

        public static Matrix CrossEntropyLossDerivative(Matrix predicted, Matrix actual) {
            // For softmax + cross-entropy, the derivative is simply (predicted - actual)
            return Matrix.Subtract(predicted, actual);
        }
    }
}