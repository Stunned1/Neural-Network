namespace NeuralNetwork
{
    public static class ActivationFunctions
    {
        public static double ReLU(double x) {
            return x > 0 ? x : 0;
        }
        public static double ReLUDerivative(double x) {
            return x > 0 ? 1 : 0;
        }
    }
}