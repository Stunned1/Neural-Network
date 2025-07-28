namespace NeuralNetwork
using System;
{
    public static class ActivationFunctions
    {
        public static double ReLU(double x) {
            return x > 0 ? x : 0;
        }

        public static double ReLUDerivative(double x) {
            return x > 0 ? 1 : 0;
        }

        public static double Sigmoid(double x) { // The sigmoid funciion is represented as 1 / (1 + e^-x), which is used to normalize the output of a neuron to a value between 0 and 1
            return 1 / (1 + Math.Exp(-x));
        }

        // The derivative of the sigmoid function is Sigmoid(x) * (1 - Sigmoid(x)), which is used to calculate the gradient of the sigmoid function
        // parameter x is already assumed to be the output of the sigmoid function
        public static double SigmoidDerivative(double x) {
            return x * (1 - x);
        }
    }
}