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

    }
}