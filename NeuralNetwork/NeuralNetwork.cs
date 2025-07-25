namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private int[] layerSize;
        private double[][][] weights;
        private double[][] biases;
        private double[][] activations;
        private double[] derivatives;

        public NeuralNetwork(int[] layerSize)
        {
            this.layerSize = layerSize;
        }
    }
}