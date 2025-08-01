using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;
namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private List<Layer> layers;
        private double learningRate;
        private Func<Matrix, Matrix, double> lossFunction;
        private Func<Matrix, Matrix, Matrix> lossFunctionDerivative;
        private List<(Func<Matrix, Matrix> activationFunction, Func<Matrix, Matrix> activationFunctionDerivative)> activationFunctions;


        public NeuralNetwork(List<int> layerSizes, 
                            double learningRate, 
                            Func<Matrix, Matrix, double> lossFunction, 
                            Func<Matrix, Matrix, Matrix> lossFunctionDerivative,
                            List<(Func<Matrix, Matrix> activationFunction, Func<Matrix, Matrix> activationFunctionDerivative)> activationFunctions
                            ) {
            this.layers = new List<Layer>();
            this.learningRate = learningRate;
            this.lossFunction = lossFunction;
            this.lossFunctionDerivative = lossFunctionDerivative;
            this.activationFunctions = activationFunctions;

            for (int i = 0; i < layerSizes.Count - 1; i++) {
                int inputSize = layerSizes[i];
                int outputSize = layerSizes[i + 1];
                layers.Add(new Layer(inputSize, outputSize, activationFunctions[i].activationFunction, activationFunctions[i].activationFunctionDerivative));
            }
        }

        private int ArgMax(Matrix matrix) {
            double max = matrix[0, 0];
            int maxIndex = 0;
            for (int i = 0; i < matrix.Rows; i++) {
                if (matrix[i, 0] > max) {
                    max = matrix[i, 0];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        public void Save(string path) {
            using (StreamWriter writer = new StreamWriter(path)) {
                writer.WriteLine(learningRate);
                writer.WriteLine(layers.Count);
                foreach (Layer layer in layers) {
                    Matrix.Serialize(layer.GetWeights(), writer);
                    Matrix.Serialize(layer.GetBiases(), writer);
                }
            }
        }

        public void Load(string path, List<(Func<Matrix, Matrix>, Func<Matrix, Matrix>)> activationFunctions) {
            layers.Clear();
            using (StreamReader reader = new StreamReader(path)) {
                learningRate = double.Parse(reader.ReadLine());
                int numLayers = int.Parse(reader.ReadLine());

                for (int i = 0; i < numLayers; i++) {
                    Matrix weights = Matrix.Deserialize(reader);
                    Matrix biases = Matrix.Deserialize(reader);

                    var (activation, activationPrime) = activationFunctions[i];
                    Layer layer = new Layer(weights.Columns, weights.Rows, activation, activationPrime); 
                    layer.SetWeights(weights);
                    layer.SetBiases(biases);
                    layers.Add(layer);
                }
            }
        }

        public Matrix ForwardPass(Matrix input) {
            Matrix currentInput = input;
            foreach (Layer layer in layers) {
                currentInput = layer.ForwardPass(currentInput);
            }
            return currentInput;
        }

        public Matrix Predict(Matrix input) { //for convenience; could be removed if not needed (same as ForwardPass). more could be added later too
            return ForwardPass(input);
        }

        public void Train(Matrix input, Matrix expectedOutput) {
            Matrix predictedOutput = ForwardPass(input);
            double loss = lossFunction(predictedOutput, expectedOutput);
            
            Matrix error = lossFunctionDerivative(predictedOutput, expectedOutput);
            for (int i = layers.Count - 1; i >= 0; i--) {
                error = layers[i].Backprop(error, learningRate);
            }
        }

        public void TrainBatch(List<Matrix> inputs, List<Matrix> expectedOutputs) { //simple batch training; could be improved with more sophisticated batching (e.g. stochastic gradient descent)
            for (int i = 0; i < inputs.Count; i++) {
                Train(inputs[i], expectedOutputs[i]);
            }
        }

        //FIX: 7.29 Fixed random number generator - use static instance to avoid same seed issue
        private static Random shuffleRandom = new Random();
        
        public static void Shuffle<T>(List<T> list) {
            int n = list.Count;
            while (n > 1) {
                n--;
                int k = shuffleRandom.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        //train for one epoch
        //could be improved with more sophisticated epoch training (e.g. mini-batch training)
        public double TrainEpoch(List<Matrix> inputs, List<Matrix> expectedOutputs) {

            if (inputs.Count != expectedOutputs.Count) {
                throw new Exception("Inputs and expected outputs must have the same number of elements");
            }

            List<int> indices = Enumerable.Range(0, inputs.Count).ToList(); //Generate list of indices to shuffle while preserving input-output pairing
            Shuffle(indices);

            List<Matrix> shuffledInputs = new List<Matrix>();
            List<Matrix> shuffledExpectedOutputs = new List<Matrix>();
            for (int i = 0; i < indices.Count; i++) {
                shuffledInputs.Add(inputs[indices[i]]);
                shuffledExpectedOutputs.Add(expectedOutputs[indices[i]]);
            }
            
            // Calculate average loss for the epoch
            double totalLoss = 0.0;
            for (int i = 0; i < shuffledInputs.Count; i++) {
                Matrix predictedOutput = ForwardPass(shuffledInputs[i]);
                totalLoss += lossFunction(predictedOutput, shuffledExpectedOutputs[i]);
            }
            double averageLoss = totalLoss / shuffledInputs.Count;
            
            // Train the batch
            TrainBatch(shuffledInputs, shuffledExpectedOutputs);
            
            return averageLoss;
        }

        public double EvaluateAccuracy(List<Matrix> inputs, List<Matrix> expectedOutputs) {
            int correct = 0;
            for (int i = 0; i < inputs.Count; i++) {
                Matrix predictedOutput = ForwardPass(inputs[i]);

                int predictedIndex = ArgMax(predictedOutput);
                int expectedIndex = ArgMax(expectedOutputs[i]);
                if (predictedIndex == expectedIndex) {
                    correct++;
                }
            }
            return (double)correct / inputs.Count;
        }

    }
}