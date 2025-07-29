using System;
using System.Collections.Generic;
using NeuralNetwork;

    class Program {
        static void Main(string[] args) {
            Func<Matrix, Matrix> sigmoid = (Matrix m) => Matrix.Sigmoid(m);
            Func<Matrix, Matrix> sigmoidDerivative = (Matrix m) => Matrix.DerivativeSigmoid(m);
            Func<Matrix, Matrix> relu = (Matrix m) => Matrix.ReLU(m);
            Func<Matrix, Matrix> reluDerivative = (Matrix m) => Matrix.DerivativeReLU(m);
            
            List<int> layerSizes = new List<int> { 784, 128, 64, 10 };
            var activations = new List<(Func<Matrix, Matrix>, Func<Matrix, Matrix>)> { 
                (relu, reluDerivative), //input layer -> hidden layer
                (relu, reluDerivative), //hidden layer -> hidden layer
                (sigmoid, sigmoidDerivative) //hidden layer -> output layer
            };

            var network = new NeuralNetwork.NeuralNetwork(
                layerSizes, 
                0.01, //learning rate
                LossFunctions.MeanSquaredError, //loss function
                LossFunctions.MeanSquaredErrorDerivative, //loss function derivative
                activations //activation functions
            );

            var (trainingData, trainingLabels) = LoadMNIST.LoadMnistData("data/mnist_train.csv", 1000);
            var (testData, testLabels) = LoadMNIST.LoadMnistData("data/mnist_test.csv", 200);
            
            Console.WriteLine("Starting training...");
            for (int epoch = 0; epoch < 10; epoch++) {
                Console.WriteLine($"Epoch {epoch + 1}/10");
                network.TrainEpoch(trainingData, trainingLabels);
                
                // Evaluate on test data every 2 epochs
                if ((epoch + 1) % 2 == 0) {
                    double accuracy = network.EvaluateAccuracy(testData, testLabels);
                    Console.WriteLine($"Test Accuracy: {accuracy:P2}");
                }
            }

            // Final evaluation
            double finalAccuracy = network.EvaluateAccuracy(testData, testLabels);
            Console.WriteLine($"Final Test Accuracy: {finalAccuracy:P2}");

            Matrix testImage = testData[0];
            Matrix predictedOutput = network.Predict(testImage);
            Console.WriteLine($"First test image prediction: {predictedOutput}");
            
            network.Save("trained_model.txt");

        }
    }