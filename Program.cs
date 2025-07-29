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
                (sigmoid, sigmoidDerivative) //hidden layer -> output layer (sigmoid for stability)
            };

            var network = new NeuralNetwork.NeuralNetwork(
                layerSizes, 
                0.0001, //much lower learning rate for stability
                LossFunctions.MeanSquaredError, //back to MSE for now
                LossFunctions.MeanSquaredErrorDerivative, //MSE derivative
                activations //activation functions
            );

            var (trainingData, trainingLabels) = LoadMNIST.LoadMnistData("data/mnist_train.csv", 3000); //reduced data size
            var (testData, testLabels) = LoadMNIST.LoadMnistData("data/mnist_test.csv", 500); //reduced test size
            
            Console.WriteLine($"Training on {trainingData.Count} samples, testing on {testData.Count} samples");
            Console.WriteLine("Starting training...");
            
            for (int epoch = 0; epoch < 10; epoch++) { //reduced epochs
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

            // Test prediction on first image
            Matrix testImage = testData[0];
            Matrix predictedOutput = network.Predict(testImage);
            int predictedClass = 0;
            double maxProb = predictedOutput[0, 0];
            for (int i = 1; i < predictedOutput.Rows; i++) {
                if (predictedOutput[i, 0] > maxProb) {
                    maxProb = predictedOutput[i, 0];
                    predictedClass = i;
                }
            }
            
            // Get actual class
            int actualClass = 0;
            for (int i = 0; i < testLabels[0].Rows; i++) {
                if (testLabels[0][i, 0] == 1.0) {
                    actualClass = i;
                    break;
                }
            }
            
            Console.WriteLine($"First test image - Predicted: {predictedClass}, Actual: {actualClass}, Confidence: {maxProb:P2}");
            
            network.Save("trained_model.txt");
            Console.WriteLine("Model saved to trained_model.txt");

        }
    }