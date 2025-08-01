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
                0.001, 
                LossFunctions.MeanSquaredError, 
                LossFunctions.MeanSquaredErrorDerivative, 
                activations 
            );

            var (trainingData, trainingLabels) = LoadMNIST.LoadMnistData("data/mnist_train.csv"); 
            var (testData, testLabels) = LoadMNIST.LoadMnistData("data/mnist_test.csv"); 
            
            Console.WriteLine($"Training on {trainingData.Count} samples, testing on {testData.Count} samples");
            Console.WriteLine("Starting training...");
            
            // Performance tracking
            var trainingAccuracies = new List<double>();
            var testAccuracies = new List<double>();
            var losses = new List<double>();
            
            for (int epoch = 0; epoch < 20; epoch++) { 
                Console.WriteLine($"\nEpoch {epoch + 1}/20");
                
                // Train and get loss
                double epochLoss = network.TrainEpoch(trainingData, trainingLabels);
                losses.Add(epochLoss);
                
                // Calculate training accuracy
                double trainAccuracy = network.EvaluateAccuracy(trainingData, trainingLabels);
                trainingAccuracies.Add(trainAccuracy);
                
                // Calculate test accuracy
                double testAccuracy = network.EvaluateAccuracy(testData, testLabels);
                testAccuracies.Add(testAccuracy);
                
                // Live performance visualization
                DisplayPerformanceVisualization(epoch + 1, 20, trainAccuracy, testAccuracy, epochLoss, trainingAccuracies, testAccuracies, losses);
            }

            // Final evaluation
            double finalAccuracy = network.EvaluateAccuracy(testData, testLabels);
            Console.WriteLine($"\nFinal Test Accuracy: {finalAccuracy:P2}");

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
        
        static void DisplayPerformanceVisualization(int currentEpoch, int totalEpochs, double trainAcc, double testAcc, double loss, List<double> trainHistory, List<double> testHistory, List<double> lossHistory)
        {
            // Clear previous lines
            Console.SetCursorPosition(0, Console.CursorTop - 1);
            Console.Write(new string(' ', Console.WindowWidth - 1));
            Console.SetCursorPosition(0, Console.CursorTop - 1);
            
            // Progress bar
            int progressBarWidth = 30;
            int filledWidth = (int)((double)currentEpoch / totalEpochs * progressBarWidth);
            string progressBar = "[" + new string('█', filledWidth) + new string('░', progressBarWidth - filledWidth) + "]";
            
            Console.WriteLine($"Progress: {progressBar} {currentEpoch}/{totalEpochs}");
            Console.WriteLine($"Loss: {loss:F4} | Train Acc: {trainAcc:P2} | Test Acc: {testAcc:P2}");
            
            // Simple trend indicators
            if (trainHistory.Count > 1)
            {
                string trainTrend = trainAcc > trainHistory[trainHistory.Count - 2] ? "↗" : "↘";
                string testTrend = testAcc > testHistory[testHistory.Count - 2] ? "↗" : "↘";
                Console.WriteLine($"Trends: Train {trainTrend} | Test {testTrend}");
            }
            
            // Performance summary
            Console.WriteLine($"Best Test Accuracy: {testHistory.Max():P2} | Best Train Accuracy: {trainHistory.Max():P2}");
        }
    }