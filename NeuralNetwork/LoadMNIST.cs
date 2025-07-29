using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public static class LoadMNIST
    {
        public static (List<Matrix> inputs, List<Matrix> labels) LoadMnistData(string filePath, int maxSamples = -1)
        {
            var inputs = new List<Matrix>();
            var labels = new List<Matrix>();
            
            using (var reader = new StreamReader(filePath))
            {
                //FIX: 7.29 Fixed header detection logic - was backwards
                string firstLine = reader.ReadLine();
                bool hasHeader = !char.IsDigit(firstLine[0]);
                
                if (hasHeader)
                {
                    // Skip header line
                }
                else
                {
                    // No header, process the first line
                    ProcessLine(firstLine, inputs, labels);
                }
                
                int sampleCount = 0;
                string line;
                while ((line = reader.ReadLine()) != null && (maxSamples == -1 || sampleCount < maxSamples))
                {
                    ProcessLine(line, inputs, labels);
                    sampleCount++;
                }
            }
            
            return (inputs, labels);
        }
        
        private static void ProcessLine(string line, List<Matrix> inputs, List<Matrix> labels)
        {
            string[] values = line.Split(',');
            
            // First value is the label (0-9)
            int label = int.Parse(values[0]);
            
            // Create one-hot encoded label
            var labelMatrix = new Matrix(10, 1);
            labelMatrix[label, 0] = 1.0;
            labels.Add(labelMatrix);
            
            // Create input matrix (784 pixels)
            var inputMatrix = new Matrix(784, 1);
            for (int i = 0; i < 784; i++)
            {
                // Normalize pixel values from 0-255 to 0-1
                double pixelValue = double.Parse(values[i + 1]) / 255.0;
                inputMatrix[i, 0] = pixelValue;
            }
            inputs.Add(inputMatrix);
        }
        

    }
} 