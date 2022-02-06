using System;
using System.Collections.Generic;

namespace AI
{
    class Program
    {
        static void Main(string[] args)
        {
            List<int> dimensions = new() { 2, 2, 2 };
            double[,][] patterns =
            {
                { new double[] { 0, 1 }, new double[] { 1, 0 } },
                { new double[] { 1, 0 }, new double[] { 1, 0 } },
                { new double[] { 1, 1 }, new double[] { 0, 1 } },
                { new double[] { 0, 0 }, new double[] { 0, 1 } },
            };

            NeuralNetwork nn = new(dimensions, NeuralNetwork.ActivationTypes.SIGMOID);

            nn.BackPropagate(patterns, 2000, .4);

            Console.WriteLine("Outputs\n");

            for (int i = 0; i < patterns.GetLength(0); i++)
            {
                nn.Inputs = patterns[i, 0];
                nn.Update();
                nn.Update();
                Console.WriteLine("Input:");
                Console.WriteLine(nn.Inputs[0]);
                Console.WriteLine(nn.Inputs[1]);
                Console.WriteLine("Output:");
                Console.WriteLine(nn.Layers[nn.Layers.Count - 1].Outputs[0]);
                Console.WriteLine(nn.Layers[nn.Layers.Count - 1].Outputs[1]);
                Console.WriteLine();
            }
        }
    }
}