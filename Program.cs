using System;
using System.Collections.Generic;

namespace AI
{
    class Program
    {
        static void Main(string[] args)
        {
            List<int> dimensions = new() { 2, 3, 1 };
            double[,][] patterns =
            {
                { new double[] { 0.0, 1.0 }, new double[] { 1.0 } },
                { new double[] { 1.0, 0.0 }, new double[] { 1.0 } },
                { new double[] { 1.0, 1.0 }, new double[] { 0.0 } },
                { new double[] { 0.0, 0.0 }, new double[] { 0.0 } },
            };

            NeuralNetwork nn = new(dimensions, NeuralNetwork.ActivationTypes.TANH);

            nn.BackPropagate(patterns, 600, 0.4);

            Console.WriteLine("Outputs\n");

            for (int i = 0; i < patterns.GetLength(0); i++)
            {
                nn.SetInputs(patterns[i, 0]).Update();
                Console.WriteLine("Input:");
                Console.WriteLine(nn.GetInputs(0));
                Console.WriteLine(nn.GetInputs(1));
                Console.WriteLine("Output:");
                Console.WriteLine(nn.Layers[nn.Layers.Count - 1].Outputs[0]);
                Console.WriteLine();
            }
        }
    }
}
