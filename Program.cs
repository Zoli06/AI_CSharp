using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using MathNet.Numerics.LinearAlgebra;

namespace AI
{
    class Program
    {
        static void Main(string[] args)
        {
            VectorBuilder<double> _v = Vector<double>.Build;
            List<int> structure = new() { 2, 2, 4 };

            // First output: xor
            // Second output: xnor
            // Third output: and
            // Fourth output: or
            Vector<double>[,] patterns =
            {
                { _v.DenseOfArray(new double[] { 0, 1 }), _v.DenseOfArray(new double[] { 1, 0, 0, 1 }) },
                { _v.DenseOfArray(new double[] { 1, 0 }), _v.DenseOfArray(new double[] { 1, 0, 0, 1 }) },
                { _v.DenseOfArray(new double[] { 1, 1 }), _v.DenseOfArray(new double[] { 0, 1, 1, 1 }) },
                { _v.DenseOfArray(new double[] { 0, 0 }), _v.DenseOfArray(new double[] { 0, 1, 0, 0 }) },
            };

            //NeuralNetwork nn = new("D:/Users/zolix/Downloads/export.nns");
            NeuralNetwork nn = new(structure, NeuralNetwork.ActivationType.TANH);

            nn.BackPropagate(patterns, 2000, .25);

            Console.WriteLine("Outputs\n");

            for (int i = 0; i < patterns.GetLength(0); i++)
            {
                nn.Update(patterns[i, 0]);
                Console.WriteLine("Input:");
                for (int j = 0; j < patterns[i, 0].Count; j++)
                {
                    Console.WriteLine(patterns[i, 0][j]);
                }
                Console.WriteLine("Output:");
                for (int j = 0; j < nn.Outputs.Count; j++)
                {
                    Console.WriteLine(nn.Outputs[j]);
                }
                Console.WriteLine();
            }

            //nn.Export("H:/export.nns");
        }
    }
}