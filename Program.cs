using System;
using System.Collections.Generic;

namespace AI
{
    class Program
    {
        static void Main(string[] args)
        {

            List<int> dimensions = new() { 2, 3, 1 };
            double[] inputs1 = { 0.0, 1.0 };
            double[] inputs2 = { 1.0, 0.0 };
            double[] inputs3 = { 1.0, 1.0 };
            double[] inputs4 = { 0.0, 0.0 };
            var asd2 = new NeuralNetwork(dimensions, ActivationTypes.TANH, 0.5);

            asd2.BackPropagate();

            Console.WriteLine("Outputs");

            asd2.SetInputs(inputs1).Update();
            Console.WriteLine(asd2.GetInputs(0));
            Console.WriteLine(asd2.GetInputs(1));
            Console.WriteLine(asd2.Layers[asd2.Layers.Count - 1].Outputs[0]);
            Console.WriteLine();

            asd2.SetInputs(inputs2).Update();
            Console.WriteLine(asd2.GetInputs(0));
            Console.WriteLine(asd2.GetInputs(1));
            Console.WriteLine(asd2.Layers[asd2.Layers.Count - 1].Outputs[0]);
            Console.WriteLine();

            asd2.SetInputs(inputs3).Update();
            Console.WriteLine(asd2.GetInputs(0));
            Console.WriteLine(asd2.GetInputs(1));
            Console.WriteLine(asd2.Layers[asd2.Layers.Count - 1].Outputs[0]);
            Console.WriteLine();

            asd2.SetInputs(inputs4).Update();
            Console.WriteLine(asd2.GetInputs(0));
            Console.WriteLine(asd2.GetInputs(1));
            Console.WriteLine(asd2.Layers[asd2.Layers.Count - 1].Outputs[0]);
            Console.WriteLine();
        }
    }
}
