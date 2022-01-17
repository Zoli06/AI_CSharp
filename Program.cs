using System;
using System.Collections.Generic;

namespace AI
{
    class Program
    {
        static void Main(string[] args)
        {
            List<int> dimensions = new() { 2, 3, 1 };
            List<double> inputs = new() { 0, 1 };
            var asd2 = new NeuralNetwork(dimensions, ActivationTypes.TANH, 0.1);
            asd2.BackPropagate().Update();
            Console.WriteLine(asd2.ToString());
            //asd2.BackPropagation();
            //asd2.SetInputs(new() { 0, 1 });
            //asd2.Calculate();
            //Console.WriteLine(asd2.ToString());
            //asd2.SetInputs(new() { 1, 0 });
            //asd2.Calculate();
            //Console.WriteLine(asd2.ToString());
            //asd2.SetInputs(new() { 0, 0 });
            //asd2.Calculate();
            //Console.WriteLine(asd2.ToString());
            //asd2.SetInputs(new() { 1, 1 });
            //asd2.Calculate();
            //Console.WriteLine(asd2.ToString());
        }
    }
}