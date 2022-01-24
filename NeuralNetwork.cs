using System;
using System.Collections.Generic;

namespace AI
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        public double LearningRate;

        public double[] GetInputs()
        {
            return Layers[0].Outputs;
        }

        public double GetInputs(int i)
        {
            return Layers[0].Outputs[i];
        }

        public NeuralNetwork SetInputs(double[] inputs)
        {
            Layers[0].Outputs = inputs;

            return this;
        }

        public NeuralNetwork SetInputs(int i, double input)
        {
            Layers[0].Outputs[i] = input;

            return this;
        }

        public NeuralNetwork(List<Layer> layers = null, double learningRate = 1)
        {
            layers ??= new List<Layer>();
            Layers = layers;
            LearningRate = learningRate;
        }

        public NeuralNetwork(List<int> structure, ActivationTypes activationType, double learningRate = 1)
        {
            LearningRate = learningRate;

            List<ActivationTypes> _activationTypes = new();

            for (int i = 0; i < structure.Count; i++)
            {
                if (i == 0)
                {
                    _activationTypes.Add(ActivationTypes.LINEAR);
                }
                else
                {
                    _activationTypes.Add(activationType);
                }
            }

            Build(structure, _activationTypes);
        }

        public NeuralNetwork(List<int> structure, List<ActivationTypes> activationTypes, double learningRate = 1)
        {
            LearningRate = learningRate;

            Build(structure, activationTypes);
        }

        private void Build(List<int> structure, List<ActivationTypes> activationTypes)
        {
            if (activationTypes.Count != structure.Count)
            {
                throw new Exception("First dimension of activationTypes must be equal to the number of layers");
            }

            Layers = new List<Layer>();

            Layers.Add(new Layer(structure[0], 0, activationTypes[0]));

            for (int i = 1; i < structure.Count; i++)
            {
                Layers.Add(new Layer(structure[i], structure[i - 1], activationTypes[i]));
            }
        }

        public double[] Update()
        {
            // Skip input layer
            for (int i = 1; i < Layers.Count; i++)
            {
                Layers[i].Update(Layers[i - 1].Outputs);
            }

            return Layers[Layers.Count - 1].Outputs;
        }

        public NeuralNetwork BackPropagate()
        {
            // TODO: take patterns as argument
            double[,][] patterns =
            {
                { new double[] { 0.0, 1.0 }, new double[] { 1.0 } },
                { new double[] { 1.0, 0.0 }, new double[] { 1.0 } },
                { new double[] { 1.0, 1.0 }, new double[] { 0.0 } },
                { new double[] { 0.0, 0.0 }, new double[] { 0.0 } },
            };

            for (int epoch = 0; epoch < 1000; epoch++)
            {
                for (int i = 0; i < Layers.Count; i++)
                {
                    Layers[i].Derivatives.Clear();
                    Layers[i].WeightsDeltas.Clear();
                    Layers[i].NodesDeltas.Clear();
                }

                for (int pattern = 0; pattern < patterns.Length; pattern++)
                {
                    double[] outputs = SetInputs(patterns[0, 0]).Update();

                    double[] errors = new double[outputs.Length];
                    double[] dErrors = new double[outputs.Length];
                    for (int i = 0; i < outputs.GetLength(0); i++)
                    {
                        errors[i] = Math.Pow(outputs[i] - patterns[i, 1][0], 2) / 2;
                        dErrors[i] = outputs[i] - patterns[i, 1][0];
                    }

                    for (int i = Layers.Count - 1; i >= 0; i--)
                    {
                        //Array.Clear(Layers[i].WeightsDeltas, 0, Layers[i].WeightsDeltas.Length);
                        //Array.Clear(Layers[i].NodesDeltas, 0, Layers[i].NodesDeltas.Length);
                        Layers[i].WeightsDeltas.Add(new double[Layers[i].NeuronsNumber, Layers[i].LastLayerNeuronsNumber]);
                        Layers[i].NodesDeltas.Add(new double[Layers[i].NeuronsNumber]);

                        for (int j = 0; j < Layers[i].NeuronsNumber; j++)
                        {
                            if (i == Layers.Count - 1)
                            {
                                Layers[i].NodesDeltas[pattern][j] = dErrors[j] * Layers[i].Derivatives[pattern][j];
                            }
                            else if (i != 0)
                            {
                                double sum = 0.0;

                                for (int k = 0; k < Layers[i + 1].NeuronsNumber; k++)
                                {
                                    sum += Layers[i + 1].NodesDeltas[pattern][k] * Layers[i + 1].Weights[k, j];
                                }

                                Layers[i].NodesDeltas[pattern][j] = sum * Layers[i].Derivatives[pattern][j];
                            }

                            if (i != 0)
                            {
                                for (int k = 0; k < Layers[i].LastLayerNeuronsNumber; k++)
                                {
                                    Layers[i].WeightsDeltas[pattern][j, k] = Layers[i].NodesDeltas[pattern][j] * Layers[i - 1].Outputs[k];
                                }
                            }
                        }
                    }
                }

                for (int pattern = 0; pattern < patterns.Length; pattern++)
                {
                    for (int i = 1; i < Layers.Count; i++)
                    {
                        for (int j = 0; j < Layers[i].NeuronsNumber; j++)
                        {
                            for (int k = 0; k < Layers[i].LastLayerNeuronsNumber; k++)
                            {
                                Layers[i].Weights[j, k] -= Layers[i].WeightsDeltas[pattern][j, k] * LearningRate;
                            }

                            Layers[i].Biases[j] -= Layers[i].NodesDeltas[pattern][j] * LearningRate;
                        }
                    }
                }

                if (epoch % 100 == 0)
                {
                    //Console.WriteLine(Layers[Layers.Count - 1].Outputs[0]);
                    //Console.WriteLine();
                }
            }

            Console.WriteLine("Finished");

            return this;
        }

        public override string ToString()
        {
            string str = "";

            foreach (Layer layer in Layers)
            {
                str += layer + "\n\n";
            }
            return str;
        }
    }

    public class Layer
    {
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }
        public double[] Outputs { get; set; }
        public List<double[]> Derivatives { get; set; }
        public List<double[,]> WeightsDeltas { get; set; }
        public List<double[]> NodesDeltas { get; set; }
        ActivationTypes ActivationType { get; set; }
        public int NeuronsNumber
        {
            get
            {
                return Weights.GetLength(0);
            }
        }
        public int LastLayerNeuronsNumber
        {
            get
            {
                return Weights.GetLength(1);
            }
        }

        public Layer(int neuronsNumber, int lastLayerNeuronsNumber, ActivationTypes activationType)
        {
            Weights = new double[neuronsNumber, lastLayerNeuronsNumber];
            Biases = new double[neuronsNumber];
            Outputs = new double[neuronsNumber];
            Derivatives = new();
            WeightsDeltas = new();
            NodesDeltas = new();

            Random random = new Random();

            for (int i = 0; i < neuronsNumber; i++)
            {
                Biases[i] = 0.0;
                for (int j = 0; j < lastLayerNeuronsNumber; j++)
                {
                    Weights[i, j] = random.NextDouble();
                }
            }

            ActivationType = activationType;
        }

        public double[] Update(double[] inputs)
        {
            if (inputs.GetLength(0) != LastLayerNeuronsNumber)
            {
                throw new Exception("Inputs number must be equal to LastLayerNeuronsNumber");
            }

            Array.Clear(Outputs, 0, Outputs.Length);
            // Array.Clear(Derivatives, 0, Derivatives.Length);

            Derivatives.Add(new double[NeuronsNumber]);

            for (int i = 0; i < NeuronsNumber; i++)
            {
                Outputs[i] = 0.0;
                for (int j = 0; j < LastLayerNeuronsNumber; j++)
                {
                    Outputs[i] += inputs[j] * Weights[i, j];
                }
                Outputs[i] += Biases[i];

                Outputs[i] = Activate(Outputs[i]);

                Derivatives[Derivatives.Count - 1][i] = Outputs[i] * (1 - Outputs[i]);
            }

            return Outputs;
        }

        public double Activate(double value)
        {
            switch (ActivationType)
            {
                case ActivationTypes.BINARYSTEP: return Activation.BinaryStep.Default(value);
                case ActivationTypes.LINEAR: return Activation.Linear.Default(value);
                case ActivationTypes.SIGMOID: return Activation.Sigmoid.Default(value); ;
                case ActivationTypes.TANH: return Activation.TanH.Default(value);
                case ActivationTypes.RELU: return Activation.ReLU.Default(value);
                default: throw new NotImplementedException();
            }
        }

        public void AddNoise()
        {
            Random random = new Random();

            for (int i = 0; i < NeuronsNumber; i++)
            {
                Biases[i] += random.NextDouble() * 2 - 1;
                for (int j = 0; j < LastLayerNeuronsNumber; j++)
                {
                    Weights[i, j] += random.NextDouble() * 2 - 1;
                }
            }
        }

        public override string ToString()
        {
            string str = "";

            for (int i = 0; i < NeuronsNumber; i++)
            {
                string _weights = "(";
                for (int j = 0; j < LastLayerNeuronsNumber; j++)
                {
                    _weights += $"{Weights[i, j]}, ";
                }
                _weights += ")";

                str += $"({_weights}, {Biases[i]}, {Outputs[i]})\n";
            }

            return str;
        }
    }

    public enum ActivationTypes
    {
        BINARYSTEP,
        LINEAR,
        SIGMOID,
        TANH,
        RELU
    }

    // TODO: implement all derivatives
    public class Activation
    {
        public static class BinaryStep
        {
            public static int Default(double value)
            {
                return Convert.ToInt32(value >= 0.0);
            }

            public static double Derivative(double value)
            {
                throw new NotImplementedException();
            }
        }

        public static class Linear
        {
            public static double Default(double value)
            {
                return value;
            }

            public static double Derivative(double value)
            {
                throw new NotImplementedException();
            }
        }

        public static class Sigmoid
        {
            public static double Default(double value)
            {
                return 1.0 / (1.0 + Math.Exp(-value));
            }

            public static double Derivative(double value)
            {
                return Math.Exp(-value) / Math.Pow(1 + Math.Exp(-value), 2.0);
            }
        }

        public static class TanH
        {
            public static double Default(double value)
            {
                return Math.Tanh(value);
            }

            public static double Derivative(double value)
            {
                throw new NotImplementedException();
            }
        }

        public static class ReLU
        {
            public static double Default(double value)
            {
                return Math.Max(0, value);
            }

            public static double Derivative(double value)
            {
                throw new NotImplementedException();
            }
        }
    }
}