using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using System.IO;

namespace AI
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }

        public double[] Inputs
        {
            get
            {
                return Layers[0].Outputs;
            }
            set
            {
                Layers[0].Outputs = value;
            }
        }

        public double[] Outputs
        {
            get
            {
                return Layers[Layers.Count - 1].Outputs;
            }
            set
            {
                Layers[Layers.Count - 1].Outputs = value;
            }
        }

        public NeuralNetwork(List<Layer> layers = null)
        {
            layers ??= new List<Layer>();
            Layers = layers;
        }

        public NeuralNetwork(List<int> structure, ActivationTypes activationType)
        {
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

        public NeuralNetwork(List<int> structure, List<ActivationTypes> activationTypes)
        {
            Build(structure, activationTypes);
        }

        public NeuralNetwork(string path)
        {
            string text = File.ReadAllText(path);

            dynamic a = JsonConvert.DeserializeObject(text);

            List<double[,]> weights = new();
            List<double[]> biases = new();
            List<ActivationTypes> activationTypes = new();

            for (int i = 0; i < a.layers.Count; i++)
            {
                weights.Add(a.layers[i].weights.ToObject<double[,]>());
                biases.Add(a.layers[i].biases.ToObject<double[]>());
                activationTypes.Add(a.layers[i].activationType.ToObject<ActivationTypes>());
            }

            Build(weights, biases, activationTypes);

            // File.WriteAllTextAsync("H:/export4.nns", JsonConvert.SerializeObject(a));
        }

        private void Build(List<int> structure, List<ActivationTypes> activationTypes)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer(structure[0], 0, activationTypes[0]));

            for (int i = 1; i < structure.Count; i++)
            {
                Layers.Add(new Layer(structure[i], structure[i - 1], activationTypes[i]));
            }
        }

        private void Build(List<double[,]> weights, List<double[]> biases, List<ActivationTypes> activationTypes)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer(weights[0].GetLength(0), 0, activationTypes[0]));

            for (int i = 1; i < weights.Count; i++)
            {
                Layers.Add(new Layer(weights[i], biases[i], activationTypes[i]));
            }
        }

        public double[] Update()
        {
            // Skip input layer
            for (int i = 1; i < Layers.Count; i++)
            {
                Layers[i].Update(Layers[i - 1].Outputs);
            }

            return Outputs;
        }

        public void BackPropagate(double[,][] patterns, int epoches, double learningRate)
        {
            for (int epoch = 0; epoch < epoches; epoch++)
            {
                for (int i = 0; i < Layers.Count; i++)
                {
                    Layers[i].DNodes.Clear();
                    Layers[i].DWeights.Clear();
                    Layers[i].DeltaNodes.Clear();
                }

                for (int pattern = 0; pattern < patterns.GetLength(0); pattern++)
                {
                    Inputs = patterns[pattern, 0];
                    double[] outputs = Update();

                    double[] errors = new double[outputs.Length];
                    double[] dErrors = new double[outputs.Length];

                    for (int i = 0; i < outputs.GetLength(0); i++)
                    {
                        errors[i] = Math.Pow(outputs[i] - patterns[pattern, 1][i], 2) / 2;
                        dErrors[i] = outputs[i] - patterns[pattern, 1][i];
                    }

                    for (int i = Layers.Count - 1; i >= 0; i--)
                    {
                        Layers[i].DWeights.Add(new double[Layers[i].NeuronsNumber, Layers[i].LastLayerNeuronsNumber]);
                        Layers[i].DeltaNodes.Add(new double[Layers[i].NeuronsNumber]);

                        for (int j = 0; j < Layers[i].NeuronsNumber; j++)
                        {
                            if (i == Layers.Count - 1)
                            {
                                Layers[i].DeltaNodes[pattern][j] = dErrors[j] * Layers[i].DNodes[pattern][j];
                            }
                            else if (i != 0)
                            {
                                double sum = 0.0;

                                for (int k = 0; k < Layers[i + 1].NeuronsNumber; k++)
                                {
                                    sum += Layers[i + 1].DeltaNodes[pattern][k] * Layers[i + 1].Weights[k, j];
                                }

                                Layers[i].DeltaNodes[pattern][j] = sum * Layers[i].DNodes[pattern][j];
                            }

                            if (i != 0)
                            {
                                for (int k = 0; k < Layers[i].LastLayerNeuronsNumber; k++)
                                {
                                    Layers[i].DWeights[pattern][j, k] = Layers[i].DeltaNodes[pattern][j] * Layers[i - 1].Outputs[k];
                                }
                            }
                        }
                    }
                }

                for (int pattern = 0; pattern < patterns.GetLength(0); pattern++)
                {
                    for (int i = 1; i < Layers.Count; i++)
                    {
                        for (int j = 0; j < Layers[i].NeuronsNumber; j++)
                        {
                            for (int k = 0; k < Layers[i].LastLayerNeuronsNumber; k++)
                            {
                                Layers[i].Weights[j, k] -= Layers[i].DWeights[pattern][j, k] * learningRate;
                            }

                            Layers[i].Biases[j] -= Layers[i].DeltaNodes[pattern][j] * learningRate;
                        }
                    }
                }

                if (epoch % 100 == 0)
                {
                    //Console.WriteLine(Outputs[0]);
                    //Console.WriteLine();
                }
            }

            Console.WriteLine("Finished");
        }

        public void Export(string path)
        {
            var datas = new
            {
                layers = Layers.Select((item, index) => new
                {
                    isInputLayer = index == 0,
                    weights = item.Weights,
                    biases = item.Biases,
                    activationType = item.ActivationType
                })
            };

            File.WriteAllTextAsync(path, JsonConvert.SerializeObject(datas));
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

        public class Layer
        {
            public double[,] Weights { get; set; }
            public double[] Biases { get; set; }
            public double[] Outputs { get; set; }
            public List<double[]> DNodes { get; set; }
            public List<double[,]> DWeights { get; set; }
            public List<double[]> DeltaNodes { get; set; }
            public ActivationTypes ActivationType { get; set; }
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
                DNodes = new();
                DWeights = new();
                DeltaNodes = new();

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

            public Layer(double[,] weights, double[] biases, ActivationTypes activationType)
            {
                Weights = new double[weights.GetLength(0), weights.GetLength(1)];
                Biases = new double[weights.GetLength(0)];
                Outputs = new double[weights.GetLength(0)];
                DNodes = new();
                DWeights = new();
                DeltaNodes = new();

                Weights = weights;
                Biases = biases;
                ActivationType = activationType;
            }

            public double[] Update(double[] inputs)
            {
                if (inputs.GetLength(0) != LastLayerNeuronsNumber)
                {
                    throw new Exception("Inputs number must be equal to LastLayerNeuronsNumber");
                }

                Array.Clear(Outputs, 0, Outputs.Length);

                DNodes.Add(new double[NeuronsNumber]);

                for (int i = 0; i < NeuronsNumber; i++)
                {
                    Outputs[i] = 0.0;
                    for (int j = 0; j < LastLayerNeuronsNumber; j++)
                    {
                        Outputs[i] += inputs[j] * Weights[i, j];
                    }
                    Outputs[i] += Biases[i];

                    Outputs[i] = Activate(Outputs[i]);

                    DNodes[DNodes.Count - 1][i] = Derivative(Outputs[i]);
                }

                return Outputs;
            }

            public double Derivative(double value)
            {
                switch (ActivationType)
                {
                    case ActivationTypes.LINEAR: return Activation.Linear.Derivative(value);
                    case ActivationTypes.SIGMOID: return Activation.Sigmoid.Derivative(value);
                    case ActivationTypes.TANH: return Activation.TanH.Derivative(value);
                    case ActivationTypes.RELU: return Activation.ReLU.Derivative(value);
                    default: throw new NotImplementedException();
                }
            }

            public double Activate(double value)
            {
                switch (ActivationType)
                {
                    case ActivationTypes.LINEAR: return Activation.Linear.Default(value);
                    case ActivationTypes.SIGMOID: return Activation.Sigmoid.Default(value);
                    case ActivationTypes.TANH: return Activation.TanH.Default(value);
                    case ActivationTypes.RELU: return Activation.ReLU.Default(value);
                    default: throw new NotImplementedException();
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

        public class Activation
        {
            public static class Linear
            {
                public static double Default(double value)
                {
                    return value;
                }

                public static double Derivative(double value)
                {
                    return 1.0;
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
                    return value * (1.0 - value);
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
                    return 1.0 - Math.Pow(Math.Tanh(value), 2);
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
                    if (value >= 0.0)
                    {
                        return 1.0;
                    }
                    else
                    {
                        return 0.0;
                    }
                }
            }
        }

        public enum ActivationTypes
        {
            LINEAR,
            SIGMOID,
            TANH,
            RELU
        }
    }
}