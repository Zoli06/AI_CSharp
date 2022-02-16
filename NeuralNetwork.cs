using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace AI
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        private static MatrixBuilder<double> _m = Matrix<double>.Build;
        private static VectorBuilder<double> _v = Vector<double>.Build;

        public Vector<double> Outputs
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

        public NeuralNetwork(List<int> structure, ActivationType activationType)
        {
            List<ActivationType> _activationTypes = new();

            for (int i = 0; i < structure.Count; i++)
            {
                if (i == 0)
                {
                    _activationTypes.Add(ActivationType.LINEAR);
                }
                else
                {
                    _activationTypes.Add(activationType);
                }
            }

            Build(structure, _activationTypes);
        }

        public NeuralNetwork(List<int> structure, List<ActivationType> activationTypes)
        {
            Build(structure, activationTypes);
        }

        public NeuralNetwork(string path)
        {
            string text = File.ReadAllText(path);

            dynamic a = JsonConvert.DeserializeObject(text);

            List<double[,]> weights = new();
            List<double[]> biases = new();
            List<ActivationType> activationTypes = new();

            for (int i = 0; i < a.layers.Count; i++)
            {
                weights.Add(a.layers[i].weights.ToObject<double[,]>());
                biases.Add(a.layers[i].biases.ToObject<double[]>());
                activationTypes.Add(a.layers[i].activationType.ToObject<ActivationType>());
            }

            Build(weights, biases, activationTypes);
        }

        private void Build(List<int> structure, List<ActivationType> activationTypes)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer(structure[0], 0, activationTypes[0]));

            for (int i = 1; i < structure.Count; i++)
            {
                Layers.Add(new Layer(structure[i], structure[i - 1], activationTypes[i]));
            }
        }

        private void Build(List<double[,]> weights, List<double[]> biases, List<ActivationType> activationTypes)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer(weights[0].GetLength(0), 0, activationTypes[0]));

            for (int i = 1; i < weights.Count; i++)
            {
                Layers.Add(new Layer(weights[i], biases[i], activationTypes[i]));
            }
        }

        public Vector<double> Update(Vector<double> inputs)
        {
            Layers[0].Outputs = inputs;

            // Skip input layer
            for (int i = 1; i < Layers.Count; i++)
            {
                Layers[i].Update(Layers[i - 1].Outputs);
            }

            return Outputs;
        }

        public void BackPropagate(Vector<double>[,] patterns, int epoches, double learningRate)
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
                    Vector<double> outputs = Update(patterns[pattern, 0]);

                    Vector<double> errors = _v.Dense(outputs.Count);
                    Vector<double> dErrors = _v.Dense(outputs.Count);

                    for (int i = 0; i < outputs.Count; i++)
                    {
                        errors[i] = Math.Pow(outputs[i] - patterns[pattern, 1][i], 2) / 2;
                        dErrors[i] = outputs[i] - patterns[pattern, 1][i];
                    }

                    for (int i = Layers.Count - 1; i >= 0; i--)
                    {
                        Layers[i].DWeights.Add(_m.Dense(Layers[i].NeuronsNumber, Layers[i].LastLayerNeuronsNumber));
                        Layers[i].DeltaNodes.Add(_v.Dense(Layers[i].NeuronsNumber));

                        if (i == Layers.Count - 1)
                        {
                            Layers[i].DeltaNodes[pattern] = dErrors * Layers[i].DNodes[pattern];
                        }
                        else if (i != 0)
                        {
                            Layers[i].DeltaNodes[pattern] = Layers[i + 1].DeltaNodes[pattern] * Layers[i + 1].Weights * Layers[i].DNodes[pattern];
                        }

                        if (i != 0)
                        {
                            Layers[i].DWeights[pattern] = Layers[i].DeltaNodes[pattern].OuterProduct(Layers[i - 1].Outputs);
                        }
                    }
                }

                for (int pattern = 0; pattern < patterns.GetLength(0); pattern++)
                {
                    for (int i = 1; i < Layers.Count; i++)
                    {
                        Layers[i].Weights -= Layers[i].DWeights[pattern] * learningRate;
                        Layers[i].Biases -= Layers[i].DeltaNodes[pattern] * learningRate;
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
            public Matrix<double> Weights { get; set; }
            public Vector<double> Biases { get; set; }
            public Vector<double> Outputs { get; set; }
            // Diagonal matrix
            public List<Matrix<double>> DNodes { get; set; }
            public List<Matrix<double>> DWeights { get; set; }
            public List<Vector<double>> DeltaNodes { get; set; }
            public ActivationType ActivationType { get; set; }

            public int NeuronsNumber
            {
                get
                {
                    return Weights.RowCount;
                }
            }
            public int LastLayerNeuronsNumber
            {
                get
                {
                    return Weights.ColumnCount;
                }
            }
            
            public Layer(int neuronsNumber, int lastLayerNeuronsNumber, ActivationType activationType)
            {
                Weights = _m.Random(neuronsNumber, lastLayerNeuronsNumber) * 2 - 1;
                Biases = _v.Dense(neuronsNumber);
                Outputs = _v.Dense(neuronsNumber);
                DNodes = new();
                DWeights = new();
                DeltaNodes = new();

                ActivationType = activationType;
            }

            public Layer(double[,] weights, double[] biases, ActivationType activationType)
            {
                Weights = _m.DenseOfArray(weights);
                Biases = _v.DenseOfArray(biases);
                Outputs = _v.Dense(NeuronsNumber);
                DNodes = new();
                DWeights = new();
                DeltaNodes = new();

                ActivationType = activationType;
            }

            public Vector<double> Update(Vector<double>inputs)
            {
                Outputs = _v.Dense(NeuronsNumber);

                DNodes.Add(_m.Diagonal(NeuronsNumber, NeuronsNumber));

                Outputs = Weights * inputs + Biases;

                for (int i = 0; i < NeuronsNumber; i++)
                {
                    Outputs[i] = Activate(Outputs[i]);

                    DNodes[DNodes.Count - 1][i, i] = Derivative(Outputs[i]);
                }

                return Outputs;
            }

            public double Derivative(double value)
            {
                switch (ActivationType)
                {
                    case ActivationType.LINEAR: return Activation.Linear.Derivative(value);
                    case ActivationType.SIGMOID: return Activation.Sigmoid.Derivative(value);
                    case ActivationType.TANH: return Activation.TanH.Derivative(value);
                    case ActivationType.RELU: return Activation.ReLU.Derivative(value);
                    default: throw new NotImplementedException();
                }
            }

            public double Activate(double value)
            {
                switch (ActivationType)
                {
                    case ActivationType.LINEAR: return Activation.Linear.Default(value);
                    case ActivationType.SIGMOID: return Activation.Sigmoid.Default(value);
                    case ActivationType.TANH: return Activation.TanH.Default(value);
                    case ActivationType.RELU: return Activation.ReLU.Default(value);
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

        public static class Activation
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

        public enum ActivationType
        {
            LINEAR,
            SIGMOID,
            TANH,
            RELU
        }
    }
}