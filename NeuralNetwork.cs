using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
            List<ActivationType> _activationTypes = new List<ActivationType> { ActivationType.LINEAR };

            for (int i = 0; i < structure.Count; i++)
            {
                _activationTypes.Add(activationType);
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

        public void BackPropagateOnline(Vector<double>[,] patterns, double learningRate, int patternPerEpoch, double targetError, ulong maxEpoches = ulong.MaxValue)
        {
            if (patterns.GetLength(0) % patternPerEpoch != 0)
            {
                throw new Exception("Patterns number must be dividable by patternPerEpoch");
            }


            int trainedForPatternsNum = 0;
            double totalError = 0;

            ulong epoch;
            for (epoch = 0; epoch < maxEpoches; epoch++)
            {
                totalError += BackPropagateForPatterns(patterns, learningRate, trainedForPatternsNum, trainedForPatternsNum + patternPerEpoch);

                trainedForPatternsNum += patternPerEpoch;
                if (patterns.GetLength(0) == trainedForPatternsNum)
                {
                    trainedForPatternsNum = 0;
                    if (totalError <= targetError) break;
                    totalError = 0;
                    //Console.WriteLine(epoch);
                }
            }

            //Console.WriteLine(epoch);
            //Console.WriteLine(totalError);
        }

        public void BackPropagateOffline(Vector<double>[,] patterns, double learningRate, double targetError, ulong maxEpoches = ulong.MaxValue)
        {
            double totalError = 0;

            ulong epoch;
            for (epoch = 0; epoch < maxEpoches; epoch++)
            {
                totalError = BackPropagateForPatterns(patterns, learningRate);

                Console.WriteLine(epoch);

                if (totalError <= targetError) break;
            }

            //Console.WriteLine();
            Console.WriteLine("Epoch: " + epoch);
            //Console.WriteLine(totalError);
        }

        private double BackPropagateForPatterns(Vector<double>[,] patterns, double learningRate, int fromPattern = 0, int? toPattern = null)
        {
            toPattern ??= patterns.GetLength(0);

            if (toPattern > patterns.GetLength(0))
            {
                throw new Exception("ToPattern can\' be bigger than patterns num");
            }

            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].DNodes.Clear();
                Layers[i].DWeights.Clear();
                Layers[i].DeltaNodes.Clear();
            }

            double errorSum = 0;

            //Console.WriteLine(fromPattern);
            //Console.WriteLine(toPattern);
            //Console.WriteLine();

            int patternNum2 = 0;
            for (int patternNum = fromPattern; patternNum < toPattern; patternNum++, patternNum2++)
            {
                Vector<double> outputs = Update(patterns[patternNum, 0]);

                Vector<double> errors = Loss.Default(outputs, patterns[patternNum, 1], LossType.SQUAREERROR);
                Vector<double> dErrors = Loss.Derivative(outputs, patterns[patternNum, 1], LossType.SQUAREERROR);

                errorSum += errors.Sum();

                for (int i = Layers.Count - 1; i >= 0; i--)
                {
                    Layers[i].DWeights.Add(_m.Dense(Layers[i].NeuronsNumber, Layers[i].LastLayerNeuronsNumber));
                    Layers[i].DeltaNodes.Add(_v.Dense(Layers[i].NeuronsNumber));

                    if (i == Layers.Count - 1)
                    {
                        Layers[i].DeltaNodes[patternNum2] = dErrors * Layers[i].DNodes[patternNum2];
                    }
                    else if (i != 0)
                    {
                        Layers[i].DeltaNodes[patternNum2] = Layers[i + 1].DeltaNodes[patternNum2] * Layers[i + 1].Weights * Layers[i].DNodes[patternNum2];
                    }

                    if (i != 0)
                    {
                        Layers[i].DWeights[patternNum2] = Layers[i].DeltaNodes[patternNum2].OuterProduct(Layers[i - 1].Outputs);
                    }
                }
            }

            patternNum2 = 0;
            for (int patternNum = fromPattern; patternNum < toPattern; patternNum++, patternNum2++)
            {
                for (int i = 1; i < Layers.Count; i++)
                {
                    Layers[i].Weights -= Layers[i].DWeights[patternNum2] * learningRate;
                    Layers[i].Biases -= Layers[i].DeltaNodes[patternNum2] * learningRate;
                }
            }

            return errorSum;

            //if (epoch % 100 == 0)
            //{
            //    //Console.WriteLine(Outputs[0]);
            //    //Console.WriteLine();
            //}
            //}

            //Console.WriteLine("Finished");
        }

        public void Export(string path)
        {
            var datas = new
            {
                layers = Layers.Select((item, index) => new
                {
                    isInputLayer = index == 0,
                    weights = item.Weights.ToArray(),
                    biases = item.Biases.ToArray(),
                    activationType = item.ActivationType
                })
            };

            File.WriteAllText(path, JsonConvert.SerializeObject(datas));
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

        public static class Loss
        {
            public static Vector<double> Default(Vector<double> outputs, Vector<double> patterns, LossType lossType)
            {
                switch (lossType)
                {
                    case LossType.SQUAREERROR: return LossFunction.SquareError.Default(outputs, patterns);
                    default: throw new NotImplementedException();
                }
            }

            public static Vector<double> Derivatve(Vector<double> outputs, Vector<double> patterns, LossType lossType)
            {
                switch (lossType)
                {
                    case LossType.SQUAREERROR: return LossFunction.SquareError.Derivative(outputs, patterns);
                    default: throw new NotImplementedException();
                }
            }
        }

        public static class LossFunction
        {
            public static class SquareError
            {
                public static Vector<double> Default(Vector<double> outputs, Vector<double> patterns)
                {
                    return (outputs - patterns).PointwisePower(2) / 2;
                }

                public static Vector<double> Derivative(Vector<double> outputs, Vector<double> patterns)
                {
                    return outputs - patterns;
                }
            }
        }

        public enum LossType
        {
            SQUAREERROR
        }

        public class Layer
        {
            protected internal Matrix<double> Weights { get; set; }
            protected internal Vector<double> Biases { get; set; }
            protected internal Vector<double> Outputs { get; set; }
            // Diagonal matrix
            protected internal List<Matrix<double>> DNodes { get; set; }
            protected internal List<Matrix<double>> DWeights { get; set; }
            protected internal List<Vector<double>> DeltaNodes { get; set; }
            protected internal ActivationType ActivationType { get; set; }

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
                Weights = _m.Random(neuronsNumber, lastLayerNeuronsNumber);
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

            public Vector<double> Update(Vector<double> inputs)
            {
                Outputs = _v.Dense(NeuronsNumber);

                DNodes.Add(_m.Diagonal(NeuronsNumber, NeuronsNumber));

                Outputs = Activate.Default(Weights * inputs + Biases, ActivationType);

                DNodes[DNodes.Count - 1] = Activate.Derivative(Outputs, ActivationType);

                return Outputs;
            }

            public static class Activate
            {
                public static Vector<double> Default(Vector<double> value, ActivationType activationType)
                {
                    switch (activationType)
                    {
                        case ActivationType.LINEAR: return Activation.Linear.Default(value);
                        case ActivationType.SIGMOID: return Activation.Sigmoid.Default(value);
                        case ActivationType.TANH: return Activation.TanH.Default(value);
                        case ActivationType.RELU: return Activation.ReLU.Default(value);
                        case ActivationType.SOFTMAX: return Activation.SoftMax.Default(value);
                        default: throw new NotImplementedException();
                    }
                }

                public static Matrix<double> Derivative(Vector<double> value, ActivationType activationType)
                {
                    switch (activationType)
                    {
                        case ActivationType.LINEAR: return Activation.Linear.Derivative(value);
                        case ActivationType.SIGMOID: return Activation.Sigmoid.Derivative(value);
                        case ActivationType.TANH: return Activation.TanH.Derivative(value);
                        case ActivationType.RELU: return Activation.ReLU.Derivative(value);
                        case ActivationType.SOFTMAX: return Activation.SoftMax.Derivative(value);
                        default: throw new NotImplementedException();
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

        public static class Activation
        {
            public static class Linear
            {
                public static Vector<double> Default(Vector<double> value)
                {
                    return value;
                }

                public static Matrix<double> Derivative(Vector<double> value)
                {
                    return _m.DenseDiagonal(value.Count, value.Count, 1.0);
                }
            }

            public static class Sigmoid
            {
                public static Vector<double> Default(Vector<double> value)
                {
                    return 1.0 / (1.0 + value.PointwiseExp());
                }

                public static Matrix<double> Derivative(Vector<double> value)
                {
                    Matrix<double> result = _m.DenseDiagonal(value.Count, value.Count);

                    for (int i = 0; i < value.Count; i++)
                    {
                        result[i, i] = value * (1.0 - value);
                    }

                    return result;
                }
            }

            public static class TanH
            {
                public static Vector<double> Default(Vector<double> value)
                {
                    return value.PointwiseTanh();
                }

                public static Matrix<double> Derivative(Vector<double> value)
                {
                    Matrix<double> result = _m.DenseDiagonal(value.Count, value.Count);

                    for (int i = 0; i < value.Count; i++)
                    {
                        result[i, i] = 1.0 - Math.Pow(Math.Tanh(value[i]), 2);
                    }

                    return result;
                }
            }

            public static class ReLU
            {
                public static Vector<double> Default(Vector<double> value)
                {
                    Vector<double> result = _v.Dense(value.Count);

                    for (int i = 0; i < value.Count; i++)
                    {
                        result[i] = Math.Max(value[i], 0.0);
                    }
                    return result;
                }

                public static Matrix<double> Derivative(Vector<double> value)
                {
                    Matrix<double> result = _m.Dense(value.Count, value.Count);

                    for (int i = 0; i < value.Count; i++)
                    {
                        result[i, i] = value[i] >= 0.0 ? 1.0 : 0.0;
                    }

                    return result;
                }
            }

            public static class SoftMax
            {
                public static Vector<double> Default(Vector<double> value)
                {
                    Vector<double> shiftValue = value - value.Maximum();
                    Vector<double> exps = shiftValue.PointwiseExp();
                    return exps / exps.Sum();
                }

                public static Matrix<double> Derivative(Vector<double> value)
                {
                    //softmax derivate is a matrix:
                    //[i,j] : how much the ith element in the output vector changes if we change the jth element
                    Matrix<double> result = _m.Dense(value.Count, value.Count);

                    Vector<double> softmax = Default(value);

                    for (int i = 0; i < value.Count; i++)
                    {
                        for (int j = 0; j < value.Count; j++)
                        {
                            result[i, j] = softmax[i] * ((i == j).ToInt() - softmax[j]);
                        }
                    }

                    return result;
                }
            }
        }

        public enum ActivationType
        {
            LINEAR,
            SIGMOID,
            TANH,
            RELU,
            SOFTMAX
        }
    }
}