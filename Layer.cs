using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace AI
{
    public partial class NeuralNetwork
    {
        public class Layer
        {
            protected internal Matrix<double> Weights { get; set; }
            protected internal Vector<double> Biases { get; set; }
            protected internal Vector<double> Outputs { get; set; }
            // Diagonal matrix
            protected internal List<Matrix<double>> DNodes { get; set; }
            protected internal List<Matrix<double>> DWeights { get; set; }
            protected internal List<Vector<double>> DeltaNodes { get; set; }
            protected internal ActivationType LayerActivationType { get; set; }

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

            public Layer(int neuronsNumber, int lastLayerNeuronsNumber, ActivationType activationType, Normal? weightsDistribution = null, Normal? biasesDistribution = null)
            {
                Random rnd = new Random();
                weightsDistribution ??= new Normal(0, 1) { RandomSource = rnd };
                biasesDistribution ??= new Normal(0, 0) { RandomSource = rnd };
                Weights = _m.Random(neuronsNumber, lastLayerNeuronsNumber, weightsDistribution);
                Biases = _v.Random(neuronsNumber, biasesDistribution);
                Outputs = _v.Dense(neuronsNumber);
                DNodes = new();
                DWeights = new();
                DeltaNodes = new();

                LayerActivationType = activationType;
            }

            public Layer(double[,] weights, double[] biases, ActivationType activationType)
            {
                Weights = _m.DenseOfArray(weights);
                Biases = _v.DenseOfArray(biases);
                Outputs = _v.Dense(NeuronsNumber);
                DNodes = new();
                DWeights = new();
                DeltaNodes = new();

                LayerActivationType = activationType;
            }

            public Vector<double> Update(Vector<double> inputs)
            {
                Outputs = _v.Dense(NeuronsNumber);

                DNodes.Add(_m.Dense(NeuronsNumber, NeuronsNumber));

                Outputs = Activate.Default(Weights * inputs + Biases, LayerActivationType);

                DNodes[DNodes.Count - 1] = Activate.Derivative(Outputs, LayerActivationType);

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
                        return 1.0 / (1.0 + (-value).PointwiseExp());
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
                            result[i, i] = value[i] > 0.0 ? 1.0 : 0.0;
                        }

                        return result;
                    }
                }

                public static class SoftMax
                {
                    public static Vector<double> Default(Vector<double> value)
                    {
                        //Stable
                        Vector<double> numerator = (value - value.Maximum()).PointwiseExp();
                        double denominator = numerator.Sum();

                        Vector<double> result = numerator / denominator;

                        return result;
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
}
