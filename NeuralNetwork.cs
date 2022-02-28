using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AI
{
    public partial class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        public static MatrixBuilder<double> _m = Matrix<double>.Build;
        public static VectorBuilder<double> _v = Vector<double>.Build;

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

        public NeuralNetwork(List<int> structure, Layer.ActivationType activationType)
        {
            List<Layer.ActivationType> _activationTypes = new List<Layer.ActivationType> { Layer.ActivationType.LINEAR };

            for (int i = 0; i < structure.Count; i++)
            {
                _activationTypes.Add(activationType);
            }

            Build(structure, _activationTypes);
        }

        public NeuralNetwork(List<int> structure, List<Layer.ActivationType> activationTypes)
        {
            Build(structure, activationTypes);
        }

        public NeuralNetwork(string path)
        {
            string text = File.ReadAllText(path);

            dynamic a = JsonConvert.DeserializeObject(text);

            List<double[,]> weights = new();
            List<double[]> biases = new();
            List<Layer.ActivationType> activationTypes = new();

            for (int i = 0; i < a.layers.Count; i++)
            {
                weights.Add(a.layers[i].weights.ToObject<double[,]>());
                biases.Add(a.layers[i].biases.ToObject<double[]>());
                activationTypes.Add(a.layers[i].activationType.ToObject<Layer.ActivationType>());
            }

            Build(weights, biases, activationTypes);
        }

        private void Build(List<int> structure, List<Layer.ActivationType> activationTypes)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer(structure[0], 0, activationTypes[0]));

            for (int i = 1; i < structure.Count; i++)
            {
                Layers.Add(new Layer(structure[i], structure[i - 1], activationTypes[i]));
            }
        }

        private void Build(List<double[,]> weights, List<double[]> biases, List<Layer.ActivationType> activationTypes)
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
                    activationType = item.LayerActivationType
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

            public static Vector<double> Derivative(Vector<double> outputs, Vector<double> patterns, LossType lossType)
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
    }
}