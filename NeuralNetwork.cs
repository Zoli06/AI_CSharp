using MathNet.Numerics.Distributions;
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
        private List<Layer> Layers;
        private static MatrixBuilder<double> _m = Matrix<double>.Build;
        private static VectorBuilder<double> _v = Vector<double>.Build;

        public Vector<double> Outputs
        {
            get
            {
                return Layers[Layers.Count - 1].Outputs;
            }
        }

        public NeuralNetwork(List<Layer> layers)
        {
            Layers = layers;
        }

        public NeuralNetwork(List<uint> structure, Layer.ActivationType activationType, Normal? weightsDistribution = null, Normal? biasesDistribution = null)
        {
            List<Layer.ActivationType> _activationTypes = new() { Layer.ActivationType.LINEAR };

            for (int i = 0; i < structure.Count; i++)
            {
                _activationTypes.Add(activationType);
            }

            List<Normal?> _weightsDistribution = new() { null };
            for (int i = 1; i < structure.Count; i++)
            {
                _weightsDistribution.Add(weightsDistribution);
            }

            List<Normal?> _biasesDistribution = new() { null };
            for (int i = 1; i < structure.Count; i++)
            {
                _biasesDistribution.Add(biasesDistribution);
            }

            Build(structure, _activationTypes, _weightsDistribution, _biasesDistribution);
        }

        public NeuralNetwork(List<uint> structure, List<Layer.ActivationType> activationTypes, List<Normal?>? weightsDistribution = null, List<Normal?>? biasesDistribution = null)
        {
            if (weightsDistribution == null)
            {
                weightsDistribution = new();

                for (int i = 0; i < structure.Count; i++)
                {
                    weightsDistribution.Add(null);
                }
            }

            if (biasesDistribution == null)
            {
                biasesDistribution = new();

                for (int i = 0; i < structure.Count; i++)
                {
                    biasesDistribution.Add(null);
                }
            }

            Build(structure, activationTypes, weightsDistribution, biasesDistribution);
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

        private void Build(List<uint> structure, List<Layer.ActivationType> activationTypes, List<Normal?> weightsDistribution, List<Normal?> biasesDistribution)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer(structure[0], 0, activationTypes[0]));

            for (int i = 1; i < structure.Count; i++)
            {
                Layers.Add(new Layer(structure[i], structure[i - 1], activationTypes[i], weightsDistribution[i], biasesDistribution[i]));
            }
        }

        private void Build(List<double[,]> weights, List<double[]> biases, List<Layer.ActivationType> activationTypes)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer((uint)weights[0].GetLength(0), 0, activationTypes[0]));

            for (int i = 1; i < weights.Count; i++)
            {
                if (weights[i].GetLength(0) != biases[i].Length) throw new Exception("Neurons in weights must be equal to neurons in biases");

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

        public void BackPropagateOnline(Vector<double>[][] patterns, double learningRate, uint batchSize, LossType lossType, Metrics metrics, double target, uint epoches)
        {
            if (learningRate <= 0) throw new Exception("LearingRate must be greater than zero");

            if (batchSize == 0) throw new Exception("BatchSize can\'t be zero");

            if (patterns.Length % batchSize != 0) throw new Exception("Patterns number must be dividable by batchSize");

            if (target <= 0) throw new Exception("Target must be greater than zero");

            Random rnd = new Random();

            uint trainedForPatternsNum = 0;
            double totalAccuracyOrError = 0;
            double accuracyOrError;

            // Always dividable
            int batchesInEpochNum = (int)(patterns.Length / batchSize);

            rnd.Shuffle(patterns);

            for (int smallEpoch = 0; smallEpoch < batchesInEpochNum * epoches; smallEpoch++)
            {
                accuracyOrError = BackPropagateForPatterns(patterns, learningRate, lossType, metrics, trainedForPatternsNum, trainedForPatternsNum + batchSize);

                trainedForPatternsNum += batchSize;

                if (metrics == Metrics.ACCURACY)
                {
                    if (accuracyOrError >= target) break;
                }
                else if (metrics == Metrics.ERROR)
                {
                    if (accuracyOrError <= target) break;
                }

                totalAccuracyOrError += accuracyOrError;

                if (patterns.Length == trainedForPatternsNum)
                {
                    if (metrics == Metrics.ACCURACY)
                    {
                        if (totalAccuracyOrError / batchesInEpochNum >= target) break;
                    }
                    else if (metrics == Metrics.ERROR)
                    {
                        if (totalAccuracyOrError / batchesInEpochNum <= target) break;
                    }

                    Console.WriteLine();
                    Console.WriteLine($"completeEpoch: {smallEpoch / batchesInEpochNum}, accuracyOrError: {totalAccuracyOrError / batchesInEpochNum}");
                    Console.WriteLine();

                    totalAccuracyOrError = 0;
                    trainedForPatternsNum = 0;

                    rnd.Shuffle(patterns);

                    Export($"c:/asd2/export{smallEpoch / batchesInEpochNum}.nns");
                }
                else
                {
                    Console.WriteLine($"smallEpoch: {smallEpoch}, accuracyOrError: {accuracyOrError}");
                }
            }

            //Console.WriteLine(epoch);
            //Console.WriteLine(totalError);
        }

        public void BackPropagateOffline(Vector<double>[][] patterns, double learningRate, LossType lossType, Metrics metrics, double target, uint epoches)
        {
            if (learningRate <= 0) throw new Exception("LearingRate must be greater than zero");

            if (target <= 0) throw new Exception("Target must be greater than zero");

            double accuracyOrError;

            int epoch;
            for (epoch = 0; epoch < epoches; epoch++)
            {
                accuracyOrError = BackPropagateForPatterns(patterns, learningRate, lossType, metrics);

                if (metrics == Metrics.ACCURACY)
                {
                    if (accuracyOrError >= target) break;
                }
                else if (metrics == Metrics.ERROR)
                {
                    if (accuracyOrError <= target) break;
                }
            }

            //Console.WriteLine();
            Console.WriteLine("Epoch: " + epoch);
            //Console.WriteLine(totalError);
        }

        private double BackPropagateForPatterns(Vector<double>[][] patterns, double learningRate, LossType lossType, Metrics metrics, uint fromPattern = 0, uint? toPattern = null)
        {
            toPattern ??= (uint)patterns.Length;

            if (learningRate <= 0) throw new Exception("LearingRate must be greater than zero");

            if (toPattern > patterns.Length) throw new Exception("ToPattern can\'t be bigger than patternsNum");

            if (fromPattern >= toPattern) throw new Exception("ToPattern must be greater than fromPattern");

            Clear();

            double errorSum = 0;
            double correctNum = 0;

            int patternNum2 = 0;
            for (int patternNum = (int)fromPattern; patternNum < toPattern; patternNum++, patternNum2++)
            {
                Vector<double> outputs = Update(patterns[patternNum][0]);

                // Unused
                // Vector<double> errors = Loss.Default(outputs, patterns[patternNum][1], NeuralNetworkLossType);

                // Doesn't call GetAccuracy due to insane memory usage
                if (metrics == Metrics.ACCURACY)
                {
                    if (outputs.MaximumIndex() == patterns[patternNum][1].MaximumIndex())
                    {
                        correctNum++;
                    }
                }
                else if (metrics == Metrics.ERROR)
                {
                    errorSum += Loss.Default(outputs, patterns[patternNum][1], lossType).Sum();
                }

                Vector<double> dErrors = Loss.Derivative(outputs, patterns[patternNum][1], lossType);

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
            for (uint patternNum = fromPattern; patternNum < toPattern; patternNum++, patternNum2++)
            {
                for (int i = 1; i < Layers.Count; i++)
                {
                    Layers[i].Weights -= Layers[i].DWeights[patternNum2] * learningRate;
                    Layers[i].Biases -= Layers[i].DeltaNodes[patternNum2] * learningRate;
                }
            }

            if (metrics == Metrics.ACCURACY)
            {
                return (double)(correctNum / (toPattern - fromPattern));
            }
            else if (metrics == Metrics.ERROR)
            {
                return (double)(errorSum / (toPattern - fromPattern));
            }

            throw new NotImplementedException();
        }

        public double GetAccuracy(Vector<double>[][] patterns, uint fromPattern = 0, uint? toPattern = null)
        {
            // One-hot encoding only

            toPattern ??= (uint)patterns.Length;

            if (toPattern > patterns.Length) throw new Exception("ToPattern can\'t be bigger than patternsNum");

            if (fromPattern >= toPattern) throw new Exception("ToPattern must be greater than fromPattern");

            Clear();

            int correctNum = 0;
            for (int patternNum = (int)fromPattern; patternNum < toPattern; patternNum++)
            {
                if (Update(patterns[patternNum][0]).MaximumIndex() == patterns[patternNum][1].MaximumIndex())
                {
                    correctNum++;
                }

                if (patternNum % 10000 == 0) Console.WriteLine(patternNum);

                Clear();
            }

            return (double)(correctNum / (toPattern - fromPattern));
        }

        public double GetError(Vector<double>[][] patterns, LossType lossType, int fromPattern = 0, int? toPattern = null)
        {
            toPattern ??= patterns.Length;

            if (toPattern > patterns.Length) throw new Exception("ToPattern can\'t be bigger than patternsNum");

            if (fromPattern >= toPattern) throw new Exception("ToPattern must be greater than fromPattern");

            Clear();

            double error = 0;
            for (int patternNum = fromPattern; patternNum < toPattern; patternNum++)
            {
                error += Loss.Default(Update(patterns[patternNum][0]), patterns[patternNum][1], lossType).Sum();

                Clear();
            }

            return (double)(error / (toPattern - fromPattern));
        }

        public void Clear()
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].DNodes.Clear();
                Layers[i].DWeights.Clear();
                Layers[i].DeltaNodes.Clear();
            }
        }

        public void Export(string path)
        {
            var datas = new
            {
                layers = Layers.Select((item, index) => new
                {
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
                    case LossType.CROSSENTROPY: return LossFunction.CrossEntropy.Default(outputs, patterns);
                    default: throw new NotImplementedException();
                }
            }

            public static Vector<double> Derivative(Vector<double> outputs, Vector<double> patterns, LossType lossType)
            {
                switch (lossType)
                {
                    case LossType.SQUAREERROR: return LossFunction.SquareError.Derivative(outputs, patterns);
                    case LossType.CROSSENTROPY: return LossFunction.CrossEntropy.Derivative(outputs, patterns);
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

            public static class CrossEntropy
            {
                public static Vector<double> Default(Vector<double> outputs, Vector<double> patterns)
                {
                    Vector<double> result = _v.Dense(outputs.Count);

                    for (int i = 0; i < outputs.Count; i++)
                    {
                        result[i] = patterns[i] * Math.Log(Math.Max(outputs[i], 0.000000001));
                    }

                    return -result;
                }

                public static Vector<double> Derivative(Vector<double> outputs, Vector<double> patterns)
                {
                    Vector<double> result = _v.Dense(outputs.Count);

                    for (int i = 0; i < outputs.Count; i++)
                    {
                        result[i] = patterns[i] / Math.Max(outputs[i], 0.000000001);
                    }

                    return -result;
                }
            }
        }

        public enum LossType
        {
            SQUAREERROR,
            CROSSENTROPY
        }

        public enum Metrics
        {
            ERROR,
            ACCURACY
        }
    }
}