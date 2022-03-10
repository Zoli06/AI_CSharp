using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace AI
{
    public partial class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        public LossType NeuralNetworkLossType { get; set; }
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

        public NeuralNetwork(List<int> structure, Layer.ActivationType activationType, LossType neuralNetworkLossType)
        {
            List<Layer.ActivationType> _activationTypes = new List<Layer.ActivationType> { Layer.ActivationType.LINEAR };

            for (int i = 0; i < structure.Count; i++)
            {
                _activationTypes.Add(activationType);
            }

            Build(structure, _activationTypes, neuralNetworkLossType);
        }

        public NeuralNetwork(List<int> structure, List<Layer.ActivationType> activationTypes, LossType neuralNetworkLossType)
        {
            Build(structure, activationTypes, neuralNetworkLossType);
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

            // TODO: make it changeable
            Build(weights, biases, activationTypes, LossType.SQUAREERROR);
        }

        private void Build(List<int> structure, List<Layer.ActivationType> activationTypes, LossType neuralNetworkLossType)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer(structure[0], 0, activationTypes[0]));

            for (int i = 1; i < structure.Count; i++)
            {
                Layers.Add(new Layer(structure[i], structure[i - 1], activationTypes[i]));
            }

            NeuralNetworkLossType = neuralNetworkLossType;
        }

        private void Build(List<double[,]> weights, List<double[]> biases, List<Layer.ActivationType> activationTypes, LossType neuralNetworkLossType)
        {
            Layers = new List<Layer>();

            Layers.Add(new Layer(weights[0].GetLength(0), 0, activationTypes[0]));

            for (int i = 1; i < weights.Count; i++)
            {
                Layers.Add(new Layer(weights[i], biases[i], activationTypes[i]));
            }

            NeuralNetworkLossType = neuralNetworkLossType;
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

        public void BackPropagateOnline(Vector<double>[][] patterns, double learningRate, int batchSize, Metrics metrics, double target, int epoches)
        {
            Random rnd = new Random();

            int trainedForPatternsNum = 0;
            double totalAccuracyOrError = 0;
            double accuracyOrError;

            int batchesInEpochNum = patterns.GetLength(0) / batchSize;

            rnd.Shuffle(patterns);

            for (int smallEpoch = 0; smallEpoch < batchesInEpochNum * epoches; smallEpoch++)
            {
                accuracyOrError = BackPropagateForPatterns(patterns, learningRate, metrics, trainedForPatternsNum, trainedForPatternsNum + batchSize);

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

                if (patterns.GetLength(0) == trainedForPatternsNum)
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

        public void BackPropagateOffline(Vector<double>[][] patterns, double learningRate, Metrics metrics, double target, int epoches)
        {
            double accuracyOrError;

            int epoch;
            for (epoch = 0; epoch < epoches; epoch++)
            {
                accuracyOrError = BackPropagateForPatterns(patterns, learningRate, metrics);

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

        private double BackPropagateForPatterns(Vector<double>[][] patterns, double learningRate, Metrics metrics, int fromPattern = 0, int? toPattern = null)
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
            double correctNum = 0;

            int patternNum2 = 0;
            for (int patternNum = fromPattern; patternNum < toPattern; patternNum++, patternNum2++)
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
                    errorSum += Loss.Default(outputs, patterns[patternNum][1], NeuralNetworkLossType).Sum();
                }

                Vector<double> dErrors = Loss.Derivative(outputs, patterns[patternNum][1], NeuralNetworkLossType);

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

            if (metrics == Metrics.ACCURACY)
            {
                return (double)(correctNum / (toPattern - fromPattern));
            }
            else if (metrics == Metrics.ERROR)
            {
                return (double)(errorSum / (toPattern - fromPattern));
            }

            throw new Exception();
        }

        public double GetAccuracy(Vector<double>[][] patterns, int fromPattern = 0, int? toPattern = null)
        {
            // One-hot encoding only

            toPattern ??= patterns.GetLength(0);

            int correctNum = 0;
            for (int patternNum = fromPattern; patternNum < toPattern; patternNum++)
            {
                if (Update(patterns[patternNum][0]).MaximumIndex() == patterns[patternNum][1].MaximumIndex())
                {
                    correctNum++;
                }
            }

            return (double)(correctNum / (toPattern - fromPattern));
        }

        public double GetError(Vector<double>[][] patterns, int fromPattern = 0, int? toPattern = null)
        {
            toPattern ??= patterns.GetLength(0);

            double error = 0;
            for (int patternNum = fromPattern; patternNum < toPattern; patternNum++)
            {
                error += Loss.Default(Update(patterns[patternNum][0]), patterns[patternNum][1], NeuralNetworkLossType).Sum();
            }

            return (double)(error / (toPattern - fromPattern));
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