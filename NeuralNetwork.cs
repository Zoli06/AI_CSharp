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

        public void BackPropagateOnline(Vector<double>[][] patterns, double learningRate, int batchSize, double targetAccuray, int epoches)
        {
            if (patterns.GetLength(0) % batchSize != 0)
            {
                throw new Exception("Patterns number must be dividable by patternPerEpoch");
            }

            Random rnd = new Random();

            int trainedForPatternsNum = 0;
            double totalAccuray = 0;

            int batchesInEpochNum = patterns.GetLength(0) / batchSize;
            double accuray;

            rnd.Shuffle(patterns);

            for (int smallEpoch = 0; smallEpoch < batchesInEpochNum * epoches; smallEpoch++)
            {
                accuray = BackPropagateForPatterns(patterns, learningRate, trainedForPatternsNum, trainedForPatternsNum + batchSize);
                totalAccuray += accuray;

                trainedForPatternsNum += batchSize;
                if (accuray >= targetAccuray) break;
                if (patterns.GetLength(0) == trainedForPatternsNum)
                {
                    if (totalAccuray / batchesInEpochNum <= targetAccuray) break;

                    Console.WriteLine();
                    Console.WriteLine($"completeEpoch: {smallEpoch / batchesInEpochNum}, accuray: {totalAccuray / batchesInEpochNum}");
                    Console.WriteLine();

                    totalAccuray = 0;
                    trainedForPatternsNum = 0;

                    rnd.Shuffle(patterns);

                    Export($"c:/asd2/export{smallEpoch / batchesInEpochNum}.nns");
                }
                else
                {
                    Console.WriteLine($"smallEpoch: {smallEpoch}, accuray: {accuray}");
                }
            }

            //Console.WriteLine(epoch);
            //Console.WriteLine(totalError);
        }

        public void BackPropagateOffline(Vector<double>[][] patterns, double learningRate, double targetAccuray, int epoches)
        {
            double accuray = 0;

            int epoch;
            for (epoch = 0; epoch < epoches; epoch++)
            {
                accuray = BackPropagateForPatterns(patterns, learningRate);

                Console.WriteLine(epoch);

                if (accuray >= targetAccuray) break;
            }

            //Console.WriteLine();
            Console.WriteLine("Epoch: " + epoch);
            //Console.WriteLine(totalError);
        }

        private double BackPropagateForPatterns(Vector<double>[][] patterns, double learningRate, int fromPattern = 0, int? toPattern = null)
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

            double correctNum = 0;

            int patternNum2 = 0;
            for (int patternNum = fromPattern; patternNum < toPattern; patternNum++, patternNum2++)
            {
                Vector<double> outputs = Update(patterns[patternNum][0]);

                if (outputs.MaximumIndex() == patterns[patternNum][1].MaximumIndex())
                {
                    correctNum++;
                }

                Vector<double> errors = Loss.Default(outputs, patterns[patternNum][1], NeuralNetworkLossType);
                Vector<double> dErrors = Loss.Derivative(outputs, patterns[patternNum][1], NeuralNetworkLossType);

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

            //return errorSum;
            return (double)(correctNum / (toPattern - fromPattern));

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
    }
}